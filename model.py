"""One-layer LightGCN, role projectors, and the GTSC language-model objective."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


INSTRUCTION = (
    "Given the Mashup requirement and the selected API set, decide whether the "
    "candidate API should be added to the current set. Answer Yes or No.\n"
    "For each entity, a graph soft token comes first. State which kind of "
    "collaborative information it represents, then read the entity's name and "
    "description. The graph token encodes historical interaction information, "
    "not names, descriptions, functionality, or textual content. Judge the role "
    "from the graph token, not from the text that follows.\n"
)
ANSWER_CUE = "\nShould the candidate API be added to the current set?\n"
ROLES = ("Mashup", "selected API", "candidate API")


class InvocationGraph:
    def __init__(self, dataset, device):
        self.n_mashups = len(dataset.mashups)
        self.n_nodes = self.n_mashups + len(dataset.apis)
        pairs = sorted((m, self.n_mashups + a) for m, a in dataset.edges)
        self.pairs = torch.tensor(pairs, dtype=torch.long, device=device).T.contiguous()
        self.keys = self.pairs[0] * self.n_nodes + self.pairs[1]

    def propagate(self, nodes, removed=()):
        pairs = self.pairs
        if removed:
            keys = self.keys.new_tensor([m * self.n_nodes + self.n_mashups + a for m, _, a in removed])
            pairs = pairs[:, ~torch.isin(self.keys, keys)]
        src = torch.cat((pairs[0], pairs[1]))
        dst = torch.cat((pairs[1], pairs[0]))
        degree = torch.bincount(src, minlength=self.n_nodes).float()
        scale = (degree[src] * degree[dst]).rsqrt().to(nodes.dtype)
        neighbors = torch.zeros_like(nodes).index_add_(0, dst, nodes[src] * scale[:, None])
        return (nodes + neighbors) * 0.5


def conditioned_query(normalized_nodes, mashup, selected, n_mashups):
    query = normalized_nodes[mashup]
    if selected:
        ids = torch.as_tensor(selected, device=query.device) + n_mashups
        query = query + normalized_nodes[ids].mean(dim=0)
    return F.normalize(query, dim=0)


def graph_loss(nodes, graph, batch, negative_ids, temperature, *, use_selected=False):
    vectors = F.normalize(nodes, dim=-1)
    if use_selected:
        normalized = F.normalize(vectors, dim=-1)
        query = torch.stack([conditioned_query(normalized, m, h, graph.n_mashups)
                             for m, h, _ in batch])
    else:
        query = F.normalize(vectors[[m for m, _, _ in batch]], dim=-1)
    positives = vectors[[graph.n_mashups + a for _, _, a in batch]]
    negatives = vectors[torch.as_tensor(negative_ids, device=nodes.device) + graph.n_mashups]
    query, positives, negatives = (F.normalize(t, dim=-1) for t in (query, positives, negatives))
    positive_logits = (query * positives).sum(dim=-1, keepdim=True)
    negative_logits = torch.einsum("bd,bnd->bn", query, negatives)
    logits = torch.cat((positive_logits, negative_logits), dim=1) / temperature
    return F.cross_entropy(logits, torch.zeros(len(batch), device=nodes.device, dtype=torch.long))


class RoleProjectors(nn.Module):
    def __init__(self, recipe, hidden_size):
        super().__init__()
        graph_dim, hidden = recipe["graph_dim"], recipe["projector_hidden"]

        def mlp():
            return nn.Sequential(nn.Linear(graph_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden_size))

        self.mashup = mlp()
        self.api = mlp()
        self.selected = mlp()

    def forward(self, nodes, n_mashups):
        # The LLM losses train the projectors. The graph loss alone trains the node table.
        nodes = nodes.detach().unsqueeze(1)
        return (self.mashup(nodes[:n_mashups]), self.selected(nodes[n_mashups:]), self.api(nodes[n_mashups:]))


def fair_prefix_lengths(lengths, budget):
    allocated = [0] * len(lengths)
    remaining = min(budget, sum(lengths))
    active = [i for i, length in enumerate(lengths) if length]
    while remaining and active:
        share = max(1, remaining // len(active))
        for i in active:
            take = min(share, lengths[i] - allocated[i], remaining)
            allocated[i] += take
            remaining -= take
            if not remaining:
                break
        active = [i for i in active if allocated[i] < lengths[i]]
    return allocated


@dataclass
class Prompt:
    embeddings: torch.Tensor
    role_positions: list
    role_ids: list


class Ranker:
    def __init__(self, recipe, device, llama):
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = torch.device(device)
        dtype = getattr(torch, recipe["dtype"])
        if dtype == torch.bfloat16 and self.device.type != "cuda":
            raise ValueError("The paper recipe uses a CUDA device with BF16 support. Run smoke.py on CPU.")
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise ValueError("The selected GPU must support BF16")
        revision = recipe["llama_revision"] if llama == recipe["llama"] else None
        if revision is None and llama != recipe["llama"] and not Path(llama).is_dir():
            raise FileNotFoundError(f"Expected a local model directory: {llama}")
        self.tokenizer = AutoTokenizer.from_pretrained(llama, revision=revision, use_fast=True)
        base = AutoModelForCausalLM.from_pretrained(
            llama, revision=revision, dtype=dtype, use_safetensors=True, low_cpu_mem_usage=True,
        )
        base.config.use_cache = False
        base.requires_grad_(False)
        self.model = get_peft_model(base, LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=recipe["lora_rank"], lora_alpha=recipe["lora_alpha"],
            lora_dropout=recipe["lora_dropout"], target_modules=recipe["lora_modules"], bias="none",
        ))
        if recipe["gradient_checkpointing"]:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model.to(self.device)
        self.embed = self.model.get_input_embeddings()
        self.max_context = recipe["max_context"]
        yes, no = self.encode("Yes"), self.encode("No")
        if len(yes) != 1 or len(no) != 1 or yes == no:
            raise ValueError("The tokenizer must give Yes and No distinct single-token IDs")
        self.yes, self.no = yes[0], no[0]
        self.role_ids = [self.encode(f"This is {role} collaborative information.") for role in ROLES]

    @lru_cache(maxsize=32768)
    def encode(self, text):
        return tuple(self.tokenizer.encode(text, add_special_tokens=False))

    def prompt(self, dataset, banks, mashup, selected, candidate, generator=None):
        if generator is not None and len(selected) > 1:
            order = torch.randperm(len(selected), generator=generator).tolist()
            selected = [selected[i] for i in order]
        else:
            selected = sorted(selected)
        name, desc = dataset.mashups[mashup]
        mashup_ids = self.encode(f"Mashup: {name}\nDescription: {desc}\n")
        name, desc = dataset.apis[candidate]
        candidate_ids = self.encode(f"Candidate API: {name}\nDescription: {desc}\n")
        api_prefixes = [self.encode(f"Selected API: {dataset.apis[a][0]}\nDescription: ") for a in selected]
        api_descriptions = [self.encode(dataset.apis[a][1]) for a in selected]
        api_suffix = self.encode("\n")
        bos = self.tokenizer.bos_token_id
        intro = (() if bos is None else (bos,)) + self.encode(INSTRUCTION)
        answer = self.encode(ANSWER_CUE)
        budget = self.max_context - (len(intro) + len(mashup_ids) + len(candidate_ids) + len(answer)
                                    + sum(len(ids) + len(api_suffix) for ids in api_prefixes)
                                    + (len(selected) + 2) * (1 + max(map(len, self.role_ids))))
        if budget < 0:
            raise ValueError("Context cannot preserve every name, role phrase, and graph token")
        allocations = fair_prefix_lengths(list(map(len, api_descriptions)), budget)
        chunks, positions, targets = [], [], []
        length = 0

        def add(ids):
            nonlocal length
            if ids:
                chunks.append(self.embed(torch.tensor(ids, device=self.device)))
                length += len(ids)

        def entity(role, node_id, ids):
            nonlocal length
            chunks.append(banks[role][node_id].to(self.embed.weight.dtype))
            length += 1
            phrase = self.role_ids[role]
            positions.extend(range(length - 1, length - 1 + len(phrase)))
            targets.extend(phrase)
            add(phrase)
            add(ids)

        add(intro)
        entity(0, mashup, mashup_ids)
        for a, prefix, desc, size in zip(selected, api_prefixes, api_descriptions, allocations):
            entity(1, a, prefix + desc[:size] + api_suffix)
        entity(2, candidate, candidate_ids)
        add(answer)
        return Prompt(torch.cat(chunks), positions, targets)

    def training_loss(self, prompt, positive, identity_weight):
        size = len(prompt.embeddings)
        positions = torch.tensor(prompt.role_positions + [size - 1], device=self.device)
        logits = self.model(inputs_embeds=prompt.embeddings.unsqueeze(0),
                            attention_mask=torch.ones((1, size), device=self.device, dtype=torch.long),
                            use_cache=False, logits_to_keep=positions).logits[0]
        role_loss = F.cross_entropy(logits[:-1], torch.tensor(prompt.role_ids, device=self.device))
        answer_loss = F.cross_entropy(logits[-1:], torch.tensor([self.yes if positive else self.no],
                                                               device=self.device))
        return answer_loss + identity_weight * role_loss

    def score(self, prompts, fixed_length):
        hidden = prompts[0].embeddings.shape[-1]
        embeds = prompts[0].embeddings.new_zeros(len(prompts), fixed_length, hidden)
        mask = torch.zeros((len(prompts), fixed_length), device=self.device, dtype=torch.long)
        for i, prompt in enumerate(prompts):
            size = len(prompt.embeddings)
            embeds[i, -size:] = prompt.embeddings
            mask[i, -size:] = 1
        logits = self.model(inputs_embeds=embeds, attention_mask=mask, use_cache=False,
                            logits_to_keep=1).logits[:, -1]
        return (logits[:, self.yes] - logits[:, self.no]).float().tolist()
