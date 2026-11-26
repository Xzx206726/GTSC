"""Train the fixed GTSC recipe and write the graph, projectors, and LoRA weights."""

import argparse
from pathlib import Path

import torch
from torch import nn

from data import DATA, ROOT, load_train, negative_samples, read_json, training_batches
from model import InvocationGraph, Ranker, RoleProjectors, graph_loss


def adamw(groups, recipe):
    return torch.optim.AdamW(groups, betas=tuple(recipe["adam_betas"]), eps=recipe["adam_eps"],
                             weight_decay=recipe["weight_decay"])


def llm_backward(ranker, dataset, banks, batch, negatives, recipe, generator):
    examples = [(m, h, a, True) for m, h, a in batch]
    examples += [(m, h, a, False) for (m, h, _), ids in zip(batch, negatives) for a in ids]
    # Release each LM forward graph after backward, then replay the token gradients once.
    leaves = [tensor.detach().requires_grad_() for tensor in banks]
    with torch.no_grad():
        examples.sort(key=lambda item: len(ranker.prompt(dataset, leaves, *item[:3]).embeddings))
    size = recipe["microbatch_size"]
    for start in range(0, len(examples), size):
        chunk = examples[start:start + size]
        losses = [ranker.training_loss(ranker.prompt(dataset, leaves, m, h, a, generator), yes,
                                       recipe["identity_weight"]) for m, h, a, yes in chunk]
        (torch.stack(losses).mean() * (len(chunk) / len(examples))).backward()
    used = [(bank, leaf.grad) for bank, leaf in zip(banks, leaves) if leaf.grad is not None]
    torch.autograd.backward([bank for bank, _ in used], [gradient for _, gradient in used])


def fit(dataset, recipe, device, llama, out):
    torch.manual_seed(recipe["seed"])
    ranker = Ranker(recipe, device, llama)
    ranker.model.train()
    graph = InvocationGraph(dataset, device)
    nodes = nn.Parameter(torch.randn(graph.n_nodes, recipe["graph_dim"], device=device))
    projectors = RoleProjectors(recipe, ranker.embed.weight.shape[1]).to(device)
    adapters = {name: parameter for name, parameter in ranker.model.named_parameters() if parameter.requires_grad}
    lora = {"params": list(adapters.values()), "lr": recipe["lora_lr"]}
    parameters = [nodes, *projectors.parameters(), *adapters.values()]
    optimizer = adamw([{"params": [nodes], "lr": recipe["graph_lr"]},
                       {"params": list(projectors.parameters()), "lr": recipe["projector_lr"]}, lora], recipe)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    steps = {"warmup": 0, "joint": 0}
    for phase, epochs, offset in (("warmup", recipe["warmup_epochs"], 0),
                                  ("joint", recipe["joint_epochs"], 100_000)):
        for epoch in range(epochs):
            batches, rng = training_batches(dataset, recipe["batch_size"], recipe["seed"] + offset + epoch)
            generator = torch.Generator().manual_seed(recipe["seed"] + 200_000 + epoch)
            for batch in batches:
                optimizer.zero_grad(set_to_none=True)
                propagated = graph.propagate(nodes, batch)
                negatives = negative_samples(dataset, batch, recipe["negatives"], rng)
                graph_loss(propagated, graph, batch, negatives, recipe["temperature"]).backward()
                if phase == "joint":
                    optimizer.param_groups[-1]["lr"] = (
                        recipe["lora_lr"] if steps[phase] < recipe["lora_decay_after"] else recipe["lora_final_lr"]
                    )
                    llm_backward(ranker, dataset, projectors(propagated, graph.n_mashups),
                                 batch, negatives, recipe, generator)
                torch.nn.utils.clip_grad_norm_(parameters, recipe["gradient_clip"], error_if_nonfinite=True)
                optimizer.step()
                steps[phase] += 1
                if steps[phase] % 100 == 0:
                    print(f"{phase} step {steps[phase]}", flush=True)
            print(f"{phase} epoch {epoch + 1} complete, {steps[phase]} total steps", flush=True)
    if recipe.get("paper_steps") and steps != recipe["paper_steps"]:
        raise ValueError(f"Expected {recipe['paper_steps']} training updates, got {steps}")
    result = out / "checkpoint.pt"
    torch.save({"recipe": recipe, "train_hashes": dataset.train_hashes, "steps": steps,
                "nodes": nodes.detach().cpu(), "projectors": projectors.cpu().state_dict(),
                "lora": {name: parameter.detach().cpu() for name, parameter in adapters.items()}}, result)
    print(result, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--llama", help="Local copy of the paper backbone")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "gtsc")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    recipe = read_json(ROOT / "recipe.json")
    fit(load_train(args.data), recipe, args.device, args.llama or recipe["llama"], args.out)


if __name__ == "__main__":
    main()
