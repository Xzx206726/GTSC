"""Rerank the fixed candidate pool or recompute metrics from saved ranked IDs."""

import argparse
import math
from collections import defaultdict
from pathlib import Path

import torch

from data import DATA, PAPER_POOL, ROOT, load_test, load_train, read_json, write_json
from model import InvocationGraph, Ranker, RoleProjectors


def verify_pool(dataset, pool, top_k):
    if set(pool) != {str(m) for m in range(len(dataset.mashups))}:
        raise ValueError("The candidate pool must cover the entire Mashup catalog")
    for m, history in enumerate(dataset.history):
        row = pool[str(m)]
        ids = row["retrieved_ids"]
        if (set(row) != {"selected_ids", "retrieved_ids", "retrieval_scores"}
                or sorted(row["selected_ids"]) != sorted(history)
                or len(ids) != min(top_k, len(dataset.apis) - len(history))
                or len(ids) != len(set(ids)) or set(ids) & set(history)
                or any(not isinstance(a, int) or not 0 <= a < len(dataset.apis) for a in ids)
                or len(row["retrieval_scores"]) != len(ids)
                or any(not math.isfinite(s) for s in row["retrieval_scores"])):
            raise ValueError(f"Invalid candidate pool for Mashup {m}")


@torch.inference_mode()
def rerank(dataset, pool, checkpoint, device, llama=None):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    recipe = saved["recipe"]
    if saved["train_hashes"] != dataset.train_hashes:
        raise ValueError("The checkpoint was fitted on a different catalog or training split")
    if recipe.get("paper_steps") and recipe["paper_steps"] != saved["steps"]:
        raise ValueError("Expected a complete checkpoint")
    verify_pool(dataset, pool, recipe["top_k"])
    ranker = Ranker(recipe, device, llama or recipe["llama"])
    parameters = {name: parameter for name, parameter in ranker.model.named_parameters() if parameter.requires_grad}
    if set(parameters) != set(saved["lora"]):
        raise ValueError("The model and checkpoint LoRA parameters differ")
    for name, parameter in parameters.items():
        parameter.copy_(saved["lora"][name])
    ranker.model.eval()
    graph = InvocationGraph(dataset, device)
    projectors = RoleProjectors(recipe, ranker.embed.weight.shape[1]).to(device)
    projectors.load_state_dict(saved["projectors"])
    banks = projectors(graph.propagate(saved["nodes"].to(device)), graph.n_mashups)
    ranked = {}
    width, microbatch = recipe["eval_bucket_size"], recipe["microbatch_size"]
    for m, history in enumerate(dataset.history):
        row = pool[str(m)]
        buckets = defaultdict(list)
        for a in row["retrieved_ids"]:
            prompt = ranker.prompt(dataset, banks, m, history, a)
            length = min(recipe["max_context"], math.ceil(len(prompt.embeddings) / width) * width)
            buckets[length].append((a, prompt))
        scores = {}
        for length, candidates in sorted(buckets.items()):
            for start in range(0, len(candidates), microbatch):
                chunk = candidates[start:start + microbatch]
                prompts = [prompt for _, prompt in chunk]
                # Match the paper's fixed-shape BF16 evaluation on final short chunks.
                prompts.extend([prompts[-1]] * (microbatch - len(prompts)))
                scores.update((a, score) for (a, _), score in zip(chunk, ranker.score(prompts, length)))
        ranked[str(m)] = row | {"reranked_ids": sorted(scores, key=lambda a: (-scores[a], a)),
                                "llm_scores": scores}
        if (m + 1) % 100 == 0:
            print(f"Scored {m + 1} Mashups", flush=True)
    return ranked


def measure(rankings, targets, cutoffs, top_k):
    sums = defaultdict(list)
    for m, held_out in enumerate(targets):
        row = rankings[str(m)]
        ranked, pool = row["reranked_ids"], row["retrieved_ids"]
        if len(ranked) != len(pool) or set(ranked) != set(pool):
            raise ValueError(f"Ranking is not a permutation of the candidates for Mashup {m}")
        truth = set(held_out)
        sums[f"candidate_recall@{top_k}"].append(len(truth.intersection(pool)) / len(truth))
        for k in cutoffs:
            ranks = [i for i, a in enumerate(ranked[:k], start=1) if a in truth]
            dcg = math.fsum(1 / math.log2(i + 1) for i in ranks)
            ideal = math.fsum(1 / math.log2(i + 1) for i in range(1, min(k, len(truth)) + 1))
            sums[f"recall@{k}"].append(len(ranks) / len(truth))
            sums[f"ndcg@{k}"].append(dcg / ideal)
    return {"evaluated_mashups": len(targets), "test_edges": sum(map(len, targets)),
            **{key: math.fsum(values) / len(targets) for key, values in sums.items()}}


def join_ranked_ids(ids, pool):
    if ids.keys() != pool.keys():
        raise ValueError("Archived query IDs and candidate pool differ")
    return {m: row | {"reranked_ids": ids[m]} for m, row in pool.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--pool", type=Path, default=PAPER_POOL)
    parser.add_argument("--llama", help="Local copy of the paper backbone")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "test")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--checkpoint", type=Path, default=ROOT / "runs" / "gtsc" / "checkpoint.pt")
    source.add_argument("--ranked-ids", type=Path, help="Recompute metrics without loading Llama")
    args = parser.parse_args()
    dataset = load_train(args.data)
    recipe = read_json(ROOT / "recipe.json")
    pool = read_json(args.pool)
    verify_pool(dataset, pool, recipe["top_k"])
    if args.ranked_ids:
        rankings = join_ranked_ids(read_json(args.ranked_ids), pool)
    else:
        rankings = rerank(dataset, pool, args.checkpoint, args.device, args.llama)
    # Held-out labels are opened only after the candidates have been ranked.
    metrics = measure(rankings, load_test(args.data, dataset), recipe["metrics_at"], recipe["top_k"])
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "rankings.json", rankings)
    write_json(args.out / "metrics.json", metrics)
    print(metrics)


if __name__ == "__main__":
    main()
