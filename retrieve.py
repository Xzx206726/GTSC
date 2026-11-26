"""Fit the independent E5-initialized LightGCN and generate a train-only pool."""

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from data import DATA, ROOT, load_train, negative_samples, normalized, read_json, training_batches, write_json
from model import InvocationGraph, conditioned_query, graph_loss
from train import adamw


def fit_retriever(dataset, initial, recipe, device):
    cfg = recipe["retriever"]
    graph = InvocationGraph(dataset, device)
    nodes = torch.nn.Parameter(initial.to(device).clone())
    optimizer = adamw([{"params": [nodes], "lr": cfg["learning_rate"]}], recipe)
    for epoch in range(cfg["epochs"]):
        batches, rng = training_batches(dataset, cfg["batch_size"], recipe["seed"] + 100_000 + epoch)
        for batch in batches:
            optimizer.zero_grad(set_to_none=True)
            negatives = negative_samples(dataset, batch, recipe["negatives"], rng)
            loss = graph_loss(graph.propagate(nodes, batch), graph, batch, negatives,
                              recipe["temperature"], use_selected=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([nodes], recipe["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
        print(f"Retriever epoch {epoch + 1} complete", flush=True)
    with torch.no_grad():
        vectors = F.normalize(F.normalize(graph.propagate(nodes), dim=-1), dim=-1)
        pool = {}
        for m, history in enumerate(dataset.history):
            query = conditioned_query(vectors, m, history, graph.n_mashups)
            scores = vectors[graph.n_mashups:] @ query
            scores[list(history)] = -torch.inf
            values, ids = scores.topk(min(recipe["top_k"], len(dataset.apis) - len(history)))
            pool[str(m)] = {"selected_ids": list(history), "retrieved_ids": ids.tolist(),
                            "retrieval_scores": values.tolist()}
    return pool


def main():
    from sentence_transformers import SentenceTransformer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "top100.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--e5", help="Local E5 snapshot directory")
    args = parser.parse_args()
    recipe = read_json(ROOT / "recipe.json")
    cfg = recipe["retriever"]
    dataset = load_train(args.data)
    model = SentenceTransformer(args.e5 or cfg["model"], revision=None if args.e5 else cfg["revision"],
                                device=args.device)
    texts = [normalized(f"{cfg['prefix']} {name}. {desc}") for name, desc in dataset.mashups + dataset.apis]
    initial = model.encode(texts, batch_size=cfg["embed_batch_size"], convert_to_tensor=True,
                           normalize_embeddings=False, show_progress_bar=True)
    initial = F.normalize(initial.detach().cpu().float(), dim=-1)
    del model
    pool = fit_retriever(dataset, initial, recipe, args.device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, pool)
    print(args.out)


if __name__ == "__main__":
    main()
