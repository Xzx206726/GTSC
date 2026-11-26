# GTSC

Graph Token Set Completion recommends candidate Web APIs for a Mashup and its selected API set. The code implements the masked-text Llama-3.1-8B-Instruct run reported in the paper.

## Install

Use Python 3.11. The paper run used one A100 with BF16 and PyTorch 2.7.1 with CUDA 12.6.

```bash
pip install -r requirements.txt
```

## Check the release on CPU

```bash
python smoke.py
python evaluate.py --ranked-ids reference/ranked_ids.json.gz --out runs/audit
```

`smoke.py` creates a tiny Llama and synthetic catalog locally, then fits the retriever and reranker and evaluates the result. The second command recomputes the paper metrics from the archived GTSC orderings on 2,289 queries. It writes metrics and rankings without loading Llama. Expected values appear in `reference/metrics.json`.

| Metric | @5 | @10 | @20 |
|---|---:|---:|---:|
| NDCG | 0.5310 | 0.5569 | 0.5738 |
| Recall | 0.6411 | 0.7207 | 0.7854 |

The frozen pool's candidate Recall@100 is 0.8498.

## Train and evaluate

Obtain access to [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) and sign in using `hf auth login`. The model revision is pinned in `recipe.json`. An existing copy of this revision can be passed with `--llama /path/to/llama` to both scripts.

```bash
python train.py
python evaluate.py
```

- `train.py` fits a fresh model on `data/train.txt` and writes the node table, three projectors, and LoRA parameters to `runs/gtsc/checkpoint.pt`.
- `evaluate.py` reads the checkpoint and `data/top100.json.gz`, reranks the candidates, and writes `runs/test/metrics.json` and `runs/test/rankings.json`. The rankings include the selected API IDs, retrieved IDs and scores, reranked IDs, and Yes-minus-No logits.
- `recipe.json` fixes the seed, model revision, temperature, optimizer, 2,657 warmup and 442 joint steps, learning-rate schedule, prompt context limit, and batch sizes. Use `--out` to give another run a new directory.

The training loop removes the batch's target edges from the invocation graph. The graph contrastive objective updates the node table. The language-model objectives update the role projectors and LoRA adapters. The test split supplies labels for scoring the saved orderings.

## Data and candidate generation

The bundled `API_desc.csv`, `Mashup_desc.csv`, `train.txt`, and `test.txt` match the [TSSGCF data release](https://github.com/IntelligentServiceLab/TSSGCF/tree/3b4f1b4e22ea988ac7878ef336fefcf39a639b58/Data). Its license is in `data/LICENSE-TSSGCF.txt`. The tables identify entities by consecutive zero-based IDs. Each interaction row contains a Mashup ID followed by its API IDs. The loader masks catalog API names in Mashup titles and descriptions. API descriptions and titles are retained.

`data/top100.json.gz` is the paper's train-only candidate pool. Use it to match the reported ranking comparison. To fit the E5-initialized LightGCN retriever and generate another pool from the training data, run:

```bash
python retrieve.py
python evaluate.py --pool runs/top100.json --out runs/test_regenerated_pool
```

This fits the retriever with its settings in `recipe.json`. Model kernels and floating-point rounding can affect regenerated scores and ranking ties. The archived pool and ranked IDs let readers reproduce the reported metric calculation exactly.

## Files

`model.py` defines LightGCN, the role projectors, and the LLM prompt and losses. `data.py` loads the catalog and builds the training batches. `train.py`, `retrieve.py`, and `evaluate.py` are the runnable entry points. The GTSC source is released under `LICENSE`.
