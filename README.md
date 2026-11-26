# GTSC

GTSC aligns graph representations with an LLM to recommend Web APIs for a Mashup and its selected APIs. This release implements the masked-text experiment reported in the paper.

## Install

Use Python 3.11.

```bash
pip install -r requirements.txt
```

## CPU checks

```bash
python smoke.py
python evaluate.py --ranked-ids reference/ranked_ids.json.gz --out runs/audit
```

The smoke test uses synthetic data. The second command recomputes the paper metrics from saved rankings without loading Llama. The expected NDCG@10 and Recall@10 are **0.5569** and **0.7207**.

## Train and evaluate

Obtain access to [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct). The paper run used one A100 with BF16.

```bash
hf auth login
python train.py
python evaluate.py
```

Training writes `runs/gtsc/checkpoint.pt`. Evaluation uses the bundled Top-100 pool and writes metrics and rankings to `runs/test/`. The model revision and hyperparameters are in [recipe.json](recipe.json). Run `python retrieve.py` to build a new pool and pass `--pool runs/top100.json` to the evaluator.

The bundled [TSSGCF data split](https://github.com/IntelligentServiceLab/TSSGCF/tree/3b4f1b4e22ea988ac7878ef336fefcf39a639b58/Data) uses the [upstream license](data/LICENSE-TSSGCF.txt). The GTSC code license is in [LICENSE](LICENSE).
