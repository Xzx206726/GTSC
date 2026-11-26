"""Run a small CPU training, retrieval, and evaluation example without downloads."""

import csv
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from data import ROOT, load_test, load_train, read_json
from evaluate import measure, rerank
from retrieve import fit_retriever
from train import fit


def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    with TemporaryDirectory() as directory:
        root = Path(directory)
        for filename, kind, names in (("Mashup_desc.csv", "mashup", ["A", "B", "C"]),
                                      ("API_desc.csv", "api", ["Maps", "Weather", "Photo", "Mail", "Pay", "Music"])):
            with (root / filename).open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow([f"{kind}_id", "description", f"{kind}_name"])
                writer.writerows((i, "Shows Maps and tracks travel." if kind == "mashup" else f"Offers {name}.", name)
                                 for i, name in enumerate(names))
        (root / "train.txt").write_text("0 0 1 2\n1 3\n2 4 5\n", encoding="utf-8")
        (root / "test.txt").write_text("0 3\n1 4\n2 0\n", encoding="utf-8")
        words = ["[UNK]", "[BOS]", "[EOS]", "Yes", "No", "Mashup", "API", "information", "."]
        tokenizer = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        model_path = root / "tiny_llama"
        fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", bos_token="[BOS]",
                                       eos_token="[EOS]", pad_token="[EOS]")
        fast.save_pretrained(model_path)
        config = LlamaConfig(vocab_size=len(words), hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                             num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256)
        LlamaForCausalLM(config).save_pretrained(model_path)

        recipe = read_json(ROOT / "recipe.json")
        recipe.update(dtype="float32", graph_dim=8, projector_hidden=16, max_context=256, top_k=3,
                      eval_bucket_size=32, batch_size=2, warmup_epochs=1, lora_decay_after=2, paper_steps=None)
        dataset = load_train(root)
        assert dataset.mashups[0][1] == "Shows [API] and tracks travel."
        initial = torch.nn.functional.normalize(torch.randn(9, 8), dim=-1)
        pool = fit_retriever(dataset, initial, recipe, "cpu")
        checkpoint = fit(dataset, recipe, "cpu", str(model_path), root / "fit")
        rankings = rerank(dataset, pool, checkpoint, "cpu", str(model_path))
        metrics = measure(rankings, load_test(root, dataset), recipe["metrics_at"], recipe["top_k"])
        assert metrics["evaluated_mashups"] == 3 and metrics["test_edges"] == 3
        assert all(0 <= value <= 1 for key, value in metrics.items() if "@" in key)
        print("Smoke test passed", metrics)


if __name__ == "__main__":
    main()
