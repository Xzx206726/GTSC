"""Catalog loading, name masking, and leave-one-invocation-out batches."""

import csv
import gzip
import hashlib
import html
import json
import random
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
PAPER_POOL = DATA / "top100.json.gz"
PAPER_DATA_SHA256 = {
    "API_desc.csv": "686171ad5b6cf5f59d57cf2cf2ce3e47dd40fb4e6084a6fe66fe28131cf20bd8",
    "Mashup_desc.csv": "5c9fd1cc5e2a6f85db1b56d530d16392b55497f6cbb8ea7441fd49cbcea4fb1d",
    "train.txt": "54653127b22d1ad86949b43f093826f25d54f86f08a9c5763dec1b60124d0272",
    "test.txt": "45dc2ae4ef0383e3cea43a26b4a4c1fc7264dcb1c38cafc4d8ff267ee48335d6",
}


def read_json(path):
    path = Path(path)
    open_text = gzip.open if path.suffix == ".gz" else open
    with open_text(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", html.unescape(text)).split())


def name_masker(api_names):
    names = sorted(set(api_names), key=lambda name: (-len(name), name.casefold(), name))
    choices = "|".join(re.escape(name) for name in names)
    # The bounds handle punctuation in names and keep short names out of words.
    pattern = re.compile(rf"(?<![^\W_])(?:{choices})(?![^\W_])", re.IGNORECASE)
    return lambda text: pattern.sub("[API]", normalized(text))


def read_catalog(path, kind):
    id_col, name_col = f"{kind}_id", f"{kind}_name"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != [id_col, "description", name_col]:
            raise ValueError(f"Unexpected CSV header in {path.name}")
        rows = sorted((int(row[id_col]), normalized(row[name_col]),
                       normalized(row["description"])) for row in reader)
    if (not rows or [i for i, _, _ in rows] != list(range(len(rows)))
            or any(not (name and desc) for _, name, desc in rows)):
        raise ValueError(f"Invalid IDs or empty text in {path.name}")
    return tuple((name, desc) for _, name, desc in rows)


def read_interactions(path, n_mashups, n_apis):
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        mashup, *apis = map(int, line.split())
        if (mashup in rows or not apis or len(apis) != len(set(apis))
                or any(not 0 <= api < n_apis for api in apis)):
            raise ValueError(f"Invalid invocation for Mashup {mashup} in {path.name}")
        rows[mashup] = tuple(apis)
    if sorted(rows) != list(range(n_mashups)):
        raise ValueError(f"Expected one row per Mashup in {path.name}")
    return tuple(rows[i] for i in range(n_mashups))


@dataclass(frozen=True)
class Dataset:
    mashups: tuple
    apis: tuple
    history: tuple
    train_hashes: dict

    @property
    def edges(self):
        return [(m, a) for m, apis in enumerate(self.history) for a in apis]


def load_train(folder=DATA):
    folder = Path(folder)
    apis = read_catalog(folder / "API_desc.csv", "api")
    mashups = read_catalog(folder / "Mashup_desc.csv", "mashup")
    masker = name_masker([name for name, _ in apis])
    masked = tuple((masker(name), masker(desc)) for name, desc in mashups)
    history = read_interactions(folder / "train.txt", len(masked), len(apis))
    hashes = {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
              for name in ("API_desc.csv", "Mashup_desc.csv", "train.txt")}
    if folder.resolve() == DATA and any(PAPER_DATA_SHA256[name] != value for name, value in hashes.items()):
        raise ValueError("The bundled catalog or training split differs from the paper snapshot")
    return Dataset(masked, apis, history, hashes)


def load_test(folder, dataset):
    path = Path(folder) / "test.txt"
    if path.parent.resolve() == DATA and hashlib.sha256(path.read_bytes()).hexdigest() != PAPER_DATA_SHA256[path.name]:
        raise ValueError("The bundled test split differs from the paper snapshot")
    targets = read_interactions(path, len(dataset.mashups), len(dataset.apis))
    if any(set(held_out) & set(seen) for held_out, seen in zip(targets, dataset.history)):
        raise ValueError("A held-out invocation also appears in training")
    return targets


def training_batches(dataset, batch_size, seed):
    rng = random.Random(seed)
    edges = dataset.edges
    rng.shuffle(edges)
    batches, batch, seen = [], [], set()
    for m, a in edges:
        if len(batch) == batch_size or m in seen:
            batches.append(batch)
            batch, seen = [], set()
        selected = tuple(sorted(b for b in dataset.history[m] if b != a))
        batch.append((m, selected, a))
        seen.add(m)
    if batch:
        batches.append(batch)
    return batches, rng


def negative_samples(dataset, batch, count, rng):
    samples = []
    for m, _, _ in batch:
        observed = set(dataset.history[m])
        samples.append(rng.sample([a for a in range(len(dataset.apis)) if a not in observed], count))
    return samples
