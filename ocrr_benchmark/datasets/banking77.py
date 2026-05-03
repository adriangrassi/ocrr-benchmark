"""Banking77 dataset loader.

Adapted from Cortex's ``cortex.text.banking77``. Same interface, same source
(PolyAI MIT-licensed CSVs auto-downloaded on first use), simplified to drop
the cortex logger dependency.

77 fine-grained banking intents, ~10k train / ~3k test examples.
Reference: Casanueva et al. 2020.
"""

from __future__ import annotations

import csv
import os
import urllib.request
from dataclasses import dataclass


DEFAULT_DIR = os.path.join("data", "banking77")
TRAIN_URL = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
    "master/banking_data/train.csv"
)
TEST_URL = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
    "master/banking_data/test.csv"
)


@dataclass
class Banking77Example:
    text: str
    label: str


# Generic alias used by the eval harness — every dataset returns objects
# that quack like Banking77Example.
DatasetExample = Banking77Example


def _download(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    print(f"  [banking77] downloading {url}", flush=True)
    with urllib.request.urlopen(url, timeout=30) as r, open(dest, "wb") as f:
        f.write(r.read())


def _parse_csv(path: str) -> list[Banking77Example]:
    out: list[Banking77Example] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = (row.get("text") or "").strip()
            label = (row.get("category") or row.get("label") or "").strip()
            if text and label:
                out.append(Banking77Example(text=text, label=label))
    return out


def load_banking77(
    data_dir: str = DEFAULT_DIR,
    download: bool = True,
) -> tuple[list[Banking77Example], list[Banking77Example], list[str]]:
    """Return ``(train, test, labels_sorted)``."""
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    if not os.path.exists(train_path):
        if not download:
            raise FileNotFoundError(
                f"{train_path} missing — run with download=True or place the files manually."
            )
        _download(TRAIN_URL, train_path)
    if not os.path.exists(test_path):
        if not download:
            raise FileNotFoundError(f"{test_path} missing")
        _download(TEST_URL, test_path)

    train = _parse_csv(train_path)
    test = _parse_csv(test_path)
    labels = sorted({ex.label for ex in train})
    return train, test, labels
