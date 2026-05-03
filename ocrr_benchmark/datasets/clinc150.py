"""CLINC150 dataset loader (via HuggingFace ``datasets``).

CLINC150 is a 150-class cross-domain intent classification benchmark
(Larson et al. 2019). The ``plus`` configuration includes the full label
set; we use ``train``/``test`` splits.

Returns the same `Banking77Example` dataclass shape used by the rest of
the eval harness, so the continual-stream harness can iterate over
heterogeneous tasks with one type.
"""

from __future__ import annotations

from ocrr_benchmark.datasets.banking77 import Banking77Example


def load_clinc150() -> tuple[
    list[Banking77Example], list[Banking77Example], list[str]
]:
    """Return (train, test, labels_sorted)."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "load_clinc150 requires the `datasets` package "
            "(installed as a transitive dep via sentence-transformers)."
        ) from e

    ds_train = load_dataset("clinc_oos", "plus", split="train")
    ds_test = load_dataset("clinc_oos", "plus", split="test")

    # The intent column is an integer index into the feature's class list.
    intent_names = ds_train.features["intent"].names

    def _to_examples(ds) -> list[Banking77Example]:
        out: list[Banking77Example] = []
        for row in ds:
            text = (row.get("text") or "").strip()
            intent_idx = row.get("intent")
            if not text or intent_idx is None:
                continue
            label = intent_names[int(intent_idx)]
            out.append(Banking77Example(text=text, label=label))
        return out

    train = _to_examples(ds_train)
    test = _to_examples(ds_test)
    labels = sorted({ex.label for ex in train})
    return train, test, labels
