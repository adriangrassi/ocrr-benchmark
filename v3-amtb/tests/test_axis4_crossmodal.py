"""Tests for AMTB Axis 4 (cross-modal) — framework only.

The framework is testable without real CLIP/CLAP models by using a
synthetic typed-store: a system that maps `(entry_id, content,
modality)` triples and retrieves by exact-content equality on a
declared cross-modality pairing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from amtb.axes import crossmodal
from amtb.types import AxisName


@dataclass
class _SyntheticTypedStore:
    """Trivial typed store: maps content equality across paired modalities."""

    storage: dict[tuple[str, str], str] = field(default_factory=dict)
    # For cross-modal demo: map text↔image by parallel keys ("img_for_X")
    pairs: dict[str, str] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return "synthetic_typed_store"

    def supports(self, axis):
        return axis == AxisName.CROSS_MODAL

    def clear(self):
        self.storage.clear()
        self.pairs.clear()

    def ingest_typed(self, entry_id, content, modality):
        self.storage[(entry_id, modality)] = content

    def query_topk_typed(self, query, query_modality, target_modality, k):
        # For test purposes: return all entries from `target_modality`
        # in arbitrary order. A real system would do similarity search.
        out = [eid for (eid, m) in self.storage if m == target_modality]
        return out[:k]


def _synthetic_loader():
    def loader(name, limit):
        # Synthetic dataset: 4 (text, image_id) pairs, query is text →
        # gold is image_id of same index.
        entries = [
            ("img_0", "<image data 0>", "image"),
            ("img_1", "<image data 1>", "image"),
            ("img_2", "<image data 2>", "image"),
            ("img_3", "<image data 3>", "image"),
        ]
        queries = [
            {"query": "caption 0", "query_modality": "text",
             "target_modality": "image", "gold_ids": ["img_0"]},
            {"query": "caption 1", "query_modality": "text",
             "target_modality": "image", "gold_ids": ["img_1"]},
        ]
        return {"entries": entries, "queries": queries}
    return loader


def test_axis4_synthetic_runs():
    sys = _SyntheticTypedStore()
    config = crossmodal.CrossModalEvalConfig(
        datasets=({"name": "synth", "modality": "text↔image", "n_pairs": 4},),
        k=4,
    )
    r = crossmodal.run(sys, config=config, dataset_loader=_synthetic_loader())
    assert r.axis == AxisName.CROSS_MODAL
    assert r.applicable is True
    # Trivial store returns all targets so Recall@k=4 is 1.0 (gold is in first k)
    per = r.details["per_dataset"]["synth"]
    assert per["recall_at_k"] == 1.0


def test_axis4_unsupported_system():
    class _NoCM:
        name = "no_cm"
        def supports(self, axis): return False
    r = crossmodal.run(_NoCM(), dataset_loader=_synthetic_loader())
    assert r.score == 0.0
    assert r.applicable is False


def test_axis4_missing_methods():
    class _Partial:
        name = "partial"
        def ingest_typed(self, e, c, m): pass
    r = crossmodal.run(_Partial(), dataset_loader=_synthetic_loader())
    assert r.score == 0.0
    assert r.applicable is False
    assert "query_topk_typed" in r.details["reason"] or "ingest_typed" in r.details["reason"]


def test_axis4_no_loader():
    sys = _SyntheticTypedStore()
    r = crossmodal.run(sys, dataset_loader=None)
    assert r.score == 0.0
    assert r.applicable is False


def test_axis4_score_bounded():
    sys = _SyntheticTypedStore()
    r = crossmodal.run(sys, dataset_loader=_synthetic_loader())
    assert 0.0 <= r.score <= 1.0
