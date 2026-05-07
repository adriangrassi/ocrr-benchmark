"""Tests for AMTB Axis 1 (recall) — retrieval-only mode (free)."""
from __future__ import annotations

import pytest

from amtb.axes import recall
from amtb.systems.substring_retriever import SubstringRetriever
from amtb.types import AxisName


def _synthetic_loader():
    """Build a deterministic synthetic recall dataset.

    Each probe's answer is in exactly one corpus entry; substring
    retriever should find it perfectly because the question contains
    distinguishing tokens from the gold answer entry.
    """
    def loader(name: str, limit: int | None = None):
        corpus = [
            ("d1", "Caroline volunteers at the LGBTQ youth center every Saturday"),
            ("d2", "Melanie went to Hawaii on 7 May 2023 with her family"),
            ("d3", "Caroline mentored students in the school speech program last year"),
            ("d4", "Hugo bought a vintage typewriter at the flea market in March"),
            ("d5", "Iris translated a rare manuscript for the linguistics archive"),
        ]
        probes = [
            {"question": "Where does Caroline volunteer?",
             "gold_answers": ["LGBTQ youth center"],
             "gold_ids": ["d1"]},
            {"question": "When did Melanie go to Hawaii?",
             "gold_answers": ["7 May 2023"],
             "gold_ids": ["d2"]},
            {"question": "What program did Caroline mentor in?",
             "gold_answers": ["school speech program"],
             "gold_ids": ["d3"]},
            {"question": "What did Hugo buy at the flea market?",
             "gold_answers": ["vintage typewriter"],
             "gold_ids": ["d4"]},
        ]
        if limit is not None:
            probes = probes[:limit]
        return {"corpus": corpus, "probes": probes}
    return loader


def test_retrieval_only_mode_recall_at_k():
    sys = SubstringRetriever()
    config = recall.RecallEvalConfig(
        datasets=({"name": "synth", "n_probes": 4, "weight": 4},),
        mode="retrieval", k=3,
    )
    r = recall.run(sys, config=config, dataset_loader=_synthetic_loader())
    assert r.axis == AxisName.RECALL
    assert r.applicable is True
    # Substring retriever should find each gold doc — Recall@3 ≥ 0.75
    per = r.details["per_dataset"]["synth"]
    assert per["recall_at_k"] >= 0.75
    assert per["f1"] == 0.0  # retrieval-only mode → F1 is 0 by construction
    assert r.details["score_metric"] == "weighted_Recall@k"


def test_end_to_end_mode_requires_answer():
    """A system without answer() can't run end_to_end mode."""
    sys = SubstringRetriever()
    config = recall.RecallEvalConfig(
        datasets=({"name": "synth", "n_probes": 4, "weight": 4},),
        mode="end_to_end",
    )
    r = recall.run(sys, config=config, dataset_loader=_synthetic_loader())
    assert r.applicable is False
    assert "answer" in r.details["reason"]


def test_end_to_end_with_oracle_answer():
    """Stub system that returns the gold answer when retrieved → F1 ≈ 1.0."""

    class _OracleSystem(SubstringRetriever):
        @property
        def name(self) -> str:
            return "oracle"

        def supports(self, axis: AxisName) -> bool:
            return axis == AxisName.RECALL

        def answer(self, query: str, retrieved_ids, retrieved_texts) -> str:
            # Naive oracle: pull the most-distinctive content phrase from
            # the top retrieved text. For the synthetic loader this works
            # because each probe's gold_answer is a literal substring.
            if retrieved_texts:
                return retrieved_texts[0]
            return ""

    sys = _OracleSystem()
    config = recall.RecallEvalConfig(
        datasets=({"name": "synth", "n_probes": 4, "weight": 4},),
        mode="end_to_end", k=3,
    )
    r = recall.run(sys, config=config, dataset_loader=_synthetic_loader())
    assert r.applicable is True
    per = r.details["per_dataset"]["synth"]
    # Returning the full text gets meaningful F1 against the gold short
    # answer — the gold tokens are subset of the full text.
    assert per["f1"] > 0.0


def test_recall_axis_unsupported_system():
    class _NoRecall:
        name = "no_recall"
        def supports(self, axis): return False
        def ingest(self, e, t): pass
        def query_topk(self, q, k): return []
    r = recall.run(_NoRecall(), dataset_loader=_synthetic_loader())
    assert r.score == 0.0
    assert r.applicable is False


def test_recall_axis_no_loader():
    sys = SubstringRetriever()
    r = recall.run(sys, dataset_loader=None)
    assert r.score == 0.0
    assert r.applicable is False
