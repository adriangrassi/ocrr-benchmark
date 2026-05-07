"""Tests for AMTB Axis 5 (scale)."""
from __future__ import annotations

import pytest

from amtb.axes import scale
from amtb.systems.substring_retriever import SubstringRetriever
from amtb.types import AxisName


def test_substring_retriever_axis5_small():
    """Substring retriever on tiny corpus — should retrieve gold at rank 1."""
    sys = SubstringRetriever()
    config = scale.ScaleEvalConfig(
        sizes=(100, 500),
        n_queries=50,
        seed=0,
    )
    result = scale.run(sys, config=config)
    assert result.axis == AxisName.SCALE
    assert result.applicable is True
    assert 0.0 <= result.score <= 1.0
    # On tiny corpus + unique synthetic factoids, retrieval is mostly
    # perfect so MRR should be high at both sizes.
    mrrs = result.details["mrr_per_size"]
    assert float(mrrs["100"]) >= 0.5
    assert float(mrrs["500"]) >= 0.5


def test_axis5_decay_ratio_bounded():
    """Decay ratio = mrr_large / mrr_small, bounded [0, 1]."""
    sys = SubstringRetriever()
    config = scale.ScaleEvalConfig(sizes=(50, 200), n_queries=20, seed=0)
    r = scale.run(sys, config=config)
    assert 0.0 <= r.score <= 1.0


def test_axis5_unsupported_system():
    """A system that declines axis returns 0.0 with applicable=False."""
    class _NoScale:
        name = "no_scale"
        def supports(self, axis): return False
        def ingest(self, eid, text): pass
        def query_topk(self, q, k): return []
    r = scale.run(_NoScale(), config=scale.ScaleEvalConfig(sizes=(50,), n_queries=5))
    assert r.score == 0.0
    assert r.applicable is False


def test_axis5_missing_methods():
    """A system missing query_topk gets 0.0 / applicable=False with reason."""
    class _Partial:
        name = "partial"
        def ingest(self, eid, text): pass
    r = scale.run(_Partial(), config=scale.ScaleEvalConfig(sizes=(50,), n_queries=5))
    assert r.score == 0.0
    assert r.applicable is False
    assert "query_topk" in r.details["reason"] or "ingest" in r.details["reason"]


def test_axis5_max_size_clamps():
    """max_size cap is for development smoke (does not change v0.1 protocol)."""
    sys = SubstringRetriever()
    config = scale.ScaleEvalConfig(
        sizes=(100, 500, 5000),
        n_queries=10,
        seed=0,
        max_size=1000,
    )
    r = scale.run(sys, config=config)
    sizes_evaluated = r.details["sizes_evaluated"]
    assert all(s <= 1000 for s in sizes_evaluated)
    assert 5000 not in sizes_evaluated
