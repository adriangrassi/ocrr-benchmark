"""Unit tests for amtb.metrics. All deterministic, no API calls."""
from __future__ import annotations

import math

import pytest

from amtb.metrics import (
    em_against_alternates,
    exact_match,
    f1_against_alternates,
    f1_score,
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
    weighted_mean,
)


# ---------- F1 / EM ---------------------------------------------------------
def test_f1_exact_match_one():
    assert f1_score("Caroline", "Caroline") == 1.0


def test_f1_normalization_articles_punct():
    # Articles dropped, punctuation dropped → identical
    assert f1_score("The Caroline.", "Caroline") == 1.0


def test_f1_partial_overlap():
    # 1 of 3 tokens overlaps both ways: f1 = 2*1/(3+3)*... actually
    # tokens "mentoring program school speech" vs "mentoring program"
    # P = 2/4 = 0.5, R = 2/2 = 1.0 → F1 = 2*0.5*1/(0.5+1) = 0.667
    assert math.isclose(
        f1_score("mentoring program school speech", "mentoring program"),
        2 * 0.5 * 1.0 / 1.5, rel_tol=1e-3,
    )


def test_f1_no_overlap():
    assert f1_score("apples", "oranges") == 0.0


def test_em_match_after_norm():
    assert exact_match("THE answer.", "answer") == 1.0


def test_em_mismatch():
    assert exact_match("foo", "bar") == 0.0


def test_f1_alternates_picks_best():
    assert f1_against_alternates("New York", ["NYC", "New York City"]) > 0.0
    assert f1_against_alternates("apple", ["banana", "cherry"]) == 0.0


def test_em_alternates_picks_best():
    assert em_against_alternates("apple", ["banana", "apple", "cherry"]) == 1.0


# ---------- Recall@k --------------------------------------------------------
def test_recall_at_k_full_hit():
    retrieved = ["d1", "d2", "d3", "d4"]
    gold = ["d2"]
    assert recall_at_k(retrieved, gold, k=4) == 1.0
    assert recall_at_k(retrieved, gold, k=2) == 1.0
    assert recall_at_k(retrieved, gold, k=1) == 0.0  # d1 first


def test_recall_at_k_partial():
    retrieved = ["d1", "d3", "d5", "d7"]
    gold = ["d3", "d5", "d9"]
    # k=4: 2 of 3 gold → 0.667
    assert math.isclose(recall_at_k(retrieved, gold, k=4), 2 / 3, rel_tol=1e-3)
    # k=2: 1 of 3 gold → 0.333
    assert math.isclose(recall_at_k(retrieved, gold, k=2), 1 / 3, rel_tol=1e-3)


def test_recall_at_k_empty_gold():
    assert recall_at_k(["a", "b"], [], k=2) == 0.0


# ---------- MRR -------------------------------------------------------------
def test_mrr_first_position():
    assert mean_reciprocal_rank(["d1", "d2", "d3"], ["d1"]) == 1.0


def test_mrr_third_position():
    assert math.isclose(
        mean_reciprocal_rank(["d1", "d2", "d3"], ["d3"]), 1 / 3, rel_tol=1e-9,
    )


def test_mrr_no_hit():
    assert mean_reciprocal_rank(["d1", "d2"], ["d99"]) == 0.0


def test_mrr_takes_first_hit():
    # gold = {d1, d2}; first retrieved gold is d2 at rank 2
    assert mean_reciprocal_rank(["d3", "d2", "d1"], ["d1", "d2"]) == 0.5


# ---------- NDCG@k ----------------------------------------------------------
def test_ndcg_perfect():
    # All gold at top → ndcg = 1
    assert math.isclose(ndcg_at_k(["d1", "d2", "d3"], ["d1", "d2", "d3"], k=3), 1.0,
                        rel_tol=1e-3)


def test_ndcg_no_gold():
    assert ndcg_at_k(["d1", "d2"], [], k=2) == 0.0


def test_ndcg_partial_relevance_drops():
    # gold at rank 2 only → dcg = 1/log2(3); idcg = 1/log2(2) = 1
    val = ndcg_at_k(["x", "g", "y"], ["g"], k=3)
    expected = (1 / math.log2(3)) / 1.0
    assert math.isclose(val, expected, rel_tol=1e-3)


# ---------- Weighted mean (axis 1 aggregation) ------------------------------
def test_weighted_mean_axis1_style():
    # Per pre-registration §3.1: axis 1 weights ∝ probe count
    f1s = [0.84, 0.55, 0.62, 0.48]
    weights = [1533, 7405, 7830, 11313]
    expected = sum(f * w for f, w in zip(f1s, weights)) / sum(weights)
    assert math.isclose(weighted_mean(f1s, weights), expected, rel_tol=1e-9)


def test_weighted_mean_zero_weight():
    assert weighted_mean([1.0, 1.0], [0.0, 0.0]) == 0.0
