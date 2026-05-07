"""Deterministic metrics for AMTB axes.

All metrics here are reproducible across machines without API spend.
No LLM-as-judge in this file. Per the pre-registration, the entire
benchmark is designed to be deterministic — judge-based metrics
introduce calibration drift that invalidates cross-system comparisons.

Each metric returns a float in a documented range. Inputs are typed
strictly to make misuse loud.
"""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import Sequence


# SQuAD-style normalization (matches HotpotQA, NQ, TriviaQA, LOCOMO conventions)
_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_WHITESPACE = re.compile(r"\s+")


def _normalise(s: str) -> str:
    """SQuAD-style normalization: lowercase, strip punct, drop articles, squash whitespace."""
    s = str(s).lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = _ARTICLES.sub(" ", s)
    s = _WHITESPACE.sub(" ", s).strip()
    return s


def f1_score(pred: str, gold: str) -> float:
    """Token-level F1 over normalized strings. Returns float in [0, 1]."""
    pt = _normalise(pred).split()
    gt = _normalise(gold).split()
    if not pt or not gt:
        return float(pt == gt)
    common = Counter(pt) & Counter(gt)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    p = n_same / len(pt)
    r = n_same / len(gt)
    return 2 * p * r / (p + r)


def exact_match(pred: str, gold: str) -> float:
    """1.0 if normalized prediction equals normalized gold, else 0.0."""
    return float(_normalise(pred) == _normalise(gold))


def f1_against_alternates(pred: str, golds: Sequence[str]) -> float:
    """Max F1 over a list of acceptable gold answers (NQ/TriviaQA convention)."""
    if not golds:
        return 0.0
    return max(f1_score(pred, g) for g in golds)


def em_against_alternates(pred: str, golds: Sequence[str]) -> float:
    """Max EM over a list of acceptable gold answers."""
    if not golds:
        return 0.0
    return max(exact_match(pred, g) for g in golds)


def recall_at_k(retrieved_ids: Sequence[str], gold_ids: Sequence[str], k: int) -> float:
    """Fraction of gold ids appearing in the top-k retrieved.

    `retrieved_ids` should be ordered by retrieval rank (best first).
    Returns float in [0, 1]. If `gold_ids` is empty returns 0.0.
    """
    if not gold_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    hit = sum(1 for g in gold_ids if g in top_k)
    return hit / len(gold_ids)


def mean_reciprocal_rank(retrieved_ids: Sequence[str], gold_ids: Sequence[str]) -> float:
    """MRR for a single query: 1/rank of first gold hit, 0 if no hit."""
    gold_set = set(gold_ids)
    for i, rid in enumerate(retrieved_ids, start=1):
        if rid in gold_set:
            return 1.0 / i
    return 0.0


def ndcg_at_k(
    retrieved_ids: Sequence[str], gold_ids: Sequence[str], k: int,
) -> float:
    """Normalized DCG@k, treating gold ids as binary relevance.

    DCG = sum_i (rel_i / log2(i + 1)). Normalizes against the ideal
    DCG (all gold hits at the top). Returns float in [0, 1].
    """
    import math
    gold_set = set(gold_ids)
    if not gold_set:
        return 0.0
    dcg = 0.0
    for i, rid in enumerate(retrieved_ids[:k], start=1):
        if rid in gold_set:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(len(gold_set), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def aggregate_mean(scores: Sequence[float]) -> float:
    """Equal-weighted mean. Used for axis aggregation in many AMTB axes."""
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    """Weighted mean. Used for Axis 1 (recall) where weights ∝ probe count."""
    if not values or len(values) != len(weights):
        return 0.0
    total_w = sum(weights)
    if total_w == 0:
        return 0.0
    return sum(v * w for v, w in zip(values, weights)) / total_w


__all__ = [
    "aggregate_mean",
    "em_against_alternates",
    "exact_match",
    "f1_against_alternates",
    "f1_score",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "recall_at_k",
    "weighted_mean",
]
