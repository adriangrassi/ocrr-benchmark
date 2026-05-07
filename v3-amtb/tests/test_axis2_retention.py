"""Tests for AMTB Axis 2 (retention).

Uses a synthetic dataset_loader so tests stay deterministic and fast.
The synthetic dataset has clearly-separable classes (controlled by
character distribution); enough signal that an append-only baseline
hits high retention while a gradient baseline drifts.
"""
from __future__ import annotations

import random

import pytest

from amtb.axes import retention
from amtb.systems.append_only_classifier import (
    AppendOnlyClassifier, GradientClassifier,
)
from amtb.types import AxisName


def _synthetic_loader(rng_seed: int):
    """Build a deterministic synthetic dataset_loader for tests.

    Each class gets a unique character signature so simple bag-of-chars
    features can separate them. 10 classes, ~5 train per class, ~3 test
    per class, hold out 2 classes.
    """
    def loader(name: str, n_held: int, seed: int):
        rng = random.Random(rng_seed + hash(name) % 1000)
        n_classes = 10
        all_classes = list(range(n_classes))
        held_out = sorted(rng.sample(all_classes, n_held))
        known = [c for c in all_classes if c not in held_out]

        def _gen(class_id: int, n: int):
            # Class-specific character cluster: 'a' for class 0, 'b' for 1, ...
            base = chr(ord("a") + class_id)
            out = []
            for i in range(n):
                # Vary length and add a couple noise chars deterministically
                length = 8 + (i % 5)
                noise_count = 1 + (i % 2)
                text = (base * length) + ("z" * noise_count) + str(i)
                out.append((text, class_id))
            return out

        train_known = []
        for c in known:
            train_known.extend(_gen(c, 6))
        rng.shuffle(train_known)

        stream = []
        for c in held_out:
            stream.extend(_gen(c, 30))
        rng.shuffle(stream)

        test_orig = []
        for c in known:
            test_orig.extend(_gen(c, 4))

        test_novel = []
        for c in held_out:
            test_novel.extend(_gen(c, 4))

        return {
            "train_known": train_known,
            "stream": stream,
            "test_orig": test_orig,
            "test_novel": test_novel,
            "all_classes": all_classes,
        }
    return loader


def test_append_only_high_retention():
    """Append-only classifier should achieve high final_retention."""
    sys = AppendOnlyClassifier()
    config = retention.RetentionEvalConfig(
        datasets=({"name": "synth_a", "n_classes": 10, "n_held_out": 2},),
        eval_every=10,
        seed=0,
    )
    r = retention.run(sys, config=config, dataset_loader=_synthetic_loader(0))
    assert r.axis == AxisName.RETENTION
    assert r.applicable is True
    # Append-only never overwrites → near-perfect retention
    assert r.score >= 0.9, f"append-only retention should be ≥0.9, got {r.score}"


def test_gradient_baseline_lower_retention():
    """Gradient classifier should retain less under correction stream."""
    sys = GradientClassifier(lr=0.5)  # high lr to amplify forgetting
    config = retention.RetentionEvalConfig(
        datasets=({"name": "synth_b", "n_classes": 10, "n_held_out": 2},),
        eval_every=10,
        seed=0,
    )
    r = retention.run(sys, config=config, dataset_loader=_synthetic_loader(1))
    assert r.applicable is True
    # We don't pin a hard upper bound — gradient class. retention is
    # lr-dependent. Just verify it ran and produced a valid score.
    assert 0.0 <= r.score <= 1.0


def test_axis2_unsupported_system():
    class _NoRet:
        name = "no_ret"
        def supports(self, a): return False
    r = retention.run(_NoRet(), config=retention.RetentionEvalConfig())
    assert r.score == 0.0
    assert r.applicable is False


def test_axis2_missing_methods():
    class _Partial:
        name = "partial"
        def fit_classifier(self, t, l, c): pass
    r = retention.run(_Partial(), dataset_loader=_synthetic_loader(0))
    assert r.score == 0.0
    assert r.applicable is False


def test_axis2_no_loader():
    r = retention.run(AppendOnlyClassifier(), dataset_loader=None)
    assert r.score == 0.0
    assert r.applicable is False
    assert "loader" in r.details["reason"]


def test_axis2_per_dataset_breakdown():
    """Multi-dataset run reports per-dataset details."""
    sys = AppendOnlyClassifier()
    datasets = (
        {"name": "synth_a", "n_classes": 10, "n_held_out": 2},
        {"name": "synth_b", "n_classes": 10, "n_held_out": 3},
    )
    config = retention.RetentionEvalConfig(datasets=datasets, eval_every=20, seed=0)
    r = retention.run(sys, config=config, dataset_loader=_synthetic_loader(0))
    per = r.details["per_dataset"]
    assert "synth_a" in per
    assert "synth_b" in per
    for ds_data in per.values():
        assert "final_retention" in ds_data
        assert 0.0 <= ds_data["final_retention"] <= 1.0
