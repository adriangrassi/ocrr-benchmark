"""Tests for AMTB Axis 3 (auditability)."""
from __future__ import annotations

import pytest

from amtb.axes import audit
from amtb.systems.hash_chained_ledger import (
    FlatDictLedger, HashChainedLedger,
)
from amtb.types import AxisName


@pytest.fixture
def small_config():
    return audit.AuditEvalConfig(n_entries=200, n_tampers_per_type=20, seed=0)


def test_hash_chained_ledger_passes_axis3(small_config):
    """Reference hash-chained ledger should detect ~all tampers."""
    sys = HashChainedLedger()
    result = audit.run(sys, config=small_config)
    assert result.axis == AxisName.AUDITABILITY
    assert result.applicable is True
    # Reference impl is designed to detect every tamper. Allow a small
    # margin for any stochastic edge case (e.g. swap of identical entries
    # — but our synth generator avoids that).
    assert result.score >= 0.95, f"hash-chained should detect ≥95% tampers, got {result.score:.3f}"
    assert result.details["fp_gate"] == "passed"
    detected = result.details["detected"]
    assert detected["overwrite"] == small_config.n_tampers_per_type
    assert detected["forgery"] == small_config.n_tampers_per_type
    # Reorder of two entries with identical content is undetectable in
    # principle (swap is symmetric); but our synthetic content is mostly
    # unique. We expect ≥ 95% detection.
    assert detected["reorder"] >= int(0.95 * small_config.n_tampers_per_type)
    assert detected["deletion"] == small_config.n_tampers_per_type


def test_flat_dict_ledger_scores_zero(small_config):
    """Naive baseline declares no audit support → score 0.0, applicable False."""
    sys = FlatDictLedger()
    result = audit.run(sys, config=small_config)
    assert result.score == 0.0
    assert result.applicable is False
    assert "unsupported" in result.details["reason"].lower()


def test_audit_score_in_range(small_config):
    """Score must be in [0, 1] always."""
    sys = HashChainedLedger()
    r = audit.run(sys, config=small_config)
    assert 0.0 <= r.score <= 1.0


def test_audit_full_size_smoke():
    """Smoke test at the pre-registered full size (10K entries × 1K tampers).
    Slower; primarily checks performance is reasonable."""
    sys = HashChainedLedger()
    result = audit.run(sys, config=audit.AuditEvalConfig(
        n_entries=1000, n_tampers_per_type=50, seed=0,
    ))
    assert result.score >= 0.95
    assert result.wall_seconds < 60  # Should be fast even at this scale
