"""Smoke test — verify the package imports and the harness contract is intact.

For a full reproduction test, see REPRODUCING.md and run
`scripts/run_ocrr.py`.
"""

import numpy as np


def test_package_imports():
    import ocrr_benchmark
    from ocrr_benchmark.eval import (
        OCRRSystem,
        RunResult,
        run_ocrr,
        policy_oracle,
        policy_random,
    )
    from ocrr_benchmark.memory import ImmutableLedger
    assert callable(run_ocrr)
    assert callable(policy_oracle)


def test_harness_contract():
    """Toy 2-class system + 10-step stream, verify the harness drives it."""
    from ocrr_benchmark.eval import OCRRSystem, run_ocrr

    class AlwaysAlice(OCRRSystem):
        name = "always_alice"
        def predict(self, vec):
            return "alice"
        def correct(self, vec, true_label):
            pass

    rng = np.random.default_rng(0)
    stream_vecs = rng.standard_normal((10, 8)).astype(np.float32)
    stream_labels = ["alice", "bob"] * 5
    eval_vecs = rng.standard_normal((4, 8)).astype(np.float32)
    eval_labels = ["alice", "bob", "alice", "bob"]

    result = run_ocrr(
        AlwaysAlice(),
        stream_vecs,
        stream_labels,
        eval_sets={"holdout": (eval_vecs, eval_labels)},
        checkpoint_every=5,
        print_progress=False,
    )

    assert result.system_name == "always_alice"
    assert len(result.checkpoints) >= 2
    final_acc = result.checkpoints[-1].accuracies["holdout"]
    assert abs(final_acc - 0.5) < 1e-9, f"expected 0.5, got {final_acc}"


def test_ledger_hash_chain():
    """Write a few entries and verify the chain validates."""
    from ocrr_benchmark.memory import ImmutableLedger

    ledger = ImmutableLedger()
    rng = np.random.default_rng(0)
    for label in ["a", "b", "c"]:
        ledger.write(
            rng.standard_normal(8).astype(np.float32),
            tags=(f"label:{label}",),
        )

    assert ledger.verify_integrity()
    assert len(ledger) == 3


def test_force_brute_and_recall_metric():
    """force_brute=True returns same top-k as default at small N (both brute);
    verify_hnsw_recall returns the expected shape."""
    from ocrr_benchmark.memory import ImmutableLedger

    ledger = ImmutableLedger()
    rng = np.random.default_rng(0)
    for i in range(20):
        ledger.write(
            rng.standard_normal(8).astype(np.float32),
            tags=(f"class_{i % 3}",),
        )

    q = rng.standard_normal(8).astype(np.float32)
    hits_default = ledger.nearest(q, k=3)
    hits_brute = ledger.nearest(q, k=3, force_brute=True)
    assert {e.id for e, _ in hits_default} == {e.id for e, _ in hits_brute}, (
        "small-N ledger should agree between default and force_brute paths"
    )

    metrics = ledger.verify_hnsw_recall(
        rng.standard_normal((5, 8)).astype(np.float32), k=3,
    )
    assert metrics["k"] == 3
    assert metrics["num_queries"] == 5
    assert metrics["recall_at_k"] == 1.0
    assert metrics["backend"] in ("brute_only", "hnsw")


def test_substrate_force_brute_flag():
    """SubstrateSystem with force_brute=True predicts the same as default
    at small N (both backends are brute below HNSW threshold)."""
    from ocrr_benchmark.eval.ocrr_systems import SubstrateSystem

    rng = np.random.default_rng(0)
    seed_vecs = rng.standard_normal((30, 8)).astype(np.float32)
    seed_labels = [f"class_{i % 3}" for i in range(30)]

    s_default = SubstrateSystem(seed_vecs, seed_labels)
    s_brute = SubstrateSystem(seed_vecs, seed_labels, force_brute=True)

    test_q = rng.standard_normal(8).astype(np.float32)
    assert s_default.predict(test_q) == s_brute.predict(test_q)
