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
