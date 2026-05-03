"""OCRR — Online Correction Recovery Rate benchmark harness.

Defines a minimal contract that any system must implement to be benchmarked,
and a driver that streams a corpus through the system, applies oracle
corrections to wrong predictions, and records accuracy curves vs.
correction count.

The point of this benchmark is to measure the property the substrate is
*designed for* — recovery from distribution shift via online correction —
which static benchmarks (Banking77, GLUE, …) by definition cannot.

API:

    class OCRRSystem:
        name: str
        def predict(self, vec: np.ndarray) -> str | None: ...
        def correct(self, vec: np.ndarray, true_label: str) -> None: ...
        def reset(self) -> None: ...

    run_ocrr(systems, stream_vecs, stream_labels, eval_sets) -> RunResult

The harness is encoder-agnostic: vectors come in, labels go out. Callers
encode once (cached embeddings preferred for reproducibility) and pass
arrays to the harness.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np


# A correction policy decides whether to APPLY a correction on a given
# stream step. Signature: (step_index, was_wrong) -> apply_correction.
# Default policy: oracle — correct every wrong prediction.
CorrectionPolicy = Callable[[int, bool], bool]


def policy_oracle(step: int, was_wrong: bool) -> bool:
    return was_wrong


def policy_random(prob: float, seed: int = 0) -> CorrectionPolicy:
    """Correct only with probability `prob` when wrong; never correct when right."""
    rng = np.random.default_rng(seed)
    def _policy(step: int, was_wrong: bool) -> bool:
        if not was_wrong:
            return False
        return bool(rng.random() < prob)
    return _policy


@dataclass
class CheckpointMetric:
    """One snapshot of a system's state during the OCRR stream."""
    step: int                       # number of stream items processed so far
    corrections_so_far: int         # cumulative count of correct() calls
    accuracies: dict[str, float]    # eval-set name -> accuracy
    pred_secs: float                # cumulative wall-time of predict() calls
    correct_secs: float             # cumulative wall-time of correct() calls


@dataclass
class RunResult:
    system_name: str
    checkpoints: list[CheckpointMetric] = field(default_factory=list)
    final_pred_secs: float = 0.0
    final_correct_secs: float = 0.0


class OCRRSystem:
    """Abstract base. Subclasses implement predict / correct / reset."""

    name: str = "abstract"

    def predict(self, vec: np.ndarray) -> str | None:
        raise NotImplementedError

    def correct(self, vec: np.ndarray, true_label: str) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        """Restore initial state (between scenario runs). Default: no-op."""


def run_ocrr(
    system: OCRRSystem,
    stream_vecs: np.ndarray,
    stream_labels: list[str],
    eval_sets: dict[str, tuple[np.ndarray, list[str]]],
    *,
    checkpoint_every: int = 50,
    correct_on_wrong: bool = True,
    correction_policy: CorrectionPolicy | None = None,
    print_progress: bool = True,
) -> RunResult:
    """Drive a system through the OCRR loop.

    For each (vec, label) in the stream:
        - call system.predict(vec); record whether it was right
        - if wrong AND correct_on_wrong: call system.correct(vec, label)
    Every ``checkpoint_every`` stream items, evaluate the system on each
    eval set and record accuracies.

    eval_sets is a dict of name -> (vecs, labels). Typically:
        "novel"    -> held-out-class test queries
        "original" -> original-distribution test queries (forgetting check)

    `correction_policy` if provided overrides `correct_on_wrong` and decides
    per-step whether to apply a correction. Use `policy_oracle` for "every
    wrong → correct", `policy_random(prob, seed)` for stochastic policies.
    """
    if correction_policy is None:
        correction_policy = policy_oracle if correct_on_wrong else (lambda _i, _w: False)
    n = len(stream_vecs)
    assert len(stream_labels) == n
    result = RunResult(system_name=system.name)

    pred_secs = 0.0
    correct_secs = 0.0
    corrections = 0

    def eval_now() -> dict[str, float]:
        out: dict[str, float] = {}
        for name, (vecs, labels) in eval_sets.items():
            n_eval = len(vecs)
            n_correct = 0
            for v, lbl in zip(vecs, labels):
                p = system.predict(v)
                if p == lbl:
                    n_correct += 1
            out[name] = n_correct / n_eval if n_eval > 0 else 0.0
        return out

    # Step-0 checkpoint (before any stream items)
    accs = eval_now()
    result.checkpoints.append(CheckpointMetric(
        step=0, corrections_so_far=0, accuracies=accs,
        pred_secs=0.0, correct_secs=0.0,
    ))
    if print_progress:
        acc_str = "  ".join(f"{k}={v:.4f}" for k, v in accs.items())
        print(f"  [step    0  corr     0]  {acc_str}", flush=True)

    for i, (vec, lbl) in enumerate(zip(stream_vecs, stream_labels)):
        t0 = time.time()
        pred = system.predict(vec)
        pred_secs += time.time() - t0

        was_wrong = (pred != lbl)
        if correction_policy(i, was_wrong):
            t0 = time.time()
            system.correct(vec, lbl)
            correct_secs += time.time() - t0
            corrections += 1

        if (i + 1) % checkpoint_every == 0 or (i + 1) == n:
            accs = eval_now()
            result.checkpoints.append(CheckpointMetric(
                step=i + 1, corrections_so_far=corrections, accuracies=accs,
                pred_secs=pred_secs, correct_secs=correct_secs,
            ))
            if print_progress:
                acc_str = "  ".join(f"{k}={v:.4f}" for k, v in accs.items())
                print(
                    f"  [step {i + 1:>4}  corr {corrections:>4}]  {acc_str}  "
                    f"(pred={pred_secs:.1f}s  corr={correct_secs:.1f}s)",
                    flush=True,
                )

    result.final_pred_secs = pred_secs
    result.final_correct_secs = correct_secs
    return result


# ---------------------------------------------------------------- summary helpers

def corrections_to_accuracy(
    result: RunResult, eval_set: str, target: float,
) -> int | None:
    """Smallest correction count where ``result``'s ``eval_set`` accuracy
    first reaches ``target``. None if never reached."""
    for cp in result.checkpoints:
        if cp.accuracies.get(eval_set, 0.0) >= target:
            return cp.corrections_so_far
    return None


def final_accuracies(result: RunResult) -> dict[str, float]:
    if not result.checkpoints:
        return {}
    return dict(result.checkpoints[-1].accuracies)


def to_csv_rows(result: RunResult) -> list[dict[str, float | int | str]]:
    rows = []
    for cp in result.checkpoints:
        row: dict[str, float | int | str] = {
            "system": result.system_name,
            "step": cp.step,
            "corrections": cp.corrections_so_far,
            "pred_secs": round(cp.pred_secs, 4),
            "correct_secs": round(cp.correct_secs, 4),
        }
        for k, v in cp.accuracies.items():
            row[f"acc_{k}"] = round(v, 6)
        rows.append(row)
    return rows
