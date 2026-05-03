"""OCRR harness, systems, baselines, and ablations."""

from ocrr_benchmark.eval.ocrr import (
    CheckpointMetric,
    CorrectionPolicy,
    OCRRSystem,
    RunResult,
    corrections_to_accuracy,
    final_accuracies,
    policy_oracle,
    policy_random,
    run_ocrr,
    to_csv_rows,
)

__all__ = [
    "CheckpointMetric",
    "CorrectionPolicy",
    "OCRRSystem",
    "RunResult",
    "corrections_to_accuracy",
    "final_accuracies",
    "policy_oracle",
    "policy_random",
    "run_ocrr",
    "to_csv_rows",
]
