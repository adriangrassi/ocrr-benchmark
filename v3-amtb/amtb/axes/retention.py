"""AMTB Axis 2 — Retention (correction-stream learning).

Pre-registered protocol (PRE-REGISTRATION.md §3.2):
- 4 datasets: Banking77, CLINC150, MASSIVE-en, 20-newsgroups
- Hold out N classes per dataset; system trained only on known classes
- Stream held-out-class items with oracle correction policy
- Re-evaluate periodically; track novel + original accuracy curves
- Score per dataset: final_retention = original_acc[final] / max(original_acc[init], eps)
- Aggregate: mean across the 4 datasets, equal-weighted

This axis directly inherits the OCRR v1 protocol from arXiv:2605.03153.
The OCRR v1 implementation already exists in this repo at
`ocrr_benchmark/`; this axis-2 evaluator is a thin adapter.

Pure code, no LLM (classification with frozen encoders is the entire
loop). Free to run.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping

from amtb.metrics import aggregate_mean
from amtb.types import AxisName, AxisResult


# Pre-registered datasets and held-out class counts. Frozen for v0.1.
DEFAULT_DATASETS: tuple[Mapping[str, Any], ...] = (
    {"name": "banking77", "n_classes": 77, "n_held_out": 8},
    {"name": "clinc150", "n_classes": 151, "n_held_out": 15},
    {"name": "massive_en", "n_classes": 60, "n_held_out": 6},
    {"name": "news20", "n_classes": 20, "n_held_out": 2},
)

DEFAULT_EVAL_EVERY = 25
DEFAULT_TARGET_NOVEL = 0.7


@dataclass(frozen=True)
class RetentionEvalConfig:
    datasets: tuple[Mapping[str, Any], ...] = field(default_factory=lambda: DEFAULT_DATASETS)
    eval_every: int = DEFAULT_EVAL_EVERY
    target_novel: float = DEFAULT_TARGET_NOVEL
    seed: int = 0
    correction_policy: str = "oracle"


def _has_required_methods(system) -> bool:
    return all(callable(getattr(system, m, None)) for m in (
        "fit_classifier", "predict_class", "correct_class",
    ))


def run(system, *, config: RetentionEvalConfig | None = None,
        dataset_loader=None) -> AxisResult:
    """Evaluate `system` on AMTB Axis 2.

    The system must expose:
    - fit_classifier(train_texts: list[str], train_labels: list[int],
                     all_classes: list[int]) -> None
    - predict_class(text: str) -> int
    - correct_class(text: str, true_label: int) -> None

    `dataset_loader` is a callable (dataset_name, n_held_out, seed) ->
    {'train_known': [(text, label)], 'stream': [(text, label)],
     'test_orig': [(text, label)], 'test_novel': [(text, label)]}.

    The reason `dataset_loader` is injected is that the AMTB package
    must NOT bundle datasets directly — they're large and have varied
    licenses. Production runs supply a loader that reads from local
    cache or HuggingFace; tests use a synthetic loader.
    """
    if config is None:
        config = RetentionEvalConfig()

    t0 = time.time()
    system_name = getattr(system, "name", type(system).__name__)

    if hasattr(system, "supports") and not system.supports(AxisName.RETENTION):
        return AxisResult(
            system_name=system_name,
            axis=AxisName.RETENTION,
            score=0.0,
            applicable=False,
            details={"reason": "system declared unsupported axis"},
            wall_seconds=time.time() - t0,
        )

    if not _has_required_methods(system):
        return AxisResult(
            system_name=system_name,
            axis=AxisName.RETENTION,
            score=0.0,
            applicable=False,
            details={
                "reason": "missing fit_classifier/predict_class/correct_class — "
                          "system does not implement the retention contract",
            },
            wall_seconds=time.time() - t0,
        )

    if dataset_loader is None:
        return AxisResult(
            system_name=system_name,
            axis=AxisName.RETENTION,
            score=0.0,
            applicable=False,
            details={"reason": "no dataset_loader provided"},
            wall_seconds=time.time() - t0,
        )

    per_dataset: dict[str, dict[str, Any]] = {}
    final_retentions: list[float] = []
    corrections_to_target: dict[str, int | None] = {}

    for ds_spec in config.datasets:
        ds_name = ds_spec["name"]
        n_held = ds_spec["n_held_out"]
        n_classes = ds_spec["n_classes"]

        try:
            data = dataset_loader(ds_name, n_held, config.seed)
        except Exception as e:
            per_dataset[ds_name] = {
                "error": f"loader failed: {type(e).__name__}: {str(e)[:200]}",
                "final_retention": 0.0,
            }
            final_retentions.append(0.0)
            continue

        train_known = data["train_known"]
        stream = data["stream"]
        test_orig = data["test_orig"]
        test_novel = data["test_novel"]
        all_classes = data.get("all_classes", list(range(n_classes)))

        # Reset / reinit the system for this dataset.
        if hasattr(system, "clear"):
            system.clear()

        train_texts = [t for t, _ in train_known]
        train_labels = [l for _, l in train_known]
        system.fit_classifier(train_texts, train_labels, all_classes)

        def _accuracy(samples: list[tuple[str, int]]) -> float:
            if not samples:
                return 0.0
            n_correct = 0
            for text, true_label in samples:
                if system.predict_class(text) == true_label:
                    n_correct += 1
            return n_correct / len(samples)

        novel_curve: list[float] = []
        orig_curve: list[float] = []
        n_corrections = 0

        novel0 = _accuracy(test_novel)
        orig0 = _accuracy(test_orig)
        novel_curve.append(novel0)
        orig_curve.append(orig0)

        target_hit_at: int | None = 0 if novel0 >= config.target_novel else None

        for step, (text, true_label) in enumerate(stream, start=1):
            pred = system.predict_class(text)
            if pred != true_label:
                if config.correction_policy == "oracle":
                    system.correct_class(text, true_label)
                    n_corrections += 1
                # other policies (random_50, random_10) can be added later
            if step % config.eval_every == 0:
                novel = _accuracy(test_novel)
                orig = _accuracy(test_orig)
                novel_curve.append(novel)
                orig_curve.append(orig)
                if target_hit_at is None and novel >= config.target_novel:
                    target_hit_at = n_corrections

        # Final eval if last step wasn't a multiple of eval_every.
        if len(stream) % config.eval_every != 0:
            novel_curve.append(_accuracy(test_novel))
            orig_curve.append(_accuracy(test_orig))

        eps = 1e-9
        final_ret = orig_curve[-1] / max(orig_curve[0], eps)
        final_ret = max(0.0, min(1.0, final_ret))

        per_dataset[ds_name] = {
            "final_retention": final_ret,
            "novel_acc_init": novel0,
            "novel_acc_final": novel_curve[-1],
            "original_acc_init": orig0,
            "original_acc_final": orig_curve[-1],
            "corrections_total": n_corrections,
            "corrections_to_target_novel": target_hit_at,
            "n_held_out": n_held,
        }
        final_retentions.append(final_ret)
        corrections_to_target[ds_name] = target_hit_at

    score = aggregate_mean(final_retentions) if final_retentions else 0.0
    score = max(0.0, min(1.0, score))

    return AxisResult(
        system_name=system_name,
        axis=AxisName.RETENTION,
        score=score,
        applicable=True,
        details={
            "per_dataset": per_dataset,
            "datasets_evaluated": [d["name"] for d in config.datasets],
            "correction_policy": config.correction_policy,
        },
        wall_seconds=time.time() - t0,
    )


__all__ = [
    "DEFAULT_DATASETS",
    "DEFAULT_EVAL_EVERY",
    "DEFAULT_TARGET_NOVEL",
    "RetentionEvalConfig",
    "run",
]
