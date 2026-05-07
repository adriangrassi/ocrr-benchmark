"""AMTB Axis 4 — Cross-modal (substrate-agnostic recall).

Pre-registered protocol (PRE-REGISTRATION.md §3.4):
- Datasets: LOCOMO-CrossModal (synthesized image-caption pairs per turn,
  ~$10 one-time), CLIP-CIFAR-100, CLAP-ESC50.
- Single shared substrate ingests entries with per-modality encoders
  (CLIP-ViT-L/14 for text + image, CLAP for audio); cross-modal
  Recall@10 measured.
- Score: mean Recall@10 across the 3 datasets.

LOCOMO-CrossModal requires LLM-driven caption synthesis (~$10 one-time).
CLIP-CIFAR-100 and CLAP-ESC50 are free — they use pre-existing labeled
images/audio. This evaluator builds the framework for all three; LOCOMO-
CrossModal slots in once captions are generated.

For systems that don't support a modality (e.g. text-only systems), the
modality contributes 0.0 to the cross-modal mean — making the
architectural blind spot visible per pre-registration §7.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from amtb.metrics import aggregate_mean, recall_at_k
from amtb.types import AxisName, AxisResult


# Pre-registered datasets. Frozen for v0.1.
DEFAULT_DATASETS: tuple[Mapping[str, Any], ...] = (
    {"name": "locomo_crossmodal", "modality": "text↔image", "n_pairs": 1533},
    {"name": "clip_cifar100",     "modality": "image↔text", "n_pairs": 10_000},
    {"name": "clap_esc50",        "modality": "audio↔text", "n_pairs": 2_000},
)


@dataclass(frozen=True)
class CrossModalEvalConfig:
    datasets: tuple[Mapping[str, Any], ...] = field(default_factory=lambda: DEFAULT_DATASETS)
    k: int = 10
    seed: int = 0
    limit_per_dataset: int | None = None


def _has_required_methods(system) -> bool:
    """Cross-modal contract: ingest typed, query typed."""
    return all(callable(getattr(system, m, None)) for m in (
        "ingest_typed", "query_topk_typed",
    ))


def run(
    system,
    *,
    config: CrossModalEvalConfig | None = None,
    dataset_loader: Callable[[str, int | None], dict] | None = None,
) -> AxisResult:
    """Evaluate `system` on AMTB Axis 4.

    The system must expose:
    - ingest_typed(entry_id, content, modality) — modality ∈ {'text', 'image', 'audio'}
    - query_topk_typed(query, query_modality, target_modality, k) -> list[entry_id]

    `dataset_loader(name, limit)` returns:
        {
          'entries': [(entry_id, content, modality), ...],
          'queries': [
            {'query': content, 'query_modality': str,
             'target_modality': str, 'gold_ids': [str]},
            ...
          ],
        }
    """
    if config is None:
        config = CrossModalEvalConfig()

    t0 = time.time()
    system_name = getattr(system, "name", type(system).__name__)

    if hasattr(system, "supports") and not system.supports(AxisName.CROSS_MODAL):
        return AxisResult(
            system_name=system_name,
            axis=AxisName.CROSS_MODAL,
            score=0.0,
            applicable=False,
            details={"reason": "system declared unsupported axis"},
            wall_seconds=time.time() - t0,
        )

    if not _has_required_methods(system):
        return AxisResult(
            system_name=system_name,
            axis=AxisName.CROSS_MODAL,
            score=0.0,
            applicable=False,
            details={
                "reason": "missing ingest_typed() or query_topk_typed() — "
                          "system does not implement the cross-modal contract",
            },
            wall_seconds=time.time() - t0,
        )

    if dataset_loader is None:
        return AxisResult(
            system_name=system_name,
            axis=AxisName.CROSS_MODAL,
            score=0.0,
            applicable=False,
            details={"reason": "no dataset_loader provided"},
            wall_seconds=time.time() - t0,
        )

    per_dataset: dict[str, dict[str, Any]] = {}
    recalls: list[float] = []

    for ds_spec in config.datasets:
        ds_name = ds_spec["name"]
        modality_pair = ds_spec.get("modality", "?")

        try:
            data = dataset_loader(ds_name, config.limit_per_dataset)
        except Exception as e:
            per_dataset[ds_name] = {
                "error": f"loader failed: {type(e).__name__}: {str(e)[:200]}",
                "recall_at_k": 0.0,
                "modality": modality_pair,
            }
            recalls.append(0.0)
            continue

        if hasattr(system, "clear"):
            system.clear()

        entries = data.get("entries", [])
        queries = data.get("queries", [])

        for eid, content, modality in entries:
            try:
                system.ingest_typed(eid, content, modality)
            except (NotImplementedError, AttributeError):
                # System cannot ingest this modality at all.
                per_dataset[ds_name] = {
                    "recall_at_k": 0.0,
                    "modality": modality_pair,
                    "reason": f"system cannot ingest modality {modality!r}",
                }
                recalls.append(0.0)
                break
        else:
            per_query_recall: list[float] = []
            for q in queries:
                try:
                    retrieved = system.query_topk_typed(
                        q["query"], q["query_modality"], q["target_modality"], config.k,
                    )
                except (NotImplementedError, AttributeError):
                    per_query_recall.append(0.0)
                    continue
                per_query_recall.append(
                    recall_at_k(retrieved, q.get("gold_ids", []), k=config.k)
                )
            mean_r = aggregate_mean(per_query_recall) if per_query_recall else 0.0
            per_dataset[ds_name] = {
                "recall_at_k": mean_r,
                "modality": modality_pair,
                "n_entries": len(entries),
                "n_queries": len(queries),
            }
            recalls.append(mean_r)

    score = aggregate_mean(recalls) if recalls else 0.0
    score = max(0.0, min(1.0, score))

    return AxisResult(
        system_name=system_name,
        axis=AxisName.CROSS_MODAL,
        score=score,
        applicable=True,
        details={
            "k": config.k,
            "per_dataset": per_dataset,
            "datasets_evaluated": [d["name"] for d in config.datasets],
            "score_metric": "mean_Recall@k_across_modality_pairs",
        },
        wall_seconds=time.time() - t0,
    )


__all__ = ["DEFAULT_DATASETS", "CrossModalEvalConfig", "run"]
