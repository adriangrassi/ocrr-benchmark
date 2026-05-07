"""AMTB Axis 1 — Recall (factoid retrieval over long context).

Pre-registered protocol (PRE-REGISTRATION.md §3.1):
- 4 datasets: LOCOMO 10 (1,533 probes), HotpotQA dev (7,405 probes),
  NaturalQuestions short (7,830 probes), TriviaQA (11,313 probes)
- Per-probe metrics: F1, EM, Recall@k
- Aggregate: weighted F1 mean, weights ∝ probe count

Two evaluation modes:
- `mode='retrieval'`: deterministic Recall@k only — no LLM, free.
- `mode='end_to_end'`: F1 + EM via system's reader — requires API budget.

Per the pre-registration, both modes are valid for v0.1. Systems that
expose only retrieval (no reader) report Recall@k and 0.0 for F1; the
matrix shows the architectural blind spot if any.

Datasets are NOT bundled in this repo (license + size). A dataset
loader callable supplies probes.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from amtb.metrics import (
    em_against_alternates, f1_against_alternates, recall_at_k, weighted_mean,
)
from amtb.types import AxisName, AxisResult


# Pre-registered datasets and weights. Frozen for v0.1.
DEFAULT_DATASETS: tuple[Mapping[str, Any], ...] = (
    {"name": "locomo10",       "n_probes": 1533,  "weight": 1533},
    {"name": "hotpotqa_dev",   "n_probes": 7405,  "weight": 7405},
    {"name": "nq_short",       "n_probes": 7830,  "weight": 7830},
    {"name": "triviaqa",       "n_probes": 11313, "weight": 11313},
)

DEFAULT_K = 20  # Top-k for Recall@k. Locked across systems per pre-reg §5.


@dataclass(frozen=True)
class RecallEvalConfig:
    datasets: tuple[Mapping[str, Any], ...] = field(default_factory=lambda: DEFAULT_DATASETS)
    mode: str = "retrieval"  # "retrieval" or "end_to_end"
    k: int = DEFAULT_K
    seed: int = 0
    limit_per_dataset: int | None = None  # for smoke tests

    def __post_init__(self) -> None:
        if self.mode not in ("retrieval", "end_to_end"):
            raise ValueError(
                f"RecallEvalConfig.mode must be 'retrieval' or 'end_to_end', got {self.mode!r}"
            )


def _has_retrieval_methods(system) -> bool:
    return all(callable(getattr(system, m, None)) for m in ("ingest", "query_topk"))


def _has_end_to_end_methods(system) -> bool:
    return all(callable(getattr(system, m, None)) for m in (
        "ingest", "query_topk", "answer",
    ))


def run(
    system,
    *,
    config: RecallEvalConfig | None = None,
    dataset_loader: Callable[[str, int | None], dict] | None = None,
) -> AxisResult:
    """Evaluate `system` on AMTB Axis 1.

    `dataset_loader(name, limit)` must return:
        {
          'corpus': [(entry_id, text), ...],     # to be ingested
          'probes': [
            {'question': str, 'gold_answers': [str], 'gold_ids': [str]},
            ...
          ],
        }

    `gold_ids` are corpus entry_ids that constitute the gold evidence.
    Used for Recall@k. `gold_answers` are short factoid alternates
    accepted by the F1/EM evaluators.

    System contract for retrieval mode:
      - ingest(entry_id, text)
      - query_topk(query, k) -> list[entry_id]

    Additional contract for end_to_end mode:
      - answer(query, retrieved_ids: list[str], retrieved_texts: list[str]) -> str
    """
    if config is None:
        config = RecallEvalConfig()

    t0 = time.time()
    system_name = getattr(system, "name", type(system).__name__)

    if hasattr(system, "supports") and not system.supports(AxisName.RECALL):
        return AxisResult(
            system_name=system_name,
            axis=AxisName.RECALL,
            score=0.0,
            applicable=False,
            details={"reason": "system declared unsupported axis"},
            wall_seconds=time.time() - t0,
        )

    needs_e2e = config.mode == "end_to_end"
    if needs_e2e and not _has_end_to_end_methods(system):
        return AxisResult(
            system_name=system_name,
            axis=AxisName.RECALL,
            score=0.0,
            applicable=False,
            details={
                "reason": "missing answer() — end_to_end mode requires "
                          "ingest/query_topk/answer",
            },
            wall_seconds=time.time() - t0,
        )
    if not needs_e2e and not _has_retrieval_methods(system):
        return AxisResult(
            system_name=system_name,
            axis=AxisName.RECALL,
            score=0.0,
            applicable=False,
            details={
                "reason": "missing ingest() or query_topk() — "
                          "system does not implement the retrieval contract",
            },
            wall_seconds=time.time() - t0,
        )

    if dataset_loader is None:
        return AxisResult(
            system_name=system_name,
            axis=AxisName.RECALL,
            score=0.0,
            applicable=False,
            details={"reason": "no dataset_loader provided"},
            wall_seconds=time.time() - t0,
        )

    per_dataset: dict[str, dict[str, Any]] = {}
    f1_per_ds: list[float] = []
    weights_per_ds: list[float] = []

    for ds_spec in config.datasets:
        ds_name = ds_spec["name"]
        weight = float(ds_spec.get("weight", ds_spec.get("n_probes", 1)))

        try:
            data = dataset_loader(ds_name, config.limit_per_dataset)
        except Exception as e:
            per_dataset[ds_name] = {
                "error": f"loader failed: {type(e).__name__}: {str(e)[:200]}",
                "f1": 0.0, "em": 0.0, "recall_at_k": 0.0,
            }
            f1_per_ds.append(0.0)
            weights_per_ds.append(weight)
            continue

        if hasattr(system, "clear"):
            system.clear()

        # Ingest corpus
        corpus = data.get("corpus", [])
        for eid, text in corpus:
            system.ingest(eid, text)

        probes = data.get("probes", [])

        # Per-probe scoring
        f1s: list[float] = []
        ems: list[float] = []
        recalls: list[float] = []
        text_by_id = dict(corpus)

        for probe in probes:
            question = probe["question"]
            gold_answers = probe.get("gold_answers", [])
            gold_ids = probe.get("gold_ids", [])

            retrieved_ids = system.query_topk(question, k=config.k)
            r_at_k = recall_at_k(retrieved_ids, gold_ids, k=config.k)
            recalls.append(r_at_k)

            if needs_e2e:
                retrieved_texts = [text_by_id.get(rid, "") for rid in retrieved_ids]
                pred = system.answer(question, retrieved_ids, retrieved_texts)
                f1s.append(f1_against_alternates(pred, gold_answers))
                ems.append(em_against_alternates(pred, gold_answers))
            else:
                # In retrieval-only mode, F1/EM are 0 (system has no reader);
                # Recall@k is the meaningful axis-1 number.
                f1s.append(0.0)
                ems.append(0.0)

        n = max(1, len(probes))
        per_dataset[ds_name] = {
            "n_probes": len(probes),
            "f1": sum(f1s) / n,
            "em": sum(ems) / n,
            "recall_at_k": sum(recalls) / n,
        }
        # In retrieval-only mode, use Recall@k as the per-dataset score for
        # axis aggregation (F1 is 0 by construction). In end_to_end, use F1.
        ds_score = per_dataset[ds_name]["f1"] if needs_e2e else per_dataset[ds_name]["recall_at_k"]
        f1_per_ds.append(ds_score)
        weights_per_ds.append(weight)

    score = weighted_mean(f1_per_ds, weights_per_ds) if f1_per_ds else 0.0
    score = max(0.0, min(1.0, score))

    return AxisResult(
        system_name=system_name,
        axis=AxisName.RECALL,
        score=score,
        applicable=True,
        details={
            "mode": config.mode,
            "k": config.k,
            "per_dataset": per_dataset,
            "datasets_evaluated": [d["name"] for d in config.datasets],
            "score_metric": "weighted_F1" if needs_e2e else "weighted_Recall@k",
        },
        wall_seconds=time.time() - t0,
    )


__all__ = ["DEFAULT_DATASETS", "DEFAULT_K", "RecallEvalConfig", "run"]
