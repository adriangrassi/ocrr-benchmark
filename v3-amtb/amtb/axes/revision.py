"""AMTB Axis 6 — Adversarial revision (override robustness).

Pre-registered protocol (PRE-REGISTRATION.md §3.6):
- Take LOCOMO conversations. Ingest all turns normally.
- After ingestion, inject N "revisions" — entries that contradict
  existing entries (e.g. "Caroline is single" → "Caroline got married
  last week").
- Three contamination levels: 5%, 10%, 20% of original facts revised.
- Query the ORIGINAL gold answers. Measure preservation rate.
- Score: mean preservation rate across the three levels.

A flat-overwrite system loses N% by construction. An append-only system
retains both versions; the question is whether retrieval still surfaces
the original under contradiction.

Two evaluation modes:
- `mode='retrieval'`: free — measures whether the original gold ENTRY
  is still in top-k after revision injection.
- `mode='end_to_end'`: paid — measures whether the reader's answer
  still matches the original gold under contradiction.

The retrieval-only mode is sufficient to demonstrate the architectural
property: append-only systems retain originals, flat-overwrite ones
don't.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from amtb.metrics import (
    aggregate_mean, em_against_alternates, f1_against_alternates, recall_at_k,
)
from amtb.types import AxisName, AxisResult


# Pre-registered contamination levels. Frozen for v0.1.
DEFAULT_LEVELS = (0.05, 0.10, 0.20)


@dataclass(frozen=True)
class RevisionEvalConfig:
    levels: tuple[float, ...] = DEFAULT_LEVELS
    mode: str = "retrieval"  # or "end_to_end"
    k: int = 20
    seed: int = 0
    limit_probes: int | None = None  # for smoke tests

    def __post_init__(self) -> None:
        if self.mode not in ("retrieval", "end_to_end"):
            raise ValueError(
                f"RevisionEvalConfig.mode must be 'retrieval' or 'end_to_end', got {self.mode!r}"
            )
        for L in self.levels:
            if not 0.0 <= L <= 1.0:
                raise ValueError(f"contamination level must be in [0, 1], got {L}")


def _has_retrieval_methods(system) -> bool:
    return all(callable(getattr(system, m, None)) for m in ("ingest", "query_topk"))


def _has_end_to_end_methods(system) -> bool:
    return _has_retrieval_methods(system) and callable(getattr(system, "answer", None))


def _generate_revision(rng: random.Random, original_text: str, eid: str) -> str:
    """Synthetically construct a contradicting entry text.

    Production runs would use LLM-generated revisions for realism; for
    deterministic axis evaluation we apply a simple pattern: prepend
    "ACTUALLY, " and reverse a polarity word. The substring retriever
    will rank both as candidates; what we measure is whether the
    ORIGINAL entry is still retrievable in top-k.
    """
    polarity = {
        "is ": "is NOT ",
        "was ": "was NOT ",
        "did ": "did NOT ",
        "went ": "did NOT go ",
        "bought ": "did NOT buy ",
        "single": "married",
        "married": "single",
        "yes": "no",
        "true": "false",
        "open": "closed",
    }
    text = original_text
    flipped = False
    for needle, repl in polarity.items():
        if needle in text.lower():
            text = text.replace(needle, repl)
            flipped = True
            break
    if not flipped:
        text = "ACTUALLY, NOT TRUE: " + text
    return text


def run(
    system,
    *,
    config: RevisionEvalConfig | None = None,
    dataset_loader: Callable[[int | None], dict] | None = None,
) -> AxisResult:
    """Evaluate `system` on AMTB Axis 6.

    `dataset_loader(limit)` must return:
      {
        'corpus': [(entry_id, text), ...],
        'probes': [
          {'question': str, 'gold_answers': [str], 'gold_ids': [str]},
          ...
        ],
      }

    System contract identical to Axis 1.
    """
    if config is None:
        config = RevisionEvalConfig()

    t0 = time.time()
    system_name = getattr(system, "name", type(system).__name__)

    if hasattr(system, "supports") and not system.supports(AxisName.ADVERSARIAL_REVISION):
        return AxisResult(
            system_name=system_name,
            axis=AxisName.ADVERSARIAL_REVISION,
            score=0.0,
            applicable=False,
            details={"reason": "system declared unsupported axis"},
            wall_seconds=time.time() - t0,
        )

    needs_e2e = config.mode == "end_to_end"
    if needs_e2e and not _has_end_to_end_methods(system):
        return AxisResult(
            system_name=system_name,
            axis=AxisName.ADVERSARIAL_REVISION,
            score=0.0,
            applicable=False,
            details={"reason": "missing answer() — end_to_end mode required"},
            wall_seconds=time.time() - t0,
        )
    if not needs_e2e and not _has_retrieval_methods(system):
        return AxisResult(
            system_name=system_name,
            axis=AxisName.ADVERSARIAL_REVISION,
            score=0.0,
            applicable=False,
            details={"reason": "missing ingest()/query_topk()"},
            wall_seconds=time.time() - t0,
        )

    if dataset_loader is None:
        return AxisResult(
            system_name=system_name,
            axis=AxisName.ADVERSARIAL_REVISION,
            score=0.0,
            applicable=False,
            details={"reason": "no dataset_loader provided"},
            wall_seconds=time.time() - t0,
        )

    data = dataset_loader(config.limit_probes)
    corpus = data["corpus"]
    probes = data["probes"]

    rng = random.Random(config.seed)

    per_level: dict[str, dict[str, float]] = {}
    preservation_per_level: list[float] = []

    # We pick which corpus entries to "revise" once per level (deterministic).
    n_corpus = len(corpus)
    corpus_id_set = {eid for eid, _ in corpus}

    for level in config.levels:
        if hasattr(system, "clear"):
            system.clear()

        # Phase 1: ingest original corpus
        for eid, text in corpus:
            system.ingest(eid, text)

        # Phase 2: choose revision targets and inject
        n_revisions = max(0, int(round(level * n_corpus)))
        revision_targets = rng.sample(range(n_corpus), n_revisions) if n_revisions else []
        revised_eids: set[str] = set()
        for ti, idx in enumerate(revision_targets):
            orig_eid, orig_text = corpus[idx]
            revised_text = _generate_revision(rng, orig_text, orig_eid)
            new_eid = f"{orig_eid}__rev{ti}"
            system.ingest(new_eid, revised_text)
            revised_eids.add(orig_eid)

        # Phase 3: query each probe — measure preservation
        # Preservation: did the ORIGINAL gold entry still appear in top-k?
        # (For end_to_end: did the reader still produce the original answer?)
        preserved = 0
        f1s: list[float] = []
        ems: list[float] = []
        text_by_id = dict(corpus)

        for probe in probes:
            question = probe["question"]
            gold_ids = probe.get("gold_ids", [])
            gold_answers = probe.get("gold_answers", [])

            retrieved = system.query_topk(question, k=config.k)
            r = recall_at_k(retrieved, gold_ids, k=config.k)
            preserved += 1 if r > 0 else 0

            if needs_e2e:
                retrieved_texts = [text_by_id.get(rid, "") for rid in retrieved]
                pred = system.answer(question, retrieved, retrieved_texts)
                f1s.append(f1_against_alternates(pred, gold_answers))
                ems.append(em_against_alternates(pred, gold_answers))

        n = max(1, len(probes))
        preservation_rate = preserved / n
        per_level[f"{level:.2f}"] = {
            "preservation_rate": preservation_rate,
            "n_revisions": n_revisions,
            "n_revised_eids": len(revised_eids),
            "f1_mean": (sum(f1s) / n) if f1s else 0.0,
            "em_mean": (sum(ems) / n) if ems else 0.0,
        }

        # In retrieval-only mode score is preservation_rate; in e2e it's F1
        ds_score = (
            sum(f1s) / n if needs_e2e and f1s else preservation_rate
        )
        preservation_per_level.append(ds_score)

    score = aggregate_mean(preservation_per_level) if preservation_per_level else 0.0
    score = max(0.0, min(1.0, score))

    return AxisResult(
        system_name=system_name,
        axis=AxisName.ADVERSARIAL_REVISION,
        score=score,
        applicable=True,
        details={
            "mode": config.mode,
            "k": config.k,
            "per_level": per_level,
            "score_metric": (
                "F1_mean_across_levels" if needs_e2e else "preservation_rate_across_levels"
            ),
        },
        wall_seconds=time.time() - t0,
    )


__all__ = ["DEFAULT_LEVELS", "RevisionEvalConfig", "run"]
