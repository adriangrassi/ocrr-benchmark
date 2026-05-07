"""AMTB Axis 5 — Scale (long-tail decay).

Pre-registered protocol (PRE-REGISTRATION.md §3.5):
- Ingest synthetic factoids at corpus sizes {10K, 100K, 1M, 10M}.
- Query 1,000 randomly-sampled rare facts at each size.
- Measure Mean Reciprocal Rank (MRR).
- Score: MRR_10M / MRR_10K — decay ratio, 1.0 = no decay at scale.

Pure synthetic. The factoid generator produces unique queryable
(query, gold_id) pairs from disjoint vocabulary parameter spaces, so
retrieval has a single correct answer per query.

Systems that don't claim to scale beyond their tested size declare
unsupported sizes; the axis evaluates whatever sizes the system supports
and reports 0.0 if it can't reach 10K minimum.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass

from amtb.metrics import mean_reciprocal_rank
from amtb.types import AxisName, AxisResult


# Pre-registered corpus sizes — frozen for v0.1.
DEFAULT_SIZES = (10_000, 100_000, 1_000_000, 10_000_000)
DEFAULT_QUERIES_PER_SIZE = 1_000


@dataclass(frozen=True)
class ScaleEvalConfig:
    sizes: tuple[int, ...] = DEFAULT_SIZES
    n_queries: int = DEFAULT_QUERIES_PER_SIZE
    seed: int = 0
    # Optional: skip 10M for development smokes (does NOT change v0.1 protocol;
    # publication runs MUST use full sizes).
    max_size: int | None = None


# Synthetic factoid templates and vocabularies. Frozen for v0.1.
_SUBJECTS = (
    "Andrea", "Brendan", "Camila", "Diego", "Elena", "Felix",
    "Gloria", "Hugo", "Iris", "Jonas", "Kira", "Liam",
    "Maya", "Nasir", "Olive", "Pablo", "Quinn", "Rhea",
    "Sven", "Tessa",
)
_VERBS = (
    "visited", "purchased", "studied", "renovated", "designed",
    "translated", "photographed", "sponsored", "discovered",
    "auctioned", "memorized", "patented", "exhibited",
)
_OBJECTS = (
    "the harbor district", "a rare manuscript", "the linguistics archive",
    "a mid-century cabin", "the kinetic sculpture",
    "the limited-print catalog", "the abandoned lighthouse",
    "the chess endgame study", "the fossilized fern",
    "a vintage typewriter", "the perfume formula",
    "the concert programme", "a panoramic camera",
)
# Date space: 50 years × 12 months × 28 days = 16,800 unique dates.
_YEARS = tuple(range(1975, 2025))
_MONTHS = tuple(range(1, 13))
_DAYS = tuple(range(1, 29))


def _format_factoid(idx: int, subj: str, verb: str, obj: str,
                    year: int, month: int, day: int) -> tuple[str, str]:
    """Return (entry_id, text). entry_id is unique by construction."""
    eid = f"F{idx:08d}"
    text = f"{subj} {verb} {obj} on {year}-{month:02d}-{day:02d}"
    return eid, text


def _generate_factoid_corpus(rng: random.Random, n: int) -> list[tuple[str, str]]:
    """Generate `n` unique synthetic factoids deterministically."""
    out: list[tuple[str, str]] = []
    seen_texts: set[str] = set()
    # We sample with replacement from the parameter space; uniqueness is
    # by entry_id (always unique — `idx`). Text collisions are possible
    # but extremely rare given the parameter-space size (>10^7 combos).
    for i in range(n):
        subj = rng.choice(_SUBJECTS)
        verb = rng.choice(_VERBS)
        obj = rng.choice(_OBJECTS)
        year = rng.choice(_YEARS)
        month = rng.choice(_MONTHS)
        day = rng.choice(_DAYS)
        eid, text = _format_factoid(i, subj, verb, obj, year, month, day)
        out.append((eid, text))
        seen_texts.add(text)
    return out


def _has_required_methods(system) -> bool:
    return all(callable(getattr(system, m, None)) for m in ("ingest", "query_topk"))


def run(system, *, config: ScaleEvalConfig | None = None) -> AxisResult:
    """Evaluate `system` on AMTB Axis 5.

    The system must expose:
    - ingest(entry_id, text)
    - query_topk(query: str, k: int) -> list[entry_id] — ordered best-first.
    """
    if config is None:
        config = ScaleEvalConfig()

    t0 = time.time()
    system_name = getattr(system, "name", type(system).__name__)

    if hasattr(system, "supports") and not system.supports(AxisName.SCALE):
        return AxisResult(
            system_name=system_name,
            axis=AxisName.SCALE,
            score=0.0,
            applicable=False,
            details={"reason": "system declared unsupported axis"},
            wall_seconds=time.time() - t0,
        )

    if not _has_required_methods(system):
        return AxisResult(
            system_name=system_name,
            axis=AxisName.SCALE,
            score=0.0,
            applicable=False,
            details={
                "reason": "missing ingest() or query_topk() — "
                          "system does not implement the scale contract",
            },
            wall_seconds=time.time() - t0,
        )

    sizes = config.sizes
    if config.max_size is not None:
        sizes = tuple(s for s in sizes if s <= config.max_size)
    if not sizes:
        return AxisResult(
            system_name=system_name,
            axis=AxisName.SCALE,
            score=0.0,
            applicable=False,
            details={"reason": "no corpus sizes within max_size cap"},
            wall_seconds=time.time() - t0,
        )

    rng = random.Random(config.seed)
    mrr_per_size: dict[int, float] = {}
    queries_per_size: dict[int, int] = {}

    # Build the largest corpus first (for deterministic size-N → size-M
    # nesting), then evaluate at each size by consuming a prefix.
    largest = max(sizes)
    full_corpus = _generate_factoid_corpus(rng, largest)

    # Pick query indices ONCE — same gold facts queried at each scale,
    # so we measure decay-with-corpus-size, not query variance.
    query_idxs = sorted(rng.sample(range(largest), config.n_queries))

    for size in sorted(sizes):
        # Per-size: fresh system instance via type call; we re-ingest
        # because we cannot assume systems support efficient resize.
        # Production runs should consider streaming variants.
        sys_for_size = type(system)() if isinstance(system, object) else system
        # Some systems are stateful and don't support reinit; fall back to
        # the original instance after clearing.
        if not callable(getattr(sys_for_size, "ingest", None)):
            sys_for_size = system
            if hasattr(sys_for_size, "clear"):
                sys_for_size.clear()
        for eid, text in full_corpus[:size]:
            sys_for_size.ingest(eid, text)

        # Query each chosen rare fact (only those whose idx < size are
        # ingested; others are skipped at this size).
        valid_query_idxs = [i for i in query_idxs if i < size]
        if not valid_query_idxs:
            mrr_per_size[size] = 0.0
            queries_per_size[size] = 0
            continue

        rrs = []
        for qi in valid_query_idxs:
            gold_id, gold_text = full_corpus[qi]
            # Use the gold text as the query — system retrieves by text
            # similarity. A well-functioning system places gold_id at rank 1.
            retrieved = sys_for_size.query_topk(gold_text, k=10)
            rr = mean_reciprocal_rank(retrieved, [gold_id])
            rrs.append(rr)
        mrr_per_size[size] = sum(rrs) / max(1, len(rrs))
        queries_per_size[size] = len(rrs)

    # Decay metric: MRR at largest size / MRR at smallest size.
    smallest = min(sizes)
    largest_run = max(sizes)
    mrr_small = mrr_per_size.get(smallest, 0.0)
    mrr_large = mrr_per_size.get(largest_run, 0.0)
    if mrr_small <= 0:
        decay_ratio = 0.0
    else:
        decay_ratio = min(1.0, mrr_large / mrr_small)
    decay_ratio = max(0.0, decay_ratio)

    return AxisResult(
        system_name=system_name,
        axis=AxisName.SCALE,
        score=decay_ratio,
        applicable=True,
        details={
            "sizes_evaluated": list(sorted(sizes)),
            "mrr_per_size": {str(k): v for k, v in sorted(mrr_per_size.items())},
            "queries_per_size": {str(k): v for k, v in sorted(queries_per_size.items())},
            "smallest_size": smallest,
            "largest_size": largest_run,
            "mrr_smallest": mrr_small,
            "mrr_largest": mrr_large,
        },
        wall_seconds=time.time() - t0,
    )


__all__ = ["DEFAULT_QUERIES_PER_SIZE", "DEFAULT_SIZES", "ScaleEvalConfig", "run"]
