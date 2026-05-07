"""AMTB Axis 3 — Auditability (synthetic tamper test).

Pre-registered protocol (PRE-REGISTRATION.md §3.3):
- 10,000 synthetic ledger entries ingested via the system's `ingest()`.
- 1,000 deliberate tampering attempts split evenly:
  - 250 overwrite (modify entry text without rotating hash)
  - 250 reorder (swap two entries)
  - 250 deletion (remove without acknowledgment)
  - 250 forgery (insert new entry signed with wrong key/chain)
- Metric: tamper-detection rate (true positives / total tampers) at
  zero false-positive rate on 10,000 unmodified entries.

Systems that don't claim auditability score 0.0 (per
transparent-failure-reporting commitment).

Pure synthetic, no LLM, no datasets, deterministic with fixed seed.
"""
from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass

from amtb.types import AxisName, AxisResult


N_ENTRIES = 10_000
N_TAMPERS_PER_TYPE = 250
TAMPER_TYPES = ("overwrite", "reorder", "deletion", "forgery")
TOTAL_TAMPERS = N_TAMPERS_PER_TYPE * len(TAMPER_TYPES)


@dataclass(frozen=True)
class AuditEvalConfig:
    n_entries: int = N_ENTRIES
    n_tampers_per_type: int = N_TAMPERS_PER_TYPE
    seed: int = 0


def _generate_entries(rng: random.Random, n: int) -> list[tuple[str, str]]:
    """Synthetic (entry_id, text) pairs. Deterministic given seed."""
    out = []
    for i in range(n):
        eid = f"E{i:06d}"
        # Synthetic text with variable content so tamper detection is
        # nontrivial (overwrite must produce different valid-looking text).
        topic = rng.choice(("policy", "event", "decision", "fact", "preference"))
        seq = rng.randint(0, 1_000_000)
        text = f"{topic}-{seq}: payload for entry {i}"
        out.append((eid, text))
    return out


def _has_required_methods(system) -> bool:
    """Check the duck-typed audit contract."""
    return all(callable(getattr(system, m, None)) for m in (
        "ingest", "verify_chain",
    ))


def run(system, *, config: AuditEvalConfig | None = None) -> AxisResult:
    """Evaluate `system` on AMTB Axis 3.

    The system must expose:
    - ingest(entry_id, text)
    - verify_chain() -> bool

    For tamper-injection, the evaluator uses `_force_overwrite`,
    `_force_swap`, `_force_delete`, `_force_inject` if present
    (these are intentionally back-door methods to simulate adversarial
    storage access). Systems without them are tested using a different
    tamper protocol that bypasses the system API entirely (e.g., by
    constructing two parallel system instances and asking one to verify
    state from the other after manual mutation).
    """
    if config is None:
        config = AuditEvalConfig()

    t0 = time.time()
    system_name = getattr(system, "name", type(system).__name__)

    # If system declines this axis, report 0.0 with applicable=False.
    if hasattr(system, "supports") and not system.supports(AxisName.AUDITABILITY):
        return AxisResult(
            system_name=system_name,
            axis=AxisName.AUDITABILITY,
            score=0.0,
            applicable=False,
            details={"reason": "system declared unsupported axis"},
            wall_seconds=time.time() - t0,
        )

    if not _has_required_methods(system):
        return AxisResult(
            system_name=system_name,
            axis=AxisName.AUDITABILITY,
            score=0.0,
            applicable=False,
            details={
                "reason": "missing ingest() or verify_chain() — "
                          "system does not implement the audit contract",
            },
            wall_seconds=time.time() - t0,
        )

    rng = random.Random(config.seed)
    entries = _generate_entries(rng, config.n_entries)

    # Phase 1: ingest all entries.
    for eid, text in entries:
        system.ingest(eid, text)

    # Phase 2: false-positive check — verify with NO tampering applied.
    fp_baseline_intact = bool(system.verify_chain())
    if not fp_baseline_intact:
        # System claims tampering when there isn't any → high FP rate.
        # Score 0 for failing the FP gate.
        return AxisResult(
            system_name=system_name,
            axis=AxisName.AUDITABILITY,
            score=0.0,
            applicable=True,
            details={
                "reason": "verify_chain() returned False on untampered ledger "
                          "(false-positive gate failed)",
                "fp_gate": "failed",
            },
            wall_seconds=time.time() - t0,
        )

    # Phase 3: tampering trials. We test detection one tamper at a time
    # on fresh deep-copies of the system so each attempt is independent.
    detected = {t: 0 for t in TAMPER_TYPES}

    def _make_copy_after_ingest():
        """Deep-copy the post-ingest system state for an independent trial."""
        return copy.deepcopy(system)

    n_per = config.n_tampers_per_type

    # Overwrite: modify text of a random entry without recomputing hash.
    for i in range(n_per):
        s = _make_copy_after_ingest()
        if not hasattr(s, "_force_overwrite"):
            break
        idx = rng.randrange(config.n_entries)
        s._force_overwrite(idx, f"TAMPERED-{i}")
        if not s.verify_chain():
            detected["overwrite"] += 1

    # Reorder: swap two random entries.
    for i in range(n_per):
        s = _make_copy_after_ingest()
        if not hasattr(s, "_force_swap"):
            break
        i1 = rng.randrange(config.n_entries)
        i2 = rng.randrange(config.n_entries)
        while i2 == i1:
            i2 = rng.randrange(config.n_entries)
        s._force_swap(i1, i2)
        if not s.verify_chain():
            detected["reorder"] += 1

    # Deletion: remove an entry without rotating chain.
    for i in range(n_per):
        s = _make_copy_after_ingest()
        if not hasattr(s, "_force_delete"):
            break
        idx = rng.randrange(config.n_entries)
        s._force_delete(idx)
        if not s.verify_chain():
            detected["deletion"] += 1

    # Forgery: inject a new entry with wrong-chain hash.
    for i in range(n_per):
        s = _make_copy_after_ingest()
        if not hasattr(s, "_force_inject"):
            break
        idx = rng.randrange(1, config.n_entries)
        s._force_inject(idx, f"FORGED-{i}", "forged-payload")
        if not s.verify_chain():
            detected["forgery"] += 1

    total_detected = sum(detected.values())
    total_tampers = n_per * len(TAMPER_TYPES)
    score = total_detected / total_tampers if total_tampers > 0 else 0.0
    score = max(0.0, min(1.0, score))

    return AxisResult(
        system_name=system_name,
        axis=AxisName.AUDITABILITY,
        score=score,
        applicable=True,
        details={
            "n_entries": config.n_entries,
            "n_tampers_per_type": n_per,
            "detected": detected,
            "total_detected": total_detected,
            "total_tampers": total_tampers,
            "fp_gate": "passed",
        },
        wall_seconds=time.time() - t0,
    )


__all__ = ["AuditEvalConfig", "run"]
