"""Core types for the AMTB benchmark.

These types are pre-registered as part of v0.1. Changing them after
measurements begin would constitute a v0.2 release with separate
baselining (per the pre-registration's invalidation clauses).

The MemorySystem Protocol defines the minimal interface a system must
satisfy to be evaluable on AMTB. A system can declare per-axis
applicability via `supports(axis_name)`; cells where it cannot run
return 0.0 in the matrix (NOT omitted — see pre-registration §7).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, runtime_checkable


class AxisName(str, Enum):
    """The six pre-registered AMTB axes. Order is canonical."""

    RECALL = "recall"
    RETENTION = "retention"
    AUDITABILITY = "auditability"
    CROSS_MODAL = "cross_modal"
    SCALE = "scale"
    ADVERSARIAL_REVISION = "adversarial_revision"


@dataclass(frozen=True)
class AxisResult:
    """Single (system, axis) evaluation result.

    `score` is the canonical 0-1 metric for that axis (axis-specific
    aggregation defined in axes/<name>.py). `details` carries per-dataset
    or per-condition breakdowns for transparency.

    `applicable=False` means the system declared it cannot run this axis
    (e.g. text-only system on cross-modal). Reported as 0.0 in the matrix
    per the pre-registration's transparent-failure-reporting commitment.
    """

    system_name: str
    axis: AxisName
    score: float
    applicable: bool = True
    details: Mapping[str, Any] = field(default_factory=dict)
    wall_seconds: float = 0.0
    cost_dollars: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(
                f"AxisResult({self.system_name}, {self.axis.value}): "
                f"score must be in [0, 1], got {self.score}"
            )


@dataclass(frozen=True)
class SystemReport:
    """Full per-system evaluation across all six axes.

    `axes[a]` is the AxisResult for axis a. Missing axes mean the
    evaluation hasn't been run yet (different from applicable=False which
    means it ran but the system declared it cannot answer).
    """

    system_name: str
    axes: Mapping[AxisName, AxisResult]
    notes: str = ""

    def amtb_mean(self) -> float | None:
        """Unweighted mean across all six axes.

        Returns None if not all six axes have results — partial systems
        cannot be aggregated under the pre-registered scoring rules.
        Inapplicable axes contribute 0.0 to the mean (per §7).
        """
        if len(self.axes) != len(AxisName):
            return None
        scores = [self.axes[a].score for a in AxisName]
        return sum(scores) / len(scores)

    def is_complete(self) -> bool:
        return len(self.axes) == len(AxisName)


@runtime_checkable
class MemorySystem(Protocol):
    """Minimal contract a system must satisfy to be evaluable on AMTB.

    Each axis's evaluator calls into the system through the methods
    declared here. Systems can implement only the methods they need;
    `supports(axis)` declares applicability before evaluation.
    """

    name: str

    def supports(self, axis: AxisName) -> bool:
        """Declare whether this system can run on this axis at all.

        Return False to skip evaluation and report 0.0 in the matrix
        with `applicable=False`. The pre-registration's transparent-
        failure-reporting commitment requires reporting 0.0 (not
        omission) for unsupported axes.
        """
        ...


@dataclass
class AMTBMatrix:
    """Aggregated results across all systems and axes.

    The matrix IS the headline output of AMTB. Per the pre-registration,
    we deliberately do NOT publish a single ranking. Systems can be
    Pareto-optimal on different axes; the matrix shows the trade-offs.
    """

    reports: list[SystemReport] = field(default_factory=list)
    pre_registration_commit: str = "745b054"
    pre_registration_date: str = "2026-05-07"
    benchmark_version: str = "0.1.0-pre"

    def add(self, report: SystemReport) -> None:
        # Replace if same name already present
        self.reports = [r for r in self.reports if r.system_name != report.system_name]
        self.reports.append(report)

    def per_axis_percentile_ranks(self) -> dict[AxisName, dict[str, float]]:
        """Per-system percentile rank within each axis. Computed only
        when ≥3 systems are present (per pre-registration §4)."""
        if len(self.reports) < 3:
            return {}
        out: dict[AxisName, dict[str, float]] = {}
        for axis in AxisName:
            scored = [
                (r.system_name, r.axes[axis].score)
                for r in self.reports if axis in r.axes
            ]
            if len(scored) < 3:
                continue
            scored.sort(key=lambda kv: kv[1])
            n = len(scored)
            out[axis] = {
                name: (i + 1) / n for i, (name, _) in enumerate(scored)
            }
        return out


__all__ = ["AMTBMatrix", "AxisName", "AxisResult", "MemorySystem", "SystemReport"]
