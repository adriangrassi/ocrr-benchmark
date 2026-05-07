"""AMTB matrix aggregator and reporting.

Per pre-registration §4: the matrix IS the headline. We refuse to
publish a single ranking. AMTB-mean is reported as a summary, not a
leaderboard score.

This module produces three artifacts:
1. Markdown table for human consumption
2. YAML for machine submission to leaderboard
3. JSON for programmatic comparison
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from amtb.types import AMTBMatrix, AxisName, AxisResult, SystemReport


# Canonical column order — frozen.
AXIS_COLUMN_ORDER: tuple[AxisName, ...] = (
    AxisName.RECALL,
    AxisName.RETENTION,
    AxisName.AUDITABILITY,
    AxisName.CROSS_MODAL,
    AxisName.SCALE,
    AxisName.ADVERSARIAL_REVISION,
)

AXIS_SHORT_LABELS: dict[AxisName, str] = {
    AxisName.RECALL: "Recall",
    AxisName.RETENTION: "Retent",
    AxisName.AUDITABILITY: "Audit",
    AxisName.CROSS_MODAL: "CrossM",
    AxisName.SCALE: "Scale",
    AxisName.ADVERSARIAL_REVISION: "AdvRev",
}


def render_markdown(matrix: AMTBMatrix) -> str:
    """Render the matrix as a Markdown table.

    Per pre-reg §7: cells where a system was evaluated but cannot run
    the axis report 0.0. Cells not yet evaluated are shown as `--`.
    """
    if not matrix.reports:
        return "_no systems evaluated yet_"

    cols = [AXIS_SHORT_LABELS[a] for a in AXIS_COLUMN_ORDER]
    header = "| System | " + " | ".join(cols) + " | AMTB-mean |"
    sep = "|---|" + "|".join("---" for _ in cols) + "|---|"

    rows = []
    for r in matrix.reports:
        cells = []
        for axis in AXIS_COLUMN_ORDER:
            if axis in r.axes:
                ar = r.axes[axis]
                cells.append(f"{ar.score:.3f}" if ar.applicable else f"0.000*")
            else:
                cells.append("--")
        mean = r.amtb_mean()
        mean_cell = f"{mean:.3f}" if mean is not None else "_partial_"
        rows.append(f"| {r.system_name} | " + " | ".join(cells) + f" | {mean_cell} |")

    legend = (
        "\n_`*` = system declared the axis unsupported (per pre-registration "
        "§7, reported as 0.0 not omitted). `--` = axis not yet evaluated. "
        "AMTB-mean reported only when all six axes have results._\n"
    )

    return "\n".join([
        f"# AMTB v{matrix.benchmark_version} — Results Matrix",
        f"_Pre-registered at commit `{matrix.pre_registration_commit}` "
        f"on {matrix.pre_registration_date}._",
        "",
        header, sep, *rows,
        legend,
    ])


def render_yaml(matrix: AMTBMatrix) -> str:
    """Render the matrix as YAML for leaderboard submission."""
    lines = [
        "# AMTB Results — auto-generated, do not hand-edit.",
        f"benchmark_version: {matrix.benchmark_version}",
        f"pre_registration_commit: {matrix.pre_registration_commit}",
        f"pre_registration_date: '{matrix.pre_registration_date}'",
        "systems:",
    ]
    for r in matrix.reports:
        lines.append(f"  - name: {r.system_name}")
        lines.append(f"    notes: {r.notes!r}" if r.notes else "    notes: ''")
        mean = r.amtb_mean()
        lines.append(f"    amtb_mean: {mean if mean is not None else 'null'}")
        lines.append("    axes:")
        for axis in AXIS_COLUMN_ORDER:
            if axis in r.axes:
                ar = r.axes[axis]
                lines.append(f"      {axis.value}:")
                lines.append(f"        score: {ar.score}")
                lines.append(f"        applicable: {str(ar.applicable).lower()}")
                lines.append(f"        wall_seconds: {ar.wall_seconds:.2f}")
                lines.append(f"        cost_dollars: {ar.cost_dollars:.4f}")
            else:
                lines.append(f"      {axis.value}: null  # not yet evaluated")
    return "\n".join(lines) + "\n"


def render_json(matrix: AMTBMatrix) -> str:
    """Render the matrix as JSON for programmatic consumption."""
    obj = {
        "benchmark_version": matrix.benchmark_version,
        "pre_registration_commit": matrix.pre_registration_commit,
        "pre_registration_date": matrix.pre_registration_date,
        "systems": [],
    }
    for r in matrix.reports:
        sys_obj = {
            "name": r.system_name,
            "notes": r.notes,
            "amtb_mean": r.amtb_mean(),
            "axes": {},
        }
        for axis in AXIS_COLUMN_ORDER:
            if axis in r.axes:
                ar = r.axes[axis]
                sys_obj["axes"][axis.value] = {
                    "score": ar.score,
                    "applicable": ar.applicable,
                    "wall_seconds": ar.wall_seconds,
                    "cost_dollars": ar.cost_dollars,
                    "details": dict(ar.details),
                }
            else:
                sys_obj["axes"][axis.value] = None
        obj["systems"].append(sys_obj)
    return json.dumps(obj, indent=2, default=str)


def write_results(matrix: AMTBMatrix, out_dir: Path) -> dict[str, Path]:
    """Write Markdown, YAML, JSON to `out_dir`. Returns paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "markdown": out_dir / "matrix.md",
        "yaml": out_dir / "matrix.yaml",
        "json": out_dir / "matrix.json",
    }
    paths["markdown"].write_text(render_markdown(matrix), encoding="utf-8")
    paths["yaml"].write_text(render_yaml(matrix), encoding="utf-8")
    paths["json"].write_text(render_json(matrix), encoding="utf-8")
    return paths


__all__ = [
    "AXIS_COLUMN_ORDER",
    "AXIS_SHORT_LABELS",
    "render_json",
    "render_markdown",
    "render_yaml",
    "write_results",
]
