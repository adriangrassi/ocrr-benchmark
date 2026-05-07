"""Tests for AMTB matrix aggregation + reporting."""
from __future__ import annotations

import json

import pytest

from amtb.matrix import (
    AXIS_COLUMN_ORDER,
    render_json,
    render_markdown,
    render_yaml,
    write_results,
)
from amtb.types import AMTBMatrix, AxisName, AxisResult, SystemReport


def _stub_report(name: str, scores: dict[AxisName, float], applicable: dict | None = None) -> SystemReport:
    applicable = applicable or {}
    axes = {
        a: AxisResult(
            system_name=name, axis=a, score=s,
            applicable=applicable.get(a, True),
        )
        for a, s in scores.items()
    }
    return SystemReport(system_name=name, axes=axes)


def test_matrix_markdown_renders():
    m = AMTBMatrix()
    m.add(_stub_report("horizon", {a: 0.85 for a in AxisName}))
    m.add(_stub_report("baseline", {a: 0.40 for a in AxisName}))
    md = render_markdown(m)
    assert "horizon" in md
    assert "baseline" in md
    assert "AMTB-mean" in md
    assert "0.850" in md
    assert "0.400" in md
    assert "Pre-registered at commit `745b054`" in md


def test_matrix_amtb_mean_only_when_complete():
    m = AMTBMatrix()
    # Only 3 of 6 axes — AMTB-mean is None
    partial = SystemReport(
        system_name="partial",
        axes={
            AxisName.RECALL: AxisResult(system_name="partial", axis=AxisName.RECALL, score=0.5),
            AxisName.RETENTION: AxisResult(system_name="partial", axis=AxisName.RETENTION, score=0.9),
            AxisName.AUDITABILITY: AxisResult(system_name="partial", axis=AxisName.AUDITABILITY, score=1.0),
        },
    )
    m.add(partial)
    md = render_markdown(m)
    assert "_partial_" in md  # amtb-mean cell shows as _partial_


def test_matrix_unapplicable_marked():
    """Per pre-reg §7, unapplicable cells show 0.0 with a star marker."""
    m = AMTBMatrix()
    scores = {a: 0.5 for a in AxisName}
    scores[AxisName.CROSS_MODAL] = 0.0
    m.add(_stub_report(
        "text_only_system", scores,
        applicable={AxisName.CROSS_MODAL: False},
    ))
    md = render_markdown(m)
    assert "0.000*" in md  # unapplicable marker


def test_matrix_yaml_round_trip():
    m = AMTBMatrix()
    m.add(_stub_report("horizon", {a: 0.85 for a in AxisName}))
    yaml_text = render_yaml(m)
    assert "benchmark_version: 0.1.0-pre" in yaml_text
    assert "horizon" in yaml_text
    assert "amtb_mean: 0.85" in yaml_text


def test_matrix_json_parses():
    m = AMTBMatrix()
    m.add(_stub_report("horizon", {a: 0.85 for a in AxisName}))
    j = render_json(m)
    obj = json.loads(j)
    assert obj["benchmark_version"] == "0.1.0-pre"
    assert obj["pre_registration_commit"] == "745b054"
    assert obj["systems"][0]["name"] == "horizon"
    assert obj["systems"][0]["amtb_mean"] == 0.85


def test_matrix_replace_on_re_add():
    m = AMTBMatrix()
    m.add(_stub_report("horizon", {a: 0.5 for a in AxisName}))
    m.add(_stub_report("horizon", {a: 0.8 for a in AxisName}))  # re-add
    assert len(m.reports) == 1
    assert m.reports[0].axes[AxisName.RECALL].score == 0.8


def test_matrix_percentile_ranks_need_3_systems():
    m = AMTBMatrix()
    m.add(_stub_report("a", {a: 0.5 for a in AxisName}))
    m.add(_stub_report("b", {a: 0.6 for a in AxisName}))
    assert m.per_axis_percentile_ranks() == {}  # need ≥3
    m.add(_stub_report("c", {a: 0.4 for a in AxisName}))
    ranks = m.per_axis_percentile_ranks()
    assert AxisName.RECALL in ranks
    assert ranks[AxisName.RECALL]["b"] > ranks[AxisName.RECALL]["a"] > ranks[AxisName.RECALL]["c"]


def test_write_results_creates_three_files(tmp_path):
    m = AMTBMatrix()
    m.add(_stub_report("horizon", {a: 0.85 for a in AxisName}))
    paths = write_results(m, tmp_path)
    assert paths["markdown"].exists()
    assert paths["yaml"].exists()
    assert paths["json"].exists()
    # Verify json is valid
    json.loads(paths["json"].read_text())
