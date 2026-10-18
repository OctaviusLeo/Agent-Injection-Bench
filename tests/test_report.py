"""Unit tests for aib.report on a fixture results tree."""

from __future__ import annotations

import json
import shutil

import pytest

from aib import metrics as M
from aib import report as RP
from aib import results_table as RT
from aib.repro import REPO_ROOT
from aib.traceability import check_files
from tests.test_metrics import _grid_records, _write_store

STAMP = {"command": "python -m aib.report", "seed": 20261017, "commit": "1" * 40}


def _hosted_header(store):
    path = store / "gpt-6-luna.jsonl"
    lines = path.read_text().splitlines()
    lines[0] = json.dumps({"header": True, "stamp": {"commit": "a" * 40},
                           "config": {"experiment": {"repetitions": 3, "base_seed": 20261017, "max_iters": 15,
                                                     "max_tokens": 1024, "sampling": {"temperature": 0.7, "top_p": 0.9}}}})
    path.write_text("\n".join(lines) + "\n")


def fixture_root(tmp_path, hosted_run=True):
    root = tmp_path / "repo"
    shutil.copytree(REPO_ROOT / "config", root / "config")
    shutil.copy(REPO_ROOT / "docs" / "report_template.md", (root / "docs").mkdir(parents=True) or root / "docs" / "report_template.md")
    store = root / "results" / "raw"
    local = _grid_records("qwen3-4b")
    for r in local:
        r["sampling"] = {"temperature": 0.7, "top_p": 0.9, "top_k": 0, "min_p": 0.0, "max_tokens": 1024}
    _write_store(store, "qwen3-4b", local)
    _write_store(store, "gpt-6-luna", _grid_records("gpt-6-luna")[:54])
    _hosted_header(store)
    if hosted_run:
        (root / "results" / "hosted_run.json").write_text(json.dumps(
            {"spend_usd": 0.4213, "spend_cap_usd": 3.0, "stopped_at_cap": False, "completed_cases": 54, "planned_cases": 1440}))
    metrics = M.compute_metrics(store, seed=20261017, resamples=300)
    metrics["stamp"] = {"command": "python -m aib.metrics --store results/raw", "seed": 20261017, "commit": "2" * 40}
    (root / "results" / "metrics.json").write_text(json.dumps(metrics))
    (root / "results" / "summary.md").write_text(RT.render_summary(metrics, {"command": "c", "seed": 20261017, "commit": "3" * 40}))
    (root / "results" / "m1_catalog.json").write_text(json.dumps({"tool_count": 24, "user_task_count": 40}))
    (root / "results" / "m3_filter_calibration.json").write_text(json.dumps(
        {"threshold": 0.5, "per_condition": {"clean": {"outputs": 84, "flagged": 36}}}))
    return root


def test_report_renders_all_sections_and_traces(tmp_path):
    root = fixture_root(tmp_path)
    ctx = RP.build_context(root, STAMP)
    text = RP.render((root / "docs" / "report_template.md").read_text(), ctx)
    for section in RP.SECTIONS:
        assert section in text
    assert "temperature 0.7 and top-p 0.9" in text and "ran 3 Repetitions of each task" in text
    assert "![Attack success rate by defense](../results/plots/asr_by_defense.png)" in text
    assert "{{" not in text and "54 of 1440 planned cases" in text and "$0.4213 of the $3.00 cap" in text
    assert "flagged 36 of 84 outputs" in text and "`qwen3-4b`" in text
    (root / "docs" / "REPORT.md").write_text(text)
    result = check_files([root / "docs" / "REPORT.md"], root / "results")[str(root / "docs" / "REPORT.md")]
    assert result.ok, result.untraceable


def test_missing_hosted_summary_is_stated(tmp_path):
    root = fixture_root(tmp_path, hosted_run=False)
    assert "No hosted run summary" in RP.build_context(root, STAMP)["hosted_statement"]


def test_unfilled_placeholder_fails():
    with pytest.raises(RP.RenderError, match="unfilled placeholders: missing_value"):
        RP.render("a {{missing_value}} b {{ok}}", {"ok": 1})
    assert RP.render("x {{ a }} y", {"a": 0}) == "x 0 y"


def test_tables_handle_empty_metrics():
    empty = {"models": {}, "pooled": [], "defense_cost": [], "cells": [], "filter": []}
    assert RP.pooled_table(empty).startswith("_No rows")
    assert RP.findings(empty).startswith("- No Cell")


def test_main_writes_report(tmp_path, monkeypatch, capsys):
    import aib.repro

    root = fixture_root(tmp_path)
    monkeypatch.setattr(aib.repro, "REPO_ROOT", root)
    monkeypatch.setattr(aib.repro, "require_clean", lambda: "4" * 40)
    monkeypatch.setattr(aib.repro, "make_stamp", lambda c, s: aib.repro.Stamp(c, s, "4" * 40))
    assert RP.main([]) == 0
    assert (root / "docs" / "REPORT.md").read_text().startswith("# Agent Injection Bench")
    assert "wrote docs/REPORT.md" in capsys.readouterr().out
