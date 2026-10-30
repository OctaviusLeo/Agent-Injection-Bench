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
from tests.test_metrics import _claude_store, _grid_records, _write_store

STAMP = {"command": "python -m aib.report", "seed": 20261017, "commit": "1" * 40}


def _hosted_header(store):
    path = store / "gpt-6-luna.jsonl"
    lines = path.read_text().splitlines()
    lines[0] = json.dumps({"header": True, "stamp": {"commit": "a" * 40},
                           "config": {"experiment": {"repetitions": 3, "base_seed": 20261017, "max_iters": 15,
                                                     "max_tokens": 1024, "sampling": {"temperature": 0.7, "top_p": 0.9}}}})
    path.write_text("\n".join(lines) + "\n")


JUDGE_STAMP = {"command": "python -m aib.judge --store results/raw/qwen3-4b.jsonl", "seed": 0, "commit": "7" * 40}


def _claude_results(root, local):
    """The Claude target's store, run summary, and ledger, and second-opinion verdicts on the local store."""
    results = root / "results"
    _claude_store(results / "raw")
    (results / "claude_run.json").write_text(json.dumps(
        {"spend_usd": 0.15, "spend_cap_usd": 5.0, "stopped_at_cap": False, "completed_cases": 162, "planned_cases": 1440}))
    ledger = [{"purpose": "benchmark:claude-haiku-5-5", "run_id": run, "cost_usd": cost, "event": "request",
               "ts": "2026-10-30T05:20:00+00:00"} for run, cost in (("a", 0.25), ("b", 0.15))]
    (results / "claude_ledger.jsonl").write_text("\n".join(json.dumps(r) for r in ledger) + "\n")
    verdicts = [{"case_id": r["case_id"], "outcome": "parsed", "carried_out": r["injection_success"] != (i % 7 == 0),
                 "agentdojo": r["injection_success"], "agree": i % 7 != 0, "judge_model": "claude-sonnet-5-5",
                 "injection_task_id": f"injection_task_{i % 6}", "stop_category": None}
                for i, r in enumerate(x for x in local if x["status"] == "classified")]
    for i in (5, 11):  # two refusals, both on injection_task_5
        verdicts[i].update(outcome="refused", carried_out=None, agree=None, stop_category="cyber")
    (results / "judge").mkdir()
    (results / "judge" / "qwen3-4b.jsonl").write_text("\n".join(json.dumps(v) for v in verdicts) + "\n")
    (results / "judge" / "qwen3-4b_agreement.json").write_text(json.dumps({"stamp": JUDGE_STAMP}))


def fixture_root(tmp_path, hosted_run=True, claude=False):
    root = tmp_path / "repo"
    shutil.copytree(REPO_ROOT / "config", root / "config")
    shutil.copy(REPO_ROOT / "docs" / "report_template.md", (root / "docs").mkdir(parents=True) or root / "docs" / "report_template.md")
    store = root / "results" / "raw"
    local = _grid_records("qwen3-4b")
    for r in local:
        r["sampling"] = {"temperature": 0.7, "top_p": 0.9, "top_k": 0, "min_p": 0.0, "max_tokens": 1024}
    _write_store(store, "qwen3-4b", local)
    hosted = _grid_records("gpt-6-luna")[:54]
    for r in hosted:
        r.update(sampling={"temperature": 0.7, "top_p": 0.9, "seed": "per request", "reasoning_effort": "none"},
                 response_models=["gpt-6-luna"])
    _write_store(store, "gpt-6-luna", hosted)
    _hosted_header(store)
    if hosted_run:
        (root / "results" / "hosted_run.json").write_text(json.dumps(
            {"spend_usd": 0.4213, "spend_cap_usd": 3.0, "stopped_at_cap": False, "completed_cases": 54, "planned_cases": 1440}))
    if claude:
        _claude_results(root, local)
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


def test_report_with_the_claude_target_and_the_judge_traces(tmp_path):
    root = fixture_root(tmp_path, claude=True)
    text = RP.render((root / "docs" / "report_template.md").read_text(), RP.build_context(root, STAMP))
    assert "Every model but `claude-haiku-5-5` ran with temperature 0.7 and top-p 0.9, with at most 1024 output tokens" in text
    assert ("`claude-haiku-5-5` sent no sampling settings and no seed (Claude models accept only their default sampling, so "
            "none is sent); it thought adaptively at effort `medium`, with at most 16000 output tokens per call") in text
    assert "Each of their requests carried a seed derived from the case seed." in text
    assert ("The Claude target `claude-haiku-5-5` ran the same conditions in the same order, under a cap of $5.00 per run: "
            "162 of 1440 planned cases completed in 2 runs; recorded spend was $0.4000") in text
    assert "Its last run ended below the Run_Cap." in text and "The OpenAI model `gpt-6-luna` ran all" in text
    assert "- **The two hosted models are not a controlled comparison.** `claude-haiku-5-5` sent no sampling" in text
    assert "`gpt-6-luna` ran at temperature 0.7 and top-p 0.9 with a per-request seed and reasoning effort `none`" in text
    assert "on `claude-haiku-5-5`'s cases echoed the requested ID, so no dated snapshot was recorded." in text
    assert "one small quantized local model, and two hosted model IDs at one point in time." in text
    assert "**Second-opinion judge.** `claude-sonnet-5-5` read classified attack cases of `qwen3-4b`" in text
    assert f"- `qwen3-4b` verdicts: command `{JUDGE_STAMP['command']}`, seed 0, commit `{JUDGE_STAMP['commit']}`." in text
    assert "| `qwen3-4b` | 108 | 108 | 106 | 2 | 0 |" in text and "{{" not in text
    assert ("Every refused verdict had stop category `cyber` and came from injected goal `injection_task_5`; refused "
            "verdicts are left out of the agreement") in text
    assert ("- `qwen3-4b` refused verdicts: stop category `cyber` 2; by injected goal, refused of judged: `injection_task_5` "
            "2 of 18.") in text
    (root / "docs" / "REPORT.md").write_text(text)
    result = check_files([root / "docs" / "REPORT.md"], root / "results")[str(root / "docs" / "REPORT.md")]
    assert result.ok, result.untraceable


def test_claude_statements_without_their_inputs():
    metrics = {"models": {"c": {"sampling": {"temperature": None, "effort": "medium"}, "response_models": ["c-2026"]},
                          "h": {"sampling": {"temperature": 0.7, "top_p": 0.9, "reasoning_effort": "none"}}}}
    assert "response `model` field on `c`'s cases recorded `c-2026`." in RP.comparability_lines(metrics, "h")[0]
    metrics["models"]["c"]["response_models"] = []
    assert RP.comparability_lines(metrics, "h")[0].endswith("on `c`'s cases was not recorded.")
    assert RP.comparability_lines(metrics, "absent") == []  # no OpenAI model to compare with
    assert RP.judge_section({}) == "No second-opinion verdicts were present when this report was generated."
    facts = {"n_defenses": 4, "claude": {"model": "c", "summary": False}}
    assert RP.hosted_statement(facts, ["clean"], 6, "h").endswith(
        "No run summary for the Claude target `c` was present when this report was generated.")
    facts["claude"].update(summary=True, completed_cases=9, planned_cases=9, stopped_at_cap=True, runs=None,
                           spend_usd="0.0100", run_cap_usd="5.00")
    text = RP.hosted_statement(facts, ["clean", "direct"], 6, "h", ["clean"])
    assert "No hosted run summary was present" in text and "ran 1 of the conditions (`clean`) in the same order" in text
    assert "9 of 9 planned cases completed; recorded spend" in text and "Its last run stopped at the Run_Cap." in text


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
