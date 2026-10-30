"""Unit tests for aib.docs_build (README generation) on the report fixture tree."""

from __future__ import annotations

import json
import shutil

import pytest

from aib import config as CFG
from aib import docs_build as DB
from aib.report import render
from aib.repro import REPO_ROOT
from aib.traceability import check_files
from tests.test_report import fixture_root

STAMP = {"command": "python -m aib.docs_build", "seed": 20261017, "commit": "5" * 40}
SECTIONS = ("## Results", "## How it works", "## Run it", "## Reproduce each result", "## Tests", "## Known limitations",
            "## Safety and scope", "```mermaid")


def test_readme_renders_every_required_section_and_traces(tmp_path):
    root = fixture_root(tmp_path)
    shutil.copy(REPO_ROOT / "docs" / "README.template.md", root / "docs" / "README.template.md")
    text = render((root / "docs" / "README.template.md").read_text(), DB.build_context(root, STAMP))
    for s in SECTIONS:
        assert s in text
    assert "[MIT License](LICENSE)" in text and "[docs/REPORT.md](docs/REPORT.md)" in text
    assert "## How this was built" in text  # the owner's section is carried through generation
    assert "`bash scripts/test_unit.sh`" in text and "`uv run --frozen python scripts/ci_smoke.py`" in text
    assert "`bash scripts/run_all_local.sh`" in text and "`bash scripts/build_results.sh`" in text
    assert "All tools are simulated" in text and "Responsible disclosure" in text
    assert "| `none` |" not in text and "| qwen3-4b | none |" in text  # the pooled table copied from summary.md
    assert "- **DeBERTa filter:**" in text and "queried on the dates recorded in" in text  # headline and hosted dates
    assert "{{" not in text
    (root / "README.md").write_text(text)
    result = check_files([root / "README.md"], root / "results")[str(root / "README.md")]
    assert result.ok, result.untraceable


def test_readme_names_both_hosted_models_and_the_judge_and_traces(tmp_path):
    root = fixture_root(tmp_path, claude=True)
    shutil.copy(REPO_ROOT / "docs" / "README.template.md", root / "docs" / "README.template.md")
    text = render((root / "docs" / "README.template.md").read_text(), DB.build_context(root, STAMP))
    assert ("and two hosted models: OpenAI's `gpt-6-luna` (API model ID), queried on the dates recorded in "
            "`results/hosted_ledger.jsonl` on 2 of the styles to stay within its spend cap, and Anthropic's `claude-haiku-5-5`, "
            "queried on 2026-10-30 (UTC) on the same styles, so that the two hosted models share a grid.") in text
    assert "- **Second-opinion judge:** `claude-sonnet-5-5`, asked whether the agent's tool calls carried out" in text
    assert "`qwen3-4b` 90 of 106 parsed verdicts (2 refused), " in text and "The judge changes no rate" in text
    assert "- OpenAI hosted run: 54 of 1440 planned cases completed" in text
    assert "- The OpenAI hosted model needs `OPENAI_API_KEY`" in text
    assert "- Claude run (claude-haiku-5-5): 162 of 1440 planned cases completed; recorded spend $0.4000 over 2 runs" in text
    assert "- **The two hosted models are not a controlled comparison.**" in text
    assert "bash scripts/run_benchmark.sh claude-haiku-5-5" in text and "{{" not in text
    (root / "README.md").write_text(text)
    result = check_files([root / "README.md"], root / "results")[str(root / "README.md")]
    assert result.ok, result.untraceable
    cfg = CFG.load_all(root / "config")
    cfg = cfg.model_copy(update={"claude": cfg.claude.model_copy(update={"conditions": ["clean", "direct"]})})
    assert "`c`, queried on the dates recorded in `results/claude_ledger.jsonl` on 1 of the styles" in DB.hosted_models(
        root, cfg, {"claude": {"model": "c"}})
    partial = {"model": "m", "classified": 10, "judged": 2, "parsed": 2, "refused": 0, "judge_models": ["j"], "cohens_kappa": None,
               "agreement": {"point": 1.0, "lower": 1.0, "upper": 1.0, "successes": 2, "n": 2}}
    assert "`m` 2 of 2 parsed verdicts, 1.000 [1.000, 1.000], Cohen's kappa undefined (both always gave the same answer) (2 of 10 " \
           "classified cases judged)." in DB.headline({"judge": [partial]})


def test_summary_helpers_fail_loudly():
    with pytest.raises(ValueError, match="no section"):
        DB.summary_section("# x\n", "Run facts")
    assert DB.summary_stamps("no stamps") == {}


def test_main_writes_readme(tmp_path, monkeypatch, capsys):
    import aib.repro

    root = fixture_root(tmp_path)
    shutil.copy(REPO_ROOT / "docs" / "README.template.md", root / "docs" / "README.template.md")
    monkeypatch.setattr(aib.repro, "REPO_ROOT", root)
    monkeypatch.setattr(aib.repro, "require_clean", lambda: "6" * 40)
    monkeypatch.setattr(aib.repro, "make_stamp", lambda c, s: aib.repro.Stamp(c, s, "6" * 40))
    assert DB.main([]) == 0
    assert (root / "README.md").read_text().startswith("# Agent Injection Bench")
    assert "wrote README.md" in capsys.readouterr().out


def test_config_local_models_cli(capsys):
    assert CFG.main(["local-models"]) == 0
    assert capsys.readouterr().out.split() == ["qwen3-4b", "qwen2.5-7b"]


def _r(point, lower, upper):
    return {"point": point, "lower": lower, "upper": upper}


def test_headline_reports_filter_range_and_prompt_defense_exceptions():
    zero = _r(0.0, -0.02, 0.02)
    metrics = {
        "pooled": [{"model": "m1", "defense": "none", "asr": _r(0.03, 0.01, 0.05)},
                   {"model": "m2", "defense": "none", "asr": {"not_computed": True, "cause": "x"}}],
        "defense_cost": [
            {"model": "m1", "defense": "filter", "condition": "pooled", "defense_cost": _r(0.3, 0.2, 0.4),
             "asr_reduction": _r(0.02, 0.005, 0.04)},
            {"model": "m2", "defense": "filter", "condition": "pooled", "defense_cost": _r(0.5, 0.4, 0.6),
             "asr_reduction": _r(0.0, 0.0, 0.0)},
            {"model": "m1", "defense": "delimit", "condition": "pooled", "defense_cost": zero,
             "asr_reduction": _r(-0.017, -0.037, -0.001)},
            {"model": "m1", "defense": "hierarchy", "condition": "pooled", "defense_cost": _r(0.05, 0.01, 0.09),
             "asr_reduction": zero},
            {"model": "m1", "defense": "filter", "condition": "clean", "defense_cost": _r(0.9, 0.8, 1.0),
             "asr_reduction": zero},
        ],
    }
    text = DB.headline(metrics)
    assert "`m1` 0.030 [0.010, 0.050]." in text and "`m2`" not in text.split("\n")[0]
    assert "by 0.300 [0.200, 0.400] (`m1`) to 0.500 [0.400, 0.600] (`m2`)" in text  # the clean-condition row is ignored
    assert "largest drop in attack success was 0.020 [0.005, 0.040] (`m1`)" in text
    assert "includes zero in 1 of 2 model-defense pairs, and for the change in attack success in 1 of 2" in text
    assert "`m1` under `hierarchy`, task-success cost 0.050 [0.010, 0.090]" in text
    assert "`m1` under `delimit`, attack-success reduction -0.017 [-0.037, -0.001] (attack success rose)" in text
    assert DB.headline({}) == "- No pooled comparison had enough classified cases for a finding."


def test_headline_uses_shared_styles_floor_and_filter_flag_rate():
    asr = lambda s, n, p: dict(_r(p, 0.0, p * 2), successes=s, n=n)  # noqa: E731
    metrics = {
        "pooled": [{"model": "h", "defense": "none", "conditions": ["direct", "important_instructions"], "asr": asr(1, 233, 0.004)},
                   {"model": "q", "defense": "none", "conditions": ["direct", "x", "important_instructions"],
                    "asr": asr(14, 470, 0.030)}],
        "shared_pooled": [{"model": "h", "defense": "none", "conditions": ["direct", "important_instructions"],
                           "asr": asr(1, 233, 0.004)},
                          {"model": "q", "defense": "none", "conditions": ["direct", "important_instructions"],
                           "asr": asr(12, 190, 0.063)}],
        "cells": [{"model": "q", "defense": "none", "condition": "important_instructions", "asr": asr(12, 97, 0.124)},
                  {"model": "q", "defense": "none", "condition": "x", "asr": asr(2, 95, 0.021)},
                  {"model": "h", "defense": "none", "condition": "direct", "asr": asr(1, 117, 0.009)}],
        "defense_cost": [{"model": "q", "defense": "filter", "condition": "pooled", "defense_cost": _r(0.3, 0.2, 0.4),
                          "asr_reduction": _r(0.02, 0.005, 0.04)}],
        "filter": [{"model": "q", "condition": "clean", "false_positive_rate": _r(0.334, 0.3, 0.37)},
                   {"model": "h", "condition": "clean", "false_positive_rate": _r(0.401, 0.35, 0.45)},
                   {"model": "h", "condition": "pooled", "false_positive_rate": _r(0.1, 0.05, 0.15)}],
    }
    text = DB.headline(metrics)
    assert "on the styles every model ran (`direct`, `important_instructions`)" in text
    assert "`h` 1 of 233 classified cases, 0.004 [0.000, 0.008], `q` 12 of 190 classified cases" in text
    assert "which differ (2 for `h`, 3 for `q`)" in text
    assert "at most 0.030 [0.000, 0.060] (`q`, pooled over its styles)" in text
    assert "`q` 12 of 14 under `important_instructions` (0.124 [0.000, 0.248]), `h` 1 of 1 under `direct`" in text
    assert "flagged 0.334 [0.300, 0.370] (`q`) to 0.401 [0.350, 0.450] (`h`) of benign tool outputs" in text


def test_hosted_dates_from_the_ledger(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    assert "results/hosted_ledger.jsonl" in DB.hosted_dates(ledger, "h")
    rows = [{"purpose": "m0_hosted_check", "ts": "2026-10-17T23:00:00+00:00"},
            {"purpose": "benchmark:h", "ts": "2026-10-18T02:45:58+00:00"},
            {"purpose": "benchmark:h", "ts": "2026-10-18T03:53:57+00:00"}]
    ledger.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert DB.hosted_dates(ledger, "h") == "on 2026-10-18 (UTC)"
    ledger.write_text(ledger.read_text() + json.dumps({"purpose": "benchmark:h", "ts": "2026-10-19T00:00:01+00:00"}) + "\n")
    assert DB.hosted_dates(ledger, "h") == "from 2026-10-18 to 2026-10-19 (UTC)"
