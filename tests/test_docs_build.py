"""Unit tests for aib.docs_build (README generation) on the report fixture tree."""

from __future__ import annotations

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
    assert "This repository is private." in text and "[docs/REPORT.md](docs/REPORT.md)" in text
    assert "`bash scripts/test_unit.sh`" in text and "`uv run --frozen python scripts/ci_smoke.py`" in text
    assert "`bash scripts/run_all_local.sh`" in text and "`bash scripts/build_results.sh`" in text
    assert "All tools are simulated" in text and "Responsible disclosure" in text
    assert "| `none` |" not in text and "| qwen3-4b | none |" in text  # the pooled table copied from summary.md
    assert "{{" not in text
    (root / "README.md").write_text(text)
    result = check_files([root / "README.md"], root / "results")[str(root / "README.md")]
    assert result.ok, result.untraceable


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
    assert capsys.readouterr().out.split() == ["qwen3-4b", "llama3.1-8b"]
