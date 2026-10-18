"""Unit tests for aib.results_table and aib.plots on a small metrics fixture."""

from __future__ import annotations

import json
import re

from aib import metrics as M
from aib import plots as P
from aib import results_table as R
from tests.test_metrics import _grid_records, _write_store

STAMP = {"command": "python -m aib.results_table", "seed": 7, "commit": "d" * 40}


def _fixture(tmp_path):
    store = tmp_path / "raw"
    _write_store(store, "qwen3-4b", _grid_records("qwen3-4b"))
    _write_store(store, "gpt-6-luna", _grid_records("gpt-6-luna")[:30])
    metrics = M.compute_metrics(store, seed=7, resamples=300)
    metrics["stamp"] = {"command": "python -m aib.metrics", "seed": 7, "commit": "e" * 40}
    return metrics


def test_every_rate_has_an_interval_and_not_computed_shows_cause(tmp_path):
    metrics = _fixture(tmp_path)
    metrics["cells"][1]["asr"] = {"not_computed": True, "cause": "no classified injected cases (all unexposed or errored)"}
    text = R.render_summary(metrics, STAMP)
    assert "not computed (no classified injected cases (all unexposed or errored))" in text
    rows = [line for line in text.splitlines() if line.startswith("| qwen3-4b |") or line.startswith("| gpt-6-luna |")]
    for row in rows:
        for cell in row.split("|"):
            if re.fullmatch(r"\s*[01]\.\d{3}\s*", cell):
                raise AssertionError(f"rate without interval: {row}")
    assert re.search(r"\d\.\d{3} \[\d\.\d{3}, \d\.\d{3}\]", text)
    assert f"commit `{'d' * 40}`" in text and f"commit `{'e' * 40}`" in text


def test_defense_cost_rows_reference_none(tmp_path):
    text = R.render_summary(_fixture(tmp_path), STAMP)
    cells = [line for line in text.splitlines() if line.startswith("| qwen3-4b | none |")]
    assert cells and all("| reference |" in line for line in cells)
    assert "Defense_Cost vs none" in text and "TSR(none) - TSR(defense)" in text


def test_formatting_helpers():
    assert R.fmt_rate(None) == "-"
    assert R.fmt_rate({"point": 0.5, "lower": 0.25, "upper": 0.75}) == "0.500 [0.250, 0.750]"
    assert R.fmt_sampling({"temperature": 0.7, "top_p": 0.9, "seed": "per request", "top_k": None}) == "temperature=0.7, top_p=0.9"


def test_plots_are_written_and_non_empty(tmp_path):
    metrics = _fixture(tmp_path)
    a = P.plot_asr_by_defense(metrics, tmp_path / "plots" / "asr_by_defense.png")
    b = P.plot_defense_cost(metrics, tmp_path / "plots" / "defense_cost.png")
    for p in (a, b):
        assert p.exists() and p.stat().st_size > 5000 and p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert P.model_colors(["gpt-6-luna", "qwen3-4b", "qwen2.5-7b", "other"]) == {
        "gpt-6-luna": P.SERIES[2], "qwen3-4b": P.SERIES[0], "qwen2.5-7b": P.SERIES[1], "other": P.OTHER}


def test_clis_write_outputs(tmp_path, monkeypatch, capsys):
    import aib.repro

    metrics = _fixture(tmp_path)
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "metrics.json").write_text(json.dumps(metrics))
    monkeypatch.setattr(aib.repro, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(aib.repro, "require_clean", lambda: "f" * 40)
    monkeypatch.setattr(aib.repro, "make_stamp", lambda c, s: aib.repro.Stamp(c, s, "f" * 40))
    assert R.main([]) == 0 and P.main([]) == 0
    assert (tmp_path / "results" / "summary.md").read_text().startswith("# Results_Table")
    assert (tmp_path / "results" / "plots" / "defense_cost.png").stat().st_size > 0
    assert "wrote results/summary.md" in capsys.readouterr().out
