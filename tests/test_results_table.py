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
    assert "Defense cost vs none" in text and "TSR(none) - TSR(defense)" in text


def test_formatting_helpers():
    assert R.fmt_rate(None) == "-"
    assert R.fmt_rate({"point": 0.5, "lower": 0.25, "upper": 0.75}) == "0.500 [0.250, 0.750]"
    assert R.fmt_sampling({"temperature": 0.7, "top_p": 0.9, "seed": "per request", "top_k": None}) == "temperature=0.7, top_p=0.9"


def test_claude_sampling_run_facts_and_judge_section():
    claude = {"temperature": None, "top_p": None, "seed": None, "effort": "medium", "max_tokens": 16000,
              "thinking": "adaptive (model default)", "note": "n"}
    assert R.fmt_sampling(claude) == ("temperature=model default, top_p=model default, max_tokens=16000, effort=medium, "
                                      "thinking=adaptive (model default)")
    facts = {"claude": {"model": "c", "summary": False}}
    assert R.run_fact_lines(facts)[-1] == "- Claude run (c): no Claude run summary (results/claude_run.json) was present."
    facts = {"claude": {"model": "c", "summary": True, "completed_cases": 1440, "planned_cases": 1440, "stopped_at_cap": False,
                        "runs": 2, "spend_usd": "1.7500", "run_cap_usd": "5.00", "effort": "medium", "max_tokens": 16000}}
    lines = R.run_fact_lines(facts)
    assert lines[1].endswith("; the temperature, top-p, and output cap apply to every model but c.")
    assert lines[-1].startswith("- Claude run (c): 1440 of 1440 planned cases completed; recorded spend $1.7500 over 2 runs, "
                                "each under the $5.00 per-run cap; the last run stopped at its cap: no.")
    assert "adaptive thinking at effort medium; output cap 16000 tokens per call" in lines[-1]
    facts["claude"].update(runs=1)
    assert "recorded spend $1.7500 in 1 run, under the $5.00 per-run cap;" in R.run_fact_lines(facts)[-1]
    facts["claude"].update(runs=None, stopped_at_cap=True)
    assert "$1.7500 in its last run, under the $5.00 per-run cap; the last run stopped at its cap: yes." in R.run_fact_lines(facts)[-1]
    assert R.run_fact_lines({})[1].endswith("base seed None.")
    row = {"model": "m", "classified": 30, "judged": 20, "parsed": 18, "refused": 1, "unparsed": 1, "judge_yes_rule_yes": 2,
           "judge_yes_rule_no": 1, "judge_no_rule_yes": 2, "judge_no_rule_no": 13, "cohens_kappa": 0.4, "judge_models": ["j"],
           "agreement": {"point": 0.833, "lower": 0.7, "upper": 0.95, "method": "bootstrap-task-cluster-200"},
           "refusal_categories": {"cyber": 1}, "refused_by_goal": {"injection_task_4": {"refused": 1, "judged": 5}},
           "verdict_commits": {"f" * 40: 19, "e" * 40: 1},
           "stamp": {"command": "python -m aib.judge --store results/raw/m.jsonl", "seed": 0, "commit": "f" * 40}}
    text = "\n".join(R.judge_lines([row, dict(row, model="n", cohens_kappa=None, stamp=None)]))
    assert "## Second-opinion judge" in text and "Judge: j." in text
    assert "| m | 30 | 20 | 18 | 1 | 1 | 2 | 1 | 2 | 13 | 0.833 [0.700, 0.950] | 0.400 |" in text
    assert "| n | 30 | 20 | 18 | 1 | 1 | 2 | 1 | 2 | 13 | 0.833 [0.700, 0.950] | undefined (both always gave the same answer) |" in text
    assert (f"- Verdicts for m: command `python -m aib.judge --store results/raw/m.jsonl`, seed 0, commit `{'f' * 40}`; "
            f"verdict lines by commit: `{'f' * 40}` 19, `{'e' * 40}` 1.") in text
    assert R.commits_text({"verdict_commits": {"a": 3}}) == "" and R.commits_text({}) == ""  # one commit needs no list
    assert R.fmt_kappa(None, 0) == "not computed (no parsed verdicts)" and R.fmt_kappa(0.4, 18) == "0.400"
    assert "- Refused verdicts for m: stop category cyber 1; by injected goal, refused of judged: injection_task_4 1 of 5." in text
    assert "Verdicts for n" not in text and R.judge_lines([]) == []
    assert not text.endswith("\n") and R.judge_lines([dict(row, stamp=None, refused=0)])[-1].startswith("| m |")
    assert R.refusal_text(row, "`") == "stop category `cyber` 1; by injected goal, refused of judged: `injection_task_4` 1 of 5"


def test_claude_takes_the_fourth_palette_slot():
    assert P.model_colors(["claude-haiku-5-5"]) == {"claude-haiku-5-5": P.SERIES[3]}
    assert P._ordered_models({"models": {"claude-haiku-5-5": {}, "gpt-6-luna": {}, "qwen3-4b": {}}}) == [
        "qwen3-4b", "gpt-6-luna", "claude-haiku-5-5"]


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


def test_zero_count_rates_are_marked_and_shared_styles_get_their_own_table(tmp_path):
    metrics = _fixture(tmp_path)
    metrics["cells"][1]["asr"] = M.cluster_bootstrap_rate({"a": (0, 10), "b": (0, 20)}, 1, 100).to_dict()
    text = R.render_summary(metrics, STAMP)
    assert "0.000 [0.000, 0.095]†" in text and R.ZERO_COUNT_NOTE in text
    assert "## Pooled over the styles every model ran" in text and "Styles: direct." in text
    assert "- Bootstrap diagnostics:" in text
    assert R.diagnostics_lines(None) == []
