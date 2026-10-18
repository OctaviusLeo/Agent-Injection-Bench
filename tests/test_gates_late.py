"""Unit tests for the M0, M3, M4, and M5 gate checks in aib.gates, on fixture trees."""

from __future__ import annotations

import json
import shutil

from aib import gates as G
from aib import metrics as M
from aib.config import load_all
from aib.experiments import build_grid
from aib.repro import REPO_ROOT
from aib.results_table import render_summary

CFG = load_all()
GOOD_STAMP = {"command": "python scripts/x.py", "seed": 0, "commit": "a" * 40}
DESIGN = ("# D\n\n## M0 outcome\n\n- a: done\n\n## Next\n\n### Responsible disclosure\n"
          "the owner follows the vendor's responsible-disclosure process\n")


def _write(root, rel, data):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data))


def _tree(tmp_path):
    root = tmp_path / "repo"
    shutil.copytree(REPO_ROOT / "config", root / "config")
    (root / "docker").mkdir()
    shutil.copy(REPO_ROOT / "docker" / "compose.yaml", root / "docker" / "compose.yaml")
    for f in (".gitignore", ".gitattributes"):
        shutil.copy(REPO_ROOT / f, root / f)
    spec = root / ".kiro" / "specs" / "agent-injection-bench"
    spec.mkdir(parents=True)
    (spec / "design.md").write_text(DESIGN)
    return root


def _m0_files(root):
    files = {"qwen3-4b": CFG.models.models["qwen3-4b"].files, "llama3.1-8b": CFG.models.models["llama3.1-8b"].files,
             "classifier": CFG.models.classifier.files}
    entries = {k: {"files": [{"name": f.name, "size": f.size, "size_ok": True, "sha256": "a" * 64} for f in fs]}
               for k, fs in files.items()}
    _write(root, "results/m0_models.json", {"stamp": GOOD_STAMP, "entries": entries})
    for key in ("qwen3-4b", "llama3.1-8b"):
        _write(root, f"results/m0_smoke_{key}.json", {"stamp": GOOD_STAMP, "python_ok": True, "docker_ok": True,
                                                       "response": "ok", "peak_mib": 9000, "total_mib": 10240})
        _write(root, f"results/m0_tool_probe_{key}.json", {"passed": True, "tool_calls": [{"function": "get_current_day"}]})
    _write(root, "results/m0_hosted_check.json", {"tool_free": {"content": "x"},
                                                  "tool_request": {"tool_calls": [{"function": "get_current_day"}]}})
    (root / "docs" / "evidence").mkdir(parents=True)
    (root / "docs" / "evidence" / "a.txt").write_text("x")
    (root / "docs" / "ENVIRONMENT.md").write_text("# Env\n\n| x | `docs/evidence/a.txt` | `tests/test_llm.py::t` |\n")
    (root / "tests").mkdir()
    (root / "tests" / "test_llm.py").write_text("")
    rows = ["| AgentDojo | 0.1.35 | MIT |", "| InjecAgent | main | MIT |", "| llama.cpp | b11580 | MIT |",
            f"| {CFG.models.classifier.repo} | r | Apache-2.0 |",
            f"| {CFG.models.models['qwen3-4b'].repo} | r | Apache-2.0 |",
            f"| {CFG.models.models['llama3.1-8b'].repo} | r | llama3.1 |"]
    (root / "THIRD_PARTY.md").write_text("\n".join(rows) + "\n")


def test_m0_passes_and_reports_problems(tmp_path):
    root = _tree(tmp_path)
    _m0_files(root)
    assert G.check_m0(root) == []
    (root / "docs" / "ENVIRONMENT.md").write_text("TBD `docs/evidence/missing.txt`\n")
    (root / "results" / "m0_smoke_qwen3-4b.json").unlink()
    _write(root, "results/m0_tool_probe_llama3.1-8b.json", {"passed": False, "tool_calls": []})
    (root / "THIRD_PARTY.md").write_text("| AgentDojo | x | MIT |\n")
    (root / "docker" / "compose.yaml").write_text("image: x:y\n")
    (root / ".gitignore").write_text("runs/\n")
    (root / ".gitattributes").write_text("")
    _write(root, "results/m0_models.json", {"entries": {}})
    _write(root, "results/m0_hosted_check.json", {"tool_free": {"content": " "}, "tool_request": {"tool_calls": []}})
    (root / "results" / "hosted_ledger.jsonl").write_text(json.dumps({"cost_usd": 3.5}) + "\n")
    (root / ".kiro/specs/agent-injection-bench/design.md").write_text("## M0 outcome\n\n- a: TBD\n\n## X\n")
    text = "\n".join(G.check_m0(root))
    for needle in ("TBD", "missing path", "m0_smoke_qwen3-4b.json is missing", "m0_tool_probe_llama3.1-8b", "InjecAgent",
                   "digest", ".gitignore lacks", ".gitattributes", "m0_models: missing stamp", "empty tool-free",
                   "no tool call", "M0 outcome", "exceeds the Spend_Cap"):
        assert needle in text, needle
    (root / "results" / "m0_models.json").unlink()
    (root / "results" / "m0_hosted_check.json").unlink()
    text = "\n".join(G.check_m0(root))
    assert "m0_models.json is missing" in text and "m0_hosted_check.json is missing" in text
    assert G.cited_paths("`docs/a.md` `results/raw/<m>.jsonl` `tests/t.py::x`") == ["docs/a.md", "tests/t.py"]


def _records(model, specs, status="classified", decisions=True):
    out = []
    for s in specs:
        st = "clean" if s.condition == "clean" else status
        dec = ([{"injected": False, "flagged": False, "score": 0.0, "output_sha256": "x"}]
               if s.defense == "filter" and decisions else [])
        out.append({"case_id": s.case_id, "model": model, "defense": s.defense, "condition": s.condition,
                    "user_task_id": s.user_task_id, "injection_task_id": s.injection_task_id, "rep": s.rep, "status": st,
                    "exposed": None if st == "clean" else True, "injection_success": False if st == "classified" else None,
                    "task_success": True, "filter_decisions": dec, "n_tool_outputs": 1 if s.defense == "filter" else 0,
                    "sampling": {"temperature": 0.7, "top_p": 0.9}, "model_version": "v", "response_models": []})
    return out


def _store_file(root, model, records):
    p = root / "results" / "raw" / f"{model}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"header": True, "stamp": GOOD_STAMP, "config": {}})] + [json.dumps(r) for r in records]
    p.write_text("\n".join(lines) + "\n")


def test_m3_store_checks(tmp_path):
    root = _tree(tmp_path)
    for key in ("qwen3-4b", "llama3.1-8b"):
        _store_file(root, key, _records(key, build_grid(key, CFG.experiment)))
    _write(root, "results/m3_filter_calibration.json", {"stamp": GOOD_STAMP})
    _write(root, "results/pilot.json", {"stamp": GOOD_STAMP})
    assert G.check_m3(root) == []
    bad = _records("qwen3-4b", build_grid("qwen3-4b", CFG.experiment)[:2000], status="errored", decisions=False)
    bad.append(dict(bad[0]))
    bad.append(dict(bad[1], status="weird", defense="filter", n_tool_outputs=3))
    _store_file(root, "qwen3-4b", bad)
    (root / "results" / "pilot.json").unlink()
    text = "\n".join(G.check_m3(root))
    for needle in ("duplicate case ids", "grid cases missing", "invalid statuses", "exceed 5%", "a decision for every",
                   "pilot.json"):
        assert needle in text, needle
    assert "missing" in G.check_store(root / "results" / "raw" / "nope.jsonl", set())[0]


def _m4_tree(tmp_path, complete=True):
    root = _tree(tmp_path)
    _store_file(root, "qwen3-4b", _records("qwen3-4b", build_grid("qwen3-4b", CFG.experiment, pilot=True)))
    hosted_specs = build_grid("gpt-6-luna", CFG.experiment, hosted=True, hosted_conditions=CFG.hosted.conditions,
                              hosted_defenses=CFG.hosted.defenses)
    _store_file(root, "gpt-6-luna", _records("gpt-6-luna", hosted_specs if complete else hosted_specs[:100]))
    _write(root, "results/hosted_run.json", {"stamp": GOOD_STAMP, "stopped_at_cap": not complete, "spend_usd": 0.5,
                                             "spend_cap_usd": 3.0, "completed_cases": 1440 if complete else 100,
                                             "planned_cases": 1440})
    (root / "results" / "hosted_ledger.jsonl").write_text(json.dumps({"cost_usd": 0.5}) + "\n")
    metrics = M.compute_metrics(root / "results" / "raw", 5, 50)
    metrics["stamp"] = GOOD_STAMP
    _write(root, "results/metrics.json", metrics)
    (root / "results" / "summary.md").write_text(render_summary(metrics, GOOD_STAMP))
    (root / "results" / "plots").mkdir()
    for p in ("asr_by_defense.png", "defense_cost.png"):
        (root / "results" / "plots" / p).write_bytes(b"png")
    return root


def test_m4_passes_on_consistent_tree_and_with_cap_stop(tmp_path):
    assert G.check_m4(_m4_tree(tmp_path / "a")) == []
    assert G.check_m4(_m4_tree(tmp_path / "b", complete=False)) == []


def test_m4_reports_problems(tmp_path):
    root = _m4_tree(tmp_path)
    _write(root, "results/hosted_run.json", {"stopped_at_cap": False})
    recs = _records("gpt-6-luna", build_grid("gpt-6-luna", CFG.experiment, hosted=True)[:10])
    recs.append(dict(recs[0], case_id="gpt-6-luna|none|fake_system|user_task_0|r0", condition="fake_system"))
    recs.append(dict(recs[0]))
    _store_file(root, "gpt-6-luna", recs)
    (root / "results" / "hosted_ledger.jsonl").write_text(json.dumps({"cost_usd": 4.0}) + "\n")
    metrics = json.loads((root / "results" / "metrics.json").read_text())
    metrics["cells"][0]["tsr"] = {"point": 0.5, "lower": 0.6, "upper": 0.7, "method": "bootstrap"}
    metrics.pop("stamp")
    _write(root, "results/metrics.json", metrics)
    (root / "results" / "plots" / "defense_cost.png").unlink()
    text = "\n".join(G.check_m4(root))
    for needle in ("outside the hosted grid", "repeats a case id", "no Spend_Cap stop",
                   "hosted_run.json is missing or has no stamp", "exceeds the Spend_Cap", "misses", "does not bracket",
                   "results/metrics.json has no stamp", "different content", "defense_cost.png"):
        assert needle in text, needle
    (root / "results" / "metrics.json").unlink()
    (root / "results" / "raw" / "gpt-6-luna.jsonl").unlink()
    text = "\n".join(G.check_m4(root))
    assert "gpt-6-luna.jsonl is missing" in text and "metrics.json is missing" in text


def _m5_tree(tmp_path):
    root = _tree(tmp_path)
    readme = "\n".join(G.README_SECTIONS) + "\n- README: command `x`, seed 1, commit `" + "a" * 40 + "`.\n"
    report = "\n".join(G.REPORT_SECTIONS) + "\ntemperature 0.7 and top-p 0.9; 3 Repetitions\n"
    files = {"results/summary.md": "# T\n", "docs/REPORT.md": report, "README.md": readme}
    for rel, text in files.items():
        for base in (root, root / "runs" / "m5_before"):
            (base / rel).parent.mkdir(parents=True, exist_ok=True)
            (base / rel).write_text(text)
    (root / "runs" / "m5_before" / "README.md").write_text(readme.replace("a" * 40, "b" * 40))  # stamps may differ
    (root / "runs" / "head_commit.txt").write_text("c" * 40 + "\n")
    _write(root, "runs/ci_latest.json", {"headSha": "c" * 40, "conclusion": "success"})
    _write(root, "docs/evidence/ci_red_check.json", {"red": {"conclusion": "failure"}})
    (root / "docs" / "ENVIRONMENT.md").write_text(
        "Direct dependencies (exact pins): agentdojo 0.1.35, huggingface_hub 2.2.0.\n`unsloth/Q-GGUF` at revision `x`\n"
        "image ghcr.io/ggml-org/llama.cpp:tag\n")
    (root / "THIRD_PARTY.md").write_text("agentdojo huggingface-hub unsloth/Q-GGUF ghcr.io/ggml-org/llama.cpp\n")
    return root


def test_m5_passes_and_reports_problems(tmp_path):
    root = _m5_tree(tmp_path)
    assert G.check_m5(root) == []
    (root / "README.md").write_text("changed\n")
    (root / "docs" / "REPORT.md").write_text("## Question\n")
    (root / "results" / "summary.md").unlink()
    _write(root, "runs/ci_latest.json", {"headSha": "d" * 40, "conclusion": "success"})
    _write(root, "docs/evidence/ci_red_check.json", {"red": {"conclusion": "success"}})
    (root / "THIRD_PARTY.md").write_text("agentdojo\n")
    (root / ".kiro/specs/agent-injection-bench/design.md").write_text("nothing\n")
    text = "\n".join(G.check_m5(root))
    for needle in ("changed README.md", "summary.md is missing", "lacks ## Setup", "temperature, top-p", "README.md lacks",
                   "not a success for HEAD", "red run", "does not list huggingface_hub", "responsible-disclosure rule"):
        assert needle in text, needle
    assert G.third_party_names("Direct dependencies: a 1, b_c 2\n`x/y` at revision") == ["a", "b_c", "x/y"]
