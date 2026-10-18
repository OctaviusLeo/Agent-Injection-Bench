"""Unit tests for aib.pilot."""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from aib import pilot as P
from aib.repro import REPO_ROOT


def _rec(i, status="classified", duration=10.0, overflow=False):
    return {"case_id": f"qwen3-4b|none|direct|user_task_{i}|r0", "status": status, "duration_s": duration,
            "error_cause": "boom" if status == "errored" else None, "prompt_tokens": 100, "cached_tokens": 50,
            "completion_tokens": 10, "n_model_calls": 3, "exposed": True, "context_overflow": overflow,
            "hit_max_iters": False, "task_success": True, "injection_success": False}


def test_summarize_projects_hours_and_counts():
    records = [_rec(0, duration=12.0), _rec(1, duration=8.0, overflow=True), _rec(2, status="errored", duration=4.0)]
    s = P.summarize(records, {"qwen3-4b": 2880, "llama3.1-8b": 2880}, {"qwen3-4b": 2, "llama3.1-8b": 1},
                    {"qwen3-4b": 120.0, "llama3.1-8b": 80.0}, "qwen3-4b")
    assert s["n_cases"] == 3 and s["errored"] == 1 and s["context_overflows"] == 1
    assert s["mean_case_seconds"] == pytest.approx(8.0)
    assert s["projections"]["qwen3-4b"]["projected_hours"] == pytest.approx(8.0 * 2880 / 2 / 3600, abs=0.01)
    assert s["projections"]["llama3.1-8b"]["decode_rate_scale"] == pytest.approx(1.5)
    assert s["projections"]["llama3.1-8b"]["projected_hours"] == pytest.approx(8.0 * 1.5 * 2880 / 3600, abs=0.01)
    assert [c["case_id"] for c in s["cases"]] == sorted(r["case_id"] for r in records)
    assert s["statuses"] == {"classified": 2, "errored": 1} and s["cases"][2]["error_cause"] == "boom"


def test_main_writes_stamped_pilot_json(tmp_path, monkeypatch, capsys):
    import aib.repro

    root = tmp_path / "repo"
    shutil.copytree(REPO_ROOT / "config", root / "config")
    (root / "results" / "pilot").mkdir(parents=True)
    store = [json.dumps({"header": True, "stamp": {}, "config": {}})] + [json.dumps(_rec(i)) for i in range(4)]
    (root / "results" / "pilot" / "qwen3-4b.jsonl").write_text("\n".join(store) + "\n")
    from aib.config import load_models

    local = load_models().local_models()
    for k, rate in zip(local, (110.0, 90.0)):
        (root / "results" / f"m0_smoke_{k}.json").write_text(json.dumps({"predicted_per_second": rate}))
    for cmd in (["init", "-q", "-b", "main"], ["config", "user.email", "t@e.com"], ["config", "user.name", "T"],
                ["add", "-A"], ["commit", "-q", "-m", "init"]):
        subprocess.run(["git", *cmd], cwd=root, check=True, capture_output=True)
    monkeypatch.setattr(aib.repro, "REPO_ROOT", root)
    monkeypatch.setattr(aib.repro, "require_clean", lambda: aib.repro.commit_hash(root))
    monkeypatch.setattr(aib.repro, "make_stamp", lambda c, s: aib.repro.Stamp(c, s, aib.repro.commit_hash(root)))
    assert P.main([]) == 0
    saved = json.loads((root / "results" / "pilot.json").read_text())
    assert len(saved["stamp"]["commit"]) == 40 and saved["n_cases"] == 4
    assert saved["projections"][local[1]]["workers"] == 1
    assert '"pilot_model": "qwen3-4b"' in capsys.readouterr().out
