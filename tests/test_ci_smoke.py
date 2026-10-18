"""Run the CI smoke subset in-process, and check that a broken expectation fails it."""

from __future__ import annotations

import json

from aib import ci_smoke as C


def test_ci_smoke_passes_in_process(tmp_path, capsys):
    assert C.main(tmp_path / "ci_smoke.json") == 0
    data = json.loads((tmp_path / "ci_smoke.json").read_text())
    assert data["passed"] and data["n_cases"] == 60 and data["first_mismatch"] is None
    assert data["summary"]["follow_injection/filter"]["injection_success"]["point"] == 0.0
    assert data["summary"]["follow_injection/none"]["injection_success"]["point"] == 1.0
    assert data["summary"]["follow_user/delimit"]["task_success"]["method"] == "wilson"
    assert "CI smoke passed: 60 cases" in capsys.readouterr().out


def test_mismatch_is_named_and_fails(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(C, "expected", lambda policy, defense: {"injection_success": True})
    assert C.main(tmp_path / "ci_smoke.json") == 1
    assert "CI smoke FAILED: follow_user/none/direct/user_task_0" in capsys.readouterr().out


def test_incomplete_run_fails_with_the_step(tmp_path, monkeypatch, capsys):
    def boom():
        raise RuntimeError("suite failed to load")

    monkeypatch.setattr(C, "run_smoke", boom)
    assert C.main(tmp_path / "x.json") == 2
    assert "did not complete: RuntimeError: suite failed to load" in capsys.readouterr().out
