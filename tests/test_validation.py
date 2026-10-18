"""Unit tests for aib.validation on a small subset."""

from __future__ import annotations

import json

from aib import validation as V


def test_subset_counts_and_shape():
    res = V.run_stub_validation({"commit": "a" * 40}, task_ids=["user_task_0", "user_task_13"], styles=["direct", "hidden_field"])
    assert res["n_cases"] == 2 * 2 * 2
    assert res["goals"] == ["injection_task_0", "injection_task_1"]
    fu = [c for c in res["cases"] if c["policy"] == "follow_user"]
    fi = [c for c in res["cases"] if c["policy"] == "follow_injection"]
    assert all(c["exposed"] and c["task_success"] and c["injection_success"] is False for c in fu)
    assert all(c["injection_success"] is True for c in fi)
    assert res["summary"]["follow_user"]["direct"]["status:classified"] == 2


def test_main_writes_stamped_file(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(V, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(V, "require_clean", lambda: "0" * 40)
    monkeypatch.setattr(V, "make_stamp", lambda c, s: type("S", (), {"to_dict": lambda self: {"command": c, "seed": s, "commit": "b" * 40}})())
    monkeypatch.setattr(V, "run_stub_validation", lambda stamp: {"cases": [], "summary": {"follow_user": {"direct": {"x": 1}}},
                                                                 "n_cases": 0, "styles": [], "goals": []})
    assert V.main([]) == 0
    assert json.loads((tmp_path / "results" / "m2_stub_validation.json").read_text())["stamp"]["commit"] == "b" * 40
    assert "0 cases" in capsys.readouterr().out
