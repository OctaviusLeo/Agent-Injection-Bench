"""Unit tests for aib.catalog."""

from __future__ import annotations

import json

from aib import catalog as C


def test_build_catalog_content_and_determinism():
    first = C.build_catalog()
    assert first["user_task_count"] == 40 and first["tool_count"] == 24
    assert first["validate_catalog"]["ok"] is True
    assert all(t["ground_truth_passes"] and t["injection_placeholders"] for t in first["user_tasks"])
    assert first["version_mapping"]["injection_tasks"]["injection_task_0"].endswith("v1_1_2.workspace.injection_tasks")
    assert first["user_tasks"][0]["ground_truth_calls"][0]["function"] == "search_calendar_events"
    json.dumps(first)  # serializable
    assert C.build_catalog() == first


def test_main_writes_stamped_file(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(C, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(C, "require_clean", lambda: "0" * 40)
    monkeypatch.setattr(C, "make_stamp", lambda command, seed: type("S", (), {"to_dict": lambda self: {"command": command, "seed": seed, "commit": "a" * 40}})())
    monkeypatch.setattr(C, "build_catalog", lambda: {"user_task_count": 1, "tool_count": 2,
                                                      "validate_catalog": {"ok": False, "failure_count": 1}})
    assert C.main(["--out", "runs/cat.json"]) == 1
    saved = json.loads((tmp_path / "runs" / "cat.json").read_text())
    assert saved["stamp"]["command"] == "python scripts/catalog.py --out runs/cat.json"
    assert "1 catalog failures" in capsys.readouterr().out
