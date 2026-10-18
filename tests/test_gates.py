"""Unit tests for aib.gates on small fixture trees."""

from __future__ import annotations

import json

from aib import gates as G

GOOD_STAMP = {"command": "python scripts/x.py", "seed": 0, "commit": "a" * 40}


def _catalog(n=40, passes=True, placeholders=("p",), tools=24, failures=()):
    return {
        "user_task_count": n, "tool_count": tools,
        "validate_catalog": {"failures": list(failures)},
        "user_tasks": [{"id": f"user_task_{i}", "ground_truth_passes": passes, "injection_placeholders": list(placeholders)}
                       for i in range(n)],
    }


def _write(root, rel, data):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data))


def test_m1_passes_on_consistent_catalog(tmp_path):
    cat = _catalog()
    _write(tmp_path, "results/m1_catalog.json", {"stamp": GOOD_STAMP, **cat})
    assert G.check_m1(tmp_path, regenerate=lambda: cat) == []
    assert (tmp_path / "runs" / "m1_catalog_regen.json").exists()


def test_m1_reports_every_problem(tmp_path):
    cat = _catalog(n=39, passes=False, placeholders=(), tools=3, failures=[{"task": "user_task_1"}])
    _write(tmp_path, "results/m1_catalog.json", {"stamp": {"commit": "0" * 39}, **cat})
    failures = G.check_m1(tmp_path, regenerate=lambda: _catalog())
    text = "\n".join(failures)
    for needle in ("expected 40", "does not pass", "reads no injection", "fewer than 4 tools", "validate_catalog failures",
                   "missing stamp", "different content"):
        assert needle in text


def test_m1_missing_file(tmp_path):
    assert "missing" in G.check_m1(tmp_path, regenerate=dict)[0]


def test_main_usage_and_dispatch(monkeypatch, capsys):
    assert G.main([]) == 2
    monkeypatch.setitem(G.GATES, "m1", lambda: ["broken thing"])
    assert G.main(["m1"]) == 1
    out = capsys.readouterr().out
    assert "GATE FAIL: broken thing" in out and "FAIL (1 problems)" in out
    monkeypatch.setitem(G.GATES, "m1", lambda: [])
    assert G.main(["m1"]) == 0


def test_stamp_helpers():
    assert G.stamp_ok({"stamp": GOOD_STAMP}) and not G.stamp_ok({}) and not G.stamp_ok(None)
    assert G.without_stamp({"stamp": 1, "a": 2}) == {"a": 2}
