"""Unit tests for aib.hosted against a fake OpenAI-compatible server (no real network, fake key)."""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from aib import hosted as H
from aib.config import load_hosted
from aib.llm import UnsupportedParameterError
from tests.fake_openai import FakeOpenAI, completion


def _repo(tmp_path):
    root = tmp_path / "repo"
    shutil.copytree(H.REPO_ROOT / "config", root / "config")
    for cmd in (["init", "-q", "-b", "main"], ["config", "user.email", "t@e.com"], ["config", "user.name", "T"],
                ["add", "-A"], ["commit", "-q", "-m", "init"]):
        subprocess.run(["git", *cmd], cwd=root, check=True, capture_output=True)
    return root


UNSUPPORTED_TOP_P = (400, {"error": {"message": "Unsupported parameter: 'top_p' is not supported with this model.",
                                     "param": "top_p", "code": "unsupported_parameter"}}, 0)


def test_check_drops_rejected_param_and_records_everything(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    root = _repo(tmp_path)
    replies = [
        UNSUPPORTED_TOP_P,
        (200, completion(content="It checks one unit of code.", prompt=40, completion_tokens=9, model="gpt-6-luna-2026-09-01"), 0),
        (200, completion(tool_calls=[("c1", "get_current_day", "{}")], prompt=120, cached=64, model="gpt-6-luna-2026-09-01"), 0),
    ]
    with FakeOpenAI(replies) as srv:
        code, res = H.run_check(root, base_url=srv.base_url)
        bodies = srv.requests
    assert code == 0 and res["passed"]
    assert res["accepted_params"] == ["temperature", "seed"] and res["dropped_params"] == ["top_p"]
    assert "top_p" in bodies[0] and "top_p" not in bodies[1] and "top_p" not in bodies[2]
    assert all(b["reasoning_effort"] == "none" and b["max_completion_tokens"] == 1024 for b in bodies)
    assert res["tool_request"]["tool_calls"][0]["function"] == "get_current_day"
    assert res["tool_free"]["response_model"] == "gpt-6-luna-2026-09-01"
    assert res["ledger_total_usd"] > 0 and res["within_cap"]
    assert load_hosted(root / "config" / "hosted.yaml").accepted_params == ["temperature", "seed"]
    ledger = [json.loads(line) for line in (root / "results" / "hosted_ledger.jsonl").read_text().splitlines()]
    assert [e["event"] for e in ledger] == ["failed", "request", "request"]
    saved = json.loads((root / "results" / "m0_hosted_check.json").read_text())
    assert len(saved["stamp"]["commit"]) == 40


def test_missing_key_is_a_named_failure(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    root = _repo(tmp_path)
    with FakeOpenAI([(200, completion(content="x"), 0)]) as srv:
        code, res = H.run_check(root, base_url=srv.base_url)
        assert srv.requests == []
    assert code == 1 and "OPENAI_API_KEY" in res["cause"]


def test_no_tool_call_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    root = _repo(tmp_path)
    with FakeOpenAI([(200, completion(content="Some text"), 0)]) as srv:
        code, res = H.run_check(root, base_url=srv.base_url)
    assert code == 1 and res["failed_condition"] == "tool call to get_current_day"


def test_http_error_is_recorded(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    root = _repo(tmp_path)
    with FakeOpenAI([(401, {"error": {"message": "bad key"}}, 0)]) as srv:
        code, res = H.run_check(root, base_url=srv.base_url)
    assert code == 1 and "bad key" in res["cause"]


def test_negotiation_raises_when_param_is_unknown():
    def send(params):
        raise UnsupportedParameterError("HTTP 400: unsupported value for stop", None)

    with pytest.raises(UnsupportedParameterError):
        H.send_with_negotiation(send, ("temperature",))
    exc = UnsupportedParameterError("Unsupported parameter: 'seed'", None)
    assert H.unsupported_param(exc, ("temperature", "seed")) == "seed"


def test_write_accepted_params_requires_one_line(tmp_path):
    p = tmp_path / "h.yaml"
    p.write_text("a: 1\n")
    with pytest.raises(ValueError):
        H.write_accepted_params(p, ("temperature",))


def test_main_prints_summary(monkeypatch, capsys):
    monkeypatch.setattr(H, "require_clean", lambda: "0" * 40)
    reply = {"response_model": "m", "prompt_tokens": 1, "cached_tokens": 0, "completion_tokens": 1, "cost_usd": 0.0,
             "content": "x", "tool_calls": []}
    monkeypatch.setattr(H, "run_check", lambda: (0, {"model": "gpt-6-luna", "passed": True, "tool_free": reply}))
    assert H.main() == 0
    assert "tool_free: model=m" in capsys.readouterr().out
