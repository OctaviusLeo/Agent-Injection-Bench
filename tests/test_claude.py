"""Unit tests for aib.claude: the Claude ledger and client factories and the live check, with a fake client."""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest
import yaml

from aib import claude as C
from aib.config import load_all
from aib.ledger import request_cost
from tests.fake_anthropic import FakeClient, message, text

THINKING = {"type": "thinking", "thinking": "", "signature": "sig-1"}
TOOL_USE = {"type": "tool_use", "id": "toolu_1", "name": "get_current_day", "input": {}}


def _repo(tmp_path, edit=None):
    root = tmp_path / "repo"
    shutil.copytree(C.REPO_ROOT / "config", root / "config")
    if edit:
        path = root / "config" / "claude.yaml"
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        edit(data)
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    for cmd in (["init", "-q", "-b", "main"], ["config", "user.email", "t@e.com"], ["config", "user.name", "T"],
                ["add", "-A"], ["commit", "-q", "-m", "init"]):
        subprocess.run(["git", *cmd], cwd=root, check=True, capture_output=True)
    return root


def test_check_sends_three_turns_and_records_everything(tmp_path):
    root = _repo(tmp_path)
    client = FakeClient([text("It checks one unit of code.", input_tokens=40, output_tokens=9),
                         message([THINKING, TOOL_USE], stop="tool_use", input_tokens=900, cache_write=800),
                         text("Today is 2024-05-15.", input_tokens=60, cache_read=900)])
    code, res = C.run_check(root, client=client)
    assert code == 0 and res["passed"] and res["within_cap"]
    assert res["tool_request"]["tool_calls"] == [{"function": "get_current_day", "args": {}}]
    assert res["tool_request"]["thinking_blocks"] == 1 and res["tool_free"]["thinking_blocks"] == 0
    assert res["tool_result_turn"]["content"] == "Today is 2024-05-15." and res["tool_result_turn"]["cached_tokens"] == 900
    third = client.requests[2]["messages"]
    assert third[1] == {"role": "assistant", "content": [THINKING, TOOL_USE]}  # the thinking block goes back unchanged
    assert third[2]["content"] == [{"type": "tool_result", "tool_use_id": "toolu_1", "content": C.CHECK_DAY}]
    saved = json.loads((root / "results" / "claude_check.json").read_text())
    assert len(saved["stamp"]["commit"]) == 40 and saved["run_id"].startswith("claude_check-")
    ledger = [json.loads(line) for line in (root / "results" / "claude_ledger.jsonl").read_text().splitlines()]
    assert [e["event"] for e in ledger] == ["request"] * 3 and {e["run_id"] for e in ledger} == {saved["run_id"]}
    assert res["run_spend_usd"] == pytest.approx(sum(e["cost_usd"] for e in ledger))


@pytest.mark.parametrize("replies,condition", [
    ([text(""), text("x")], "non-empty tool-free response"),
    ([text("ok"), text("no tool for me")], "tool call to get_current_day"),
    ([text("ok"), message([TOOL_USE], stop="tool_use"), message([], stop="refusal", category="cyber")],
     "non-empty answer after the tool result"),
])
def test_check_names_the_failed_condition(tmp_path, replies, condition):
    code, res = C.run_check(_repo(tmp_path), client=FakeClient(replies))
    assert code == 1 and res["passed"] is False and res["failed_condition"] == condition


def test_missing_key_and_run_cap_are_named_failures(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_FOREVER_API_KEY", raising=False)
    code, res = C.run_check(_repo(tmp_path / "a"))
    assert code == 1 and res["failed_condition"] == "Anthropic API key accepted" and "ANTHROPIC_FOREVER_API_KEY" in res["cause"]

    client = FakeClient([text("never sent")])
    code, res = C.run_check(_repo(tmp_path / "b", edit=lambda d: d.__setitem__("run_cap_usd", 0)), client=client)
    assert code == 1 and res["failed_condition"] == "Claude request" and res["cause"].startswith("SpendCapExceeded")
    assert client.requests == [] and res["within_cap"]


def test_ledger_prices_carry_the_long_context_tier_and_runs_get_their_own_id(tmp_path):
    cfg = load_all()
    haiku = C.ledger_prices(cfg.claude.prices_per_million_usd["claude-haiku-5-5"])
    assert haiku.long_context_tokens == 100_000 and haiku.long_context.output == 2.50
    assert request_cost(haiku, 100_001, 0, 0) > request_cost(haiku, 100_000, 0, 0) * 4
    assert C.ledger_prices(cfg.claude.prices_per_million_usd["claude-sonnet-5-5"]).long_context is None
    a = C.make_claude_ledger(cfg.claude, "claude-haiku-5-5", tmp_path)
    b = C.make_claude_ledger(cfg.claude, "claude-haiku-5-5", tmp_path)
    assert a.run_id != b.run_id and a.run_id.startswith("claude-haiku-5-5-") and a.cap_usd == 5.0
    llm = C.make_claude_llm(cfg, "claude-haiku-5-5", a, client=FakeClient([]))
    assert (llm.model, llm.max_tokens, llm.effort, llm.api_key_env) == ("claude-haiku-5-5", 16000, "medium",
                                                                       "ANTHROPIC_FOREVER_API_KEY")


def test_main_prints_a_summary(monkeypatch, capsys):
    monkeypatch.setattr(C, "require_clean", lambda: None)
    rec = {"response_model": "claude-haiku-5-5", "finish_reason": "end_turn", "prompt_tokens": 1, "cached_tokens": 0,
           "completion_tokens": 1, "cost_usd": 0.0, "content": "ok", "tool_calls": [], "thinking_blocks": 0}
    monkeypatch.setattr(C, "run_check", lambda: (0, {"model": "claude-haiku-5-5", "passed": True, "tool_free": rec}))
    assert C.main() == 0
    out = capsys.readouterr().out
    assert '"passed": true' in out and "tool_free: model=claude-haiku-5-5 stop=end_turn thinking=0" in out
