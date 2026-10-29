"""Unit tests for aib.claude_llm against a fake client that returns the SDK's own response types (no key, no network)."""

from __future__ import annotations

import json

import anthropic
import pytest
from agentdojo.default_suites.v1.tools.calendar_client import get_current_day
from agentdojo.functions_runtime import FunctionCall, FunctionsRuntime, make_function
from agentdojo.types import (
    ChatAssistantMessage,
    ChatSystemMessage,
    ChatToolResultMessage,
    ChatUserMessage,
    get_text_content_as_str,
    text_content_block_from_string,
)

from aib import claude_llm as CL
from aib.ledger import Prices, SpendCapExceeded, SpendLedger, request_cost
from aib.llm import ContextOverflowError, LLMBadRequestError, LLMConnectionError, LLMServerError, LLMTimeoutError, MissingSecretError
from tests.fake_anthropic import REQUEST, FakeClient, message, status_error

PRICES = Prices(cache_write=0.125, cached_input=0.01, output=0.50)
TOOLS = FunctionsRuntime([make_function(get_current_day)])


def _llm(client, ledger, **kw):
    return CL.ClaudeLLM("claude-haiku-5-5", "claude-haiku-5-5", "ANTHROPIC_TEST_KEY", 16000, "medium", 60.0, ledger,
                        client=client, backoff_s=0.0, **kw)


def _ledger(tmp_path, cap=1.0):
    return SpendLedger(tmp_path / "claude_ledger.jsonl", cap, PRICES, run_id="run-1")


def _user(text):
    return ChatUserMessage(role="user", content=[text_content_block_from_string(text)])


def test_tool_turn_replays_claude_blocks_verbatim_and_hides_thinking(tmp_path):
    thinking = {"type": "thinking", "thinking": "", "signature": "sig-1"}
    tool_use = {"type": "tool_use", "id": "toolu_1", "name": "get_current_day", "input": {}}
    client = FakeClient([message([thinking, {"type": "text", "text": "Checking."}, tool_use], stop="tool_use"),
                         message([{"type": "text", "text": "Today is 2026-10-29."}])])
    ledger = _ledger(tmp_path)
    llm = _llm(client, ledger)
    msgs = [ChatSystemMessage(role="system", content=[text_content_block_from_string("You are a helpful agent.")]),
            _user("What day is it?")]
    _, _, _, msgs, extra = llm.query("q", TOOLS, messages=msgs, extra_args={})
    first = msgs[-1]
    assert get_text_content_as_str(first["content"]) == "Checking."  # the thinking block never reaches AgentDojo
    assert first["tool_calls"] == [FunctionCall(function="get_current_day", args={}, id="toolu_1")]
    assert first[CL.CLAUDE_CONTENT] == [thinking, {"type": "text", "text": "Checking."}, tool_use]

    call = first["tool_calls"][0]
    msgs = [*msgs, ChatToolResultMessage(role="tool", content=[text_content_block_from_string("2026-10-29")],
                                         tool_call_id="toolu_1", tool_call=call, error=None)]
    _, _, _, msgs, extra = llm.query("q", TOOLS, messages=msgs, extra_args=extra)
    assert get_text_content_as_str(msgs[-1]["content"]) == "Today is 2026-10-29." and msgs[-1]["tool_calls"] is None

    second = client.requests[1]
    assert second["system"] == "You are a helpful agent."
    assert second["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "What day is it?"}]},
        {"role": "assistant", "content": [thinking, {"type": "text", "text": "Checking."}, tool_use]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "2026-10-29"}]},
    ]
    assert second["tools"][0]["name"] == "get_current_day" and "input_schema" in second["tools"][0]
    assert second["output_config"] == {"effort": "medium"} and second["cache_control"] == {"type": "ephemeral"}
    assert second["max_tokens"] == 16000
    assert not {"temperature", "top_p", "top_k", "seed"} & set(second)
    records = extra["aib_calls"]
    assert [r["turn"] for r in records] == [0, 1] and all(r["seed"] is None for r in records)
    assert [r["finish_reason"] for r in records] == ["tool_use", "end_turn"]
    assert [r["n_tool_calls"] for r in records] == [1, 0]
    assert ledger.total_usd == pytest.approx(sum(r["cost_usd"] for r in records))
    lines = [json.loads(line) for line in (tmp_path / "claude_ledger.jsonl").read_text().splitlines()]
    assert [e["event"] for e in lines] == ["request", "request"] and {e["run_id"] for e in lines} == {"run-1"}


def test_usage_maps_cache_writes_and_reads_onto_the_ledger(tmp_path):
    client = FakeClient([message([{"type": "text", "text": "ok"}], input_tokens=100, output_tokens=20, cache_write=50,
                                 cache_read=400)])
    _, _, _, _, extra = _llm(client, _ledger(tmp_path)).query("q", TOOLS, messages=[_user("hi")], extra_args={})
    rec = extra["aib_calls"][0]
    assert (rec["prompt_tokens"], rec["cached_tokens"], rec["cache_write_tokens"], rec["completion_tokens"]) == (550, 400, 50, 20)
    assert rec["cost_usd"] == pytest.approx(request_cost(PRICES, 550, 400, 20))
    assert rec["response_model"] == "claude-haiku-5-5"


def test_refusal_is_recorded_with_its_category(tmp_path):
    client = FakeClient([message([], stop="refusal", category="cyber")])
    _, _, _, msgs, extra = _llm(client, _ledger(tmp_path)).query("q", TOOLS, messages=[_user("hi")], extra_args={})
    rec = extra["aib_calls"][0]
    assert rec["finish_reason"] == "refusal" and rec["stop_category"] == "cyber"
    assert msgs[-1]["content"] is None and msgs[-1]["tool_calls"] is None
    # the empty turn is skipped when the conversation is converted again
    assert CL.to_claude_messages([_user("hi"), msgs[-1], _user("again")])[1] == [
        {"role": "user", "content": [{"type": "text", "text": "hi"}, {"type": "text", "text": "again"}]}]


def test_tool_calls_count_only_when_claude_stopped_to_use_them(tmp_path):
    cut = {"type": "tool_use", "id": "toolu_9", "name": "get_current_day", "input": {}}
    client = FakeClient([message([cut], stop="max_tokens")])
    _, _, _, msgs, extra = _llm(client, _ledger(tmp_path)).query("q", TOOLS, messages=[_user("hi")], extra_args={})
    assert msgs[-1]["tool_calls"] is None and extra["aib_calls"][0]["finish_reason"] == "max_tokens"


def test_non_object_tool_input_is_flagged(tmp_path):
    reply = message([{"type": "tool_use", "id": "toolu_2", "name": "get_current_day", "input": {}}], stop="tool_use")
    object.__setattr__(reply.content[0], "input", "not an object")  # the API types `input` loosely; force the odd case
    client = FakeClient([reply])
    _, _, _, msgs, extra = _llm(client, _ledger(tmp_path)).query("q", TOOLS, messages=[_user("hi")], extra_args={})
    assert msgs[-1]["tool_calls"][0].args == {} and extra["aib_calls"][0]["malformed_args"] == ["get_current_day"]


def test_429_and_5xx_are_retried_then_typed(tmp_path):
    ok = message([{"type": "text", "text": "ok"}])
    client = FakeClient([status_error(anthropic.RateLimitError, 429, "rate limited"),
                         status_error(anthropic.APIStatusError, 529, "overloaded"), ok])
    _, _, _, _, extra = _llm(client, _ledger(tmp_path)).query("q", TOOLS, messages=[_user("hi")], extra_args={})
    assert extra["aib_calls"][0]["attempts"] == 3

    ledger = _ledger(tmp_path / "b")
    client = FakeClient([status_error(anthropic.InternalServerError, 500, "down")])
    with pytest.raises(LLMServerError) as err:
        _llm(client, ledger, max_attempts=2).query("q", TOOLS, messages=[_user("hi")], extra_args={})
    assert len(client.requests) == 2 and ledger.total_usd == 0.0
    assert err.value.extra_args["aib_calls"][0]["error"].startswith("server: HTTP 500")
    assert json.loads((tmp_path / "b" / "claude_ledger.jsonl").read_text())["event"] == "failed"


def test_timeouts_and_connection_errors_are_typed(tmp_path):
    with pytest.raises(LLMTimeoutError):
        _llm(FakeClient([anthropic.APITimeoutError(request=REQUEST)]), _ledger(tmp_path), max_attempts=1).query(
            "q", TOOLS, messages=[_user("hi")], extra_args={})
    with pytest.raises(LLMConnectionError):
        _llm(FakeClient([anthropic.APIConnectionError(request=REQUEST)]), _ledger(tmp_path), max_attempts=1).query(
            "q", TOOLS, messages=[_user("hi")], extra_args={})


def test_context_overflow_and_other_bad_requests_are_not_retried(tmp_path):
    client = FakeClient([status_error(anthropic.BadRequestError, 400, "prompt is too long: 1200000 tokens > 1000000 maximum")])
    with pytest.raises(ContextOverflowError):
        _llm(client, _ledger(tmp_path)).query("q", TOOLS, messages=[_user("hi")], extra_args={})
    client = FakeClient([status_error(anthropic.BadRequestError, 400, "messages.0: text content blocks must be non-empty")])
    with pytest.raises(LLMBadRequestError):
        _llm(client, _ledger(tmp_path)).query("q", TOOLS, messages=[_user("hi")], extra_args={})
    assert len(client.requests) == 1


@pytest.mark.parametrize("error", [
    status_error(anthropic.AuthenticationError, 401, "invalid x-api-key"),
    status_error(anthropic.BadRequestError, 400, "Your credit balance is too low to access the Anthropic API."),
])
def test_rejected_key_or_spent_credit_stops_like_a_missing_key(tmp_path, error):
    ledger = _ledger(tmp_path)
    with pytest.raises(MissingSecretError, match="refused the key in ANTHROPIC_TEST_KEY") as err:
        _llm(FakeClient([error]), ledger).query("q", TOOLS, messages=[_user("hi")], extra_args={})
    assert isinstance(err.value, CL.AccountError) and ledger.total_usd == 0.0


@pytest.mark.parametrize("value", [None, "   "])
def test_missing_key_raises_before_any_request(tmp_path, monkeypatch, value):
    if value is None:
        monkeypatch.delenv("ANTHROPIC_TEST_KEY", raising=False)
    else:
        monkeypatch.setenv("ANTHROPIC_TEST_KEY", value)
    with pytest.raises(MissingSecretError, match="ANTHROPIC_TEST_KEY"):
        CL.ClaudeLLM("m", "m", "ANTHROPIC_TEST_KEY", 100, "low", 5.0, _ledger(tmp_path))
    monkeypatch.setenv("ANTHROPIC_TEST_KEY", "test-key")  # a set key builds the real client; nothing is sent
    assert isinstance(CL.ClaudeLLM("m", "m", "ANTHROPIC_TEST_KEY", 100, "low", 5.0, _ledger(tmp_path)).client, anthropic.Anthropic)


def test_run_cap_refuses_before_sending(tmp_path):
    client = FakeClient([message([{"type": "text", "text": "ok"}])])
    with pytest.raises(SpendCapExceeded):
        _llm(client, _ledger(tmp_path, cap=0.0)).query("q", TOOLS, messages=[_user("hi")], extra_args={})
    assert client.requests == []


def test_conversion_without_claude_blocks_and_error_results():
    call = FunctionCall(function="get_current_day", args={}, id="c1")
    msgs = [_user("hi"),
            ChatAssistantMessage(role="assistant", content=[text_content_block_from_string("Let me look.")], tool_calls=[call]),
            ChatToolResultMessage(role="tool", content=[], tool_call_id="c1", tool_call=call, error="ValueError: boom")]
    system, out = CL.to_claude_messages(msgs)
    assert system is None
    assert out[1] == {"role": "assistant", "content": [{"type": "text", "text": "Let me look."},
                                                       {"type": "tool_use", "id": "c1", "name": "get_current_day", "input": {}}]}
    assert out[2] == {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "ValueError: boom",
                                                   "is_error": True}]}
    with pytest.raises(ValueError, match="first message"):
        CL.to_claude_messages([_user("hi"), ChatSystemMessage(role="system", content=[text_content_block_from_string("x")])])


def test_structured_output_and_no_cache_for_one_off_requests(tmp_path):
    schema = {"type": "json_schema", "schema": {"type": "object", "properties": {}, "additionalProperties": False}}
    client = FakeClient([message([{"type": "text", "text": "{}"}])])
    _llm(client, _ledger(tmp_path), output_format=schema, cache=False).query(
        "q", FunctionsRuntime([]), messages=[_user("hi")], extra_args={})
    params = client.requests[0]
    assert params["output_config"] == {"effort": "medium", "format": schema}
    assert "cache_control" not in params and "tools" not in params and "system" not in params
