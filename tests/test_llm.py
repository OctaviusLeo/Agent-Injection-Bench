"""Unit tests for aib.llm and aib.ledger against a fake OpenAI-compatible server on 127.0.0.1."""

from __future__ import annotations

import json

import pytest
from agentdojo.functions_runtime import FunctionCall, FunctionsRuntime
from agentdojo.types import ChatToolResultMessage, ChatUserMessage, text_content_block_from_string

from aib import llm as L
from aib.ledger import Prices, SpendCapExceeded, SpendLedger, request_cost
from tests.fake_openai import FakeOpenAI, completion

SAMPLING = L.Sampling(temperature=0.7, top_p=0.9, max_tokens=64, extra_body={"top_k": 0, "min_p": 0.0})
PRICES = Prices(cache_write=0.125, cached_input=0.01, output=0.50)


def get_current_day() -> str:
    """Returns the current day in ISO format, e.g. '2022-01-01'."""
    return "2024-05-15"


def add(a: int, b: int) -> int:
    """Adds two numbers.

    :param a: the first number
    :param b: the second number
    """
    return a + b


def _runtime():
    rt = FunctionsRuntime([])
    rt.register_function(get_current_day)
    rt.register_function(add)
    return rt


def _llm(server, **kw):
    defaults = dict(name="test", base_url=server.base_url, model="m", api_key_env=None, sampling=SAMPLING,
                    timeout_s=5.0, backoff_s=0.0)
    defaults.update(kw)
    return L.ChatLLM(**defaults)


def _user(text):
    return ChatUserMessage(role="user", content=[text_content_block_from_string(text)])


def test_two_turn_tool_conversation_converts_both_ways():
    replies = [
        (200, completion(tool_calls=[("id-a", "get_current_day", "{}"), ("id-b", "add", '{"a": 1, "b": 2}')]), 0),
        (200, completion(content="Today is 2024-05-15 and 1+2=3."), 0),
    ]
    with FakeOpenAI(replies) as srv:
        llm = _llm(srv)
        rt = _runtime()
        system = {"role": "system", "content": [text_content_block_from_string("sys")]}
        msgs = [system, _user("day?")]
        _, _, _, msgs, extra = llm.query("day?", rt, messages=msgs, extra_args={"aib_seed": 7})
        assistant = msgs[-1]
        assert [c.function for c in assistant["tool_calls"]] == ["get_current_day", "add"]
        assert assistant["tool_calls"][1].args == {"a": 1, "b": 2}
        assert assistant["tool_calls"][0].id == "id-a"
        # tool results: one success, one error
        msgs = [
            *msgs,
            ChatToolResultMessage(role="tool", content=[text_content_block_from_string("2024-05-15")],
                                  tool_call_id="id-a", tool_call=assistant["tool_calls"][0], error=None),
            ChatToolResultMessage(role="tool", content=[text_content_block_from_string("")],
                                  tool_call_id="id-b", tool_call=assistant["tool_calls"][1], error="ValueError: bad"),
        ]
        _, _, _, msgs, extra = llm.query("day?", rt, messages=msgs, extra_args=extra)
        assert msgs[-1]["content"][0]["content"].startswith("Today is")
        first, second = srv.requests
    assert first["messages"][0] == {"role": "system", "content": "sys"}
    assert first["temperature"] == 0.7 and first["top_p"] == 0.9 and first["max_tokens"] == 64
    assert first["top_k"] == 0 and first["min_p"] == 0.0
    assert first["seed"] == L.request_seed(7, 0) and second["seed"] == L.request_seed(7, 1)
    assert {t["function"]["name"] for t in first["tools"]} == {"get_current_day", "add"}
    assert first["tools"][1]["function"]["parameters"]["properties"]["a"]["type"] == "integer"
    sent_assistant = second["messages"][2]
    assert sent_assistant["content"] is None
    assert [tc["id"] for tc in sent_assistant["tool_calls"]] == ["id-a", "id-b"]
    assert json.loads(sent_assistant["tool_calls"][1]["function"]["arguments"]) == {"a": 1, "b": 2}
    assert second["messages"][3] == {"role": "tool", "tool_call_id": "id-a", "content": "2024-05-15"}
    assert second["messages"][4] == {"role": "tool", "tool_call_id": "id-b", "content": "ValueError: bad"}
    calls = extra["aib_calls"]
    assert [c["turn"] for c in calls] == [0, 1]
    assert calls[0]["n_tool_calls"] == 2 and calls[0]["prompt_tokens"] == 10 and calls[0]["finish_reason"] == "tool_calls"


def test_missing_call_ids_are_generated_and_invalid_json_is_flagged():
    replies = [(200, completion(tool_calls=[(None, "add", "{not json"), (None, "add", "[1, 2]")]), 0)]
    with FakeOpenAI(replies) as srv:
        _, _, _, msgs, extra = _llm(srv).query("q", _runtime(), messages=[_user("q")], extra_args={})
    calls = msgs[-1]["tool_calls"]
    assert [c.args for c in calls] == [{}, {}]
    assert [c.id for c in calls] == ["call_0_0", "call_0_1"]
    assert extra["aib_calls"][0]["malformed_args"] == ["add", "add"]


def test_429_and_500_are_retried():
    replies = [(429, {"error": {"message": "slow down"}}, 0), (500, {"error": {"message": "boom"}}, 0),
               (200, completion(content="ok"), 0)]
    with FakeOpenAI(replies) as srv:
        _, _, _, msgs, extra = _llm(srv).query("q", _runtime(), messages=[_user("q")], extra_args={})
        assert len(srv.requests) == 3
    assert msgs[-1]["content"][0]["content"] == "ok"
    assert extra["aib_calls"][0]["attempts"] == 3


def test_server_error_after_retry_cap_raises_typed_error():
    with FakeOpenAI([(503, {"error": {"message": "down"}}, 0)]) as srv:
        with pytest.raises(L.LLMServerError) as info:
            _llm(srv, max_attempts=2).query("q", _runtime(), messages=[_user("q")], extra_args={})
        assert len(srv.requests) == 2
    assert info.value.extra_args["aib_calls"][0]["error"].startswith("server")
    assert info.value.messages[0]["role"] == "user"


def test_timeout_surfaces_as_typed_error():
    with FakeOpenAI([(200, completion(content="late"), 1.5)]) as srv:
        with pytest.raises(L.LLMTimeoutError):
            _llm(srv, timeout_s=0.3, max_attempts=1).query("q", _runtime(), messages=[_user("q")], extra_args={})


def test_connection_error_is_typed():
    llm = L.ChatLLM("t", "http://127.0.0.1:9/v1", "m", None, SAMPLING, 1.0, max_attempts=1, backoff_s=0)
    with pytest.raises(L.LLMConnectionError):
        llm.query("q", _runtime(), messages=[_user("q")], extra_args={})


def test_context_overflow_and_unsupported_parameter_and_other_400():
    overflow = {"error": {"code": 400, "message": "the request exceeds the available context size, try increasing it",
                          "type": "exceed_context_size_error"}}
    unsupported = {"error": {"message": "Unsupported parameter: 'top_p' is not supported with this model.",
                             "param": "top_p", "code": "unsupported_parameter"}}
    other = {"error": {"message": "messages must not be empty"}}
    with FakeOpenAI([(400, overflow, 0)]) as srv:
        with pytest.raises(L.ContextOverflowError):
            _llm(srv).query("q", _runtime(), messages=[_user("q")], extra_args={})
        assert len(srv.requests) == 1
    with FakeOpenAI([(400, unsupported, 0)]) as srv:
        with pytest.raises(L.UnsupportedParameterError) as info:
            _llm(srv).query("q", _runtime(), messages=[_user("q")], extra_args={})
    assert info.value.param == "top_p"
    with FakeOpenAI([(400, other, 0)]) as srv:
        with pytest.raises(L.LLMBadRequestError) as info:
            _llm(srv).query("q", _runtime(), messages=[_user("q")], extra_args={})
    assert type(info.value) is L.LLMBadRequestError


@pytest.mark.parametrize("value", [None, "", "   "])
def test_missing_secret_raises_before_any_request(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    else:
        monkeypatch.setenv("OPENAI_API_KEY", value)
    with FakeOpenAI([(200, completion(content="x"), 0)]) as srv:
        with pytest.raises(L.MissingSecretError) as info:
            _llm(srv, api_key_env="OPENAI_API_KEY", provider="openai")
        assert srv.requests == []
    assert "OPENAI_API_KEY" in str(info.value)


def test_hosted_parameters_respect_accepted_params(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with FakeOpenAI([(200, completion(content="x"), 0)]) as srv:
        llm = _llm(srv, api_key_env="OPENAI_API_KEY", provider="openai", reasoning_effort="none",
                   accepted_params=["temperature", "seed"])
        llm.query("q", FunctionsRuntime([]), messages=[_user("q")], extra_args={})
        body = srv.requests[0]
    assert body["reasoning_effort"] == "none" and body["max_completion_tokens"] == 64
    assert "temperature" in body and "seed" in body and "top_p" not in body
    assert "max_tokens" not in body and "top_k" not in body and "tools" not in body


def test_ledger_refuses_before_sending_and_records_cached_cost(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    ledger_path = tmp_path / "ledger.jsonl"
    ledger = SpendLedger(ledger_path, cap_usd=0.01, prices=PRICES)
    usage_reply = completion(content="ok", prompt=1000, cached=800, completion_tokens=100)
    with FakeOpenAI([(200, usage_reply, 0)]) as srv:
        llm = _llm(srv, api_key_env="OPENAI_API_KEY", provider="openai", ledger=ledger)
        _, _, _, _, extra = llm.query("q", FunctionsRuntime([]), messages=[_user("q")], extra_args={})
        expected = request_cost(PRICES, 1000, 800, 100)
        assert extra["aib_calls"][0]["cost_usd"] == pytest.approx(expected)
        assert expected == pytest.approx((200 * 0.125 + 800 * 0.01 + 100 * 0.50) / 1e6)
        assert ledger.total_usd == pytest.approx(expected)
        # a huge prompt's worst case exceeds the cap: refused before sending
        big = _user("x" * 300_000)
        with pytest.raises(SpendCapExceeded):
            llm.query("q", FunctionsRuntime([]), messages=[big], extra_args={})
        assert len(srv.requests) == 1
    lines = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    assert [line["event"] for line in lines] == ["request", "refused"]
    assert lines[0]["cached_tokens"] == 800
    # a new ledger on the same file resumes the total
    assert SpendLedger(ledger_path, cap_usd=0.01, prices=PRICES).total_usd == pytest.approx(expected)


def test_ledger_reservations_and_release(tmp_path):
    ledger = SpendLedger(tmp_path / "l.jsonl", cap_usd=0.001, prices=PRICES)
    worst = ledger.worst_case(3000, 1000)  # 1000 tokens at cache-write + 1000 output
    assert worst == pytest.approx((1000 * 0.125 + 1000 * 0.5) / 1e6)
    r1 = ledger.reserve(3000, 1000)
    with pytest.raises(SpendCapExceeded):
        ledger.reserve(3000, 1000)  # second in-flight worst case would exceed the cap
    ledger.release(r1, "HTTP 500")
    r2 = ledger.reserve(3000, 1000)
    ledger.record(r2, {"prompt_tokens": 10, "completion_tokens": 1}, "m")
    assert ledger.total_usd > 0
    with pytest.raises(ValueError):
        SpendLedger(tmp_path / "x.jsonl", cap_usd=-1, prices=PRICES)


def test_ledger_with_run_id_counts_only_its_own_lines(tmp_path):
    path = tmp_path / "l.jsonl"
    first = SpendLedger(path, cap_usd=1.0, prices=PRICES, run_id="run-a")
    first.record(first.reserve(300, 100), {"prompt_tokens": 1000, "completion_tokens": 1000}, "m")
    spent = first.total_usd
    assert spent > 0
    assert SpendLedger(path, cap_usd=1.0, prices=PRICES, run_id="run-a").total_usd == pytest.approx(spent)
    assert SpendLedger(path, cap_usd=1.0, prices=PRICES, run_id="run-b").total_usd == 0.0
    assert SpendLedger(path, cap_usd=1.0, prices=PRICES).total_usd == pytest.approx(spent)  # no run id: the whole file
    assert {json.loads(line)["run_id"] for line in path.read_text().splitlines()} == {"run-a"}


def test_long_context_tier_prices_prompts_over_the_threshold():
    long = Prices(cache_write=0.625, cached_input=0.05, output=2.50)
    tiered = Prices(cache_write=0.125, cached_input=0.01, output=0.50, long_context_tokens=100_000, long_context=long)
    assert request_cost(tiered, 100_000, 0, 10) == pytest.approx(request_cost(PRICES, 100_000, 0, 10))
    assert request_cost(tiered, 100_001, 1, 10) == pytest.approx((100_000 * 0.625 + 1 * 0.05 + 10 * 2.50) / 1e6)


def test_failed_hosted_request_releases_reservation(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    ledger = SpendLedger(tmp_path / "l.jsonl", cap_usd=1.0, prices=PRICES)
    with FakeOpenAI([(500, {"error": {"message": "down"}}, 0)]) as srv:
        llm = _llm(srv, api_key_env="OPENAI_API_KEY", provider="openai", ledger=ledger, max_attempts=1)
        with pytest.raises(L.LLMServerError):
            llm.query("q", FunctionsRuntime([]), messages=[_user("q")], extra_args={})
    assert ledger.total_usd == 0.0
    assert json.loads((tmp_path / "l.jsonl").read_text().splitlines()[0])["event"] == "failed"


def test_response_without_choices_is_a_typed_error():
    body = completion(content="x")
    body["choices"] = []
    with FakeOpenAI([(200, body, 0)]) as srv:
        with pytest.raises(L.LLMResponseError):
            _llm(srv).query("q", _runtime(), messages=[_user("q")], extra_args={})


def test_message_conversion_handles_text_with_tool_calls_and_missing_ids():
    call = FunctionCall(function="add", args={"a": 1, "b": 2}, id="x1")
    msgs = [
        {"role": "assistant", "content": [text_content_block_from_string("thinking")], "tool_calls": [call]},
        {"role": "assistant", "content": None, "tool_calls": None},
        ChatToolResultMessage(role="tool", content=[text_content_block_from_string("3")], tool_call_id=None,
                              tool_call=call, error=None),
    ]
    out = L.to_openai_messages(msgs)
    assert out[0]["content"] == "thinking" and out[0]["tool_calls"][0]["id"] == "x1"
    assert out[1] == {"role": "assistant", "content": ""}
    assert out[2]["tool_call_id"] == "x1"


def test_invalid_provider_is_rejected():
    with pytest.raises(ValueError):
        L.ChatLLM("t", "http://127.0.0.1:9/v1", "m", None, SAMPLING, 1.0, provider="other")


def test_unparseable_model_output_500_is_not_retried():
    body = {"error": {"code": 500, "message": "The model produced output that does not match the expected peg-native format",
                      "type": "server_error"}}
    with FakeOpenAI([(500, body, 0)]) as srv:
        with pytest.raises(L.LLMResponseError) as info:
            _llm(srv).query("q", _runtime(), messages=[_user("q")], extra_args={})
        assert len(srv.requests) == 1  # deterministic model output: retrying the same request cannot help
    assert info.value.kind == "bad_response" and "could not be parsed" in str(info.value)
