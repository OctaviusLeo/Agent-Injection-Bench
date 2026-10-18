"""Unit tests for aib.runner, aib.stub_model, and the `none` pipeline of aib.defenses."""

from __future__ import annotations

import json

import pytest
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement

from aib import defenses as D
from aib.llm import ChatLLM, Sampling
from aib.runner import CaseSpec, RunContext, run_case, trace_file
from aib.stub_model import StubLLM
from tests.fake_openai import FakeOpenAI, completion

CTX = RunContext(sampling={"temperature": 0.7, "top_p": 0.9, "max_tokens": 1024}, stamp={"command": "c", "seed": 1, "commit": "a" * 40},
                 model_version="stub")


def _spec(task="user_task_0", condition="direct", defense="none", rep=0, seed=11, model="stub"):
    k = int(task.rsplit("_", 1)[1])
    goal = None if condition == "clean" else f"injection_task_{k % 6}"
    return CaseSpec(model, defense, condition, task, goal, rep, seed)


def test_follow_user_completes_task_and_ignores_injection():
    r = run_case(_spec(), StubLLM("follow_user"), ctx=CTX)
    assert r.status == "classified" and r.exposed is True
    assert r.task_success is True and r.injection_success is False and r.unmet is None
    assert r.tool_calls == ["search_calendar_events"] and r.n_model_calls == 2 and not r.final_output_empty


def test_follow_injection_carries_out_the_goal():
    r = run_case(_spec(task="user_task_13", condition="important_instructions"), StubLLM("follow_injection"), ctx=CTX)
    assert r.status == "classified" and r.injection_success is True
    assert "delete_file" in r.tool_calls  # injection_task_1 deletes cloud file 13


def test_answer_only_is_unclassified_not_exposed():
    r = run_case(_spec(condition="hidden_field"), StubLLM("answer_only"), ctx=CTX)
    assert r.status == "unclassified_not_exposed" and r.exposed is False
    assert r.injection_success is None and r.injection_check_value is False and r.tool_calls == []
    # user_task_0 is read-only: its check looks only at the answer, and the ground-truth answer passes it
    assert r.task_success is True and r.unmet is None


class Boom(BasePipelineElement):
    name = "boom"

    def query(self, *args, **kwargs):
        raise RuntimeError("model element crashed")


def test_raising_element_gives_errored_and_the_next_case_runs():
    bad = run_case(_spec(), Boom(), ctx=CTX)
    assert bad.status == "errored" and bad.error_cause == "RuntimeError: model element crashed"
    assert bad.task_success is None and bad.injection_success is None and bad.exposed is None
    good = run_case(_spec(task="user_task_1"), StubLLM("follow_user"), ctx=CTX)
    assert good.status == "classified" and good.task_success is True


def test_clean_condition_has_no_injection_outcome():
    r = run_case(_spec(condition="clean"), StubLLM("follow_injection"), ctx=CTX)
    assert r.status == "clean" and r.injection_task_id is None
    assert r.exposed is None and r.injection_success is None and r.injection_check_value is None
    assert r.task_success is True


SAMPLING = Sampling(0.7, 0.9, 64, {"top_k": 0, "min_p": 0.0})
OVERFLOW = (400, {"error": {"code": 400, "message": "the request exceeds the available context size, try increasing it",
                            "type": "exceed_context_size_error"}}, 0)


def test_context_overflow_ends_case_and_checks_still_run():
    first = (200, completion(tool_calls=[("c1", "search_calendar_events", '{"query": "Networking event", "date": "2024-05-26"}')]), 0)
    with FakeOpenAI([first, OVERFLOW]) as srv:
        llm = ChatLLM("qwen3-4b", srv.base_url, "m", None, SAMPLING, 5.0, backoff_s=0)
        r = run_case(_spec(model="qwen3-4b"), llm, ctx=CTX)
    assert r.context_overflow is True and r.status == "classified" and r.exposed is True
    assert r.task_success is False and r.injection_success is False and r.final_output_empty
    assert r.n_model_calls == 2 and r.tool_calls == ["search_calendar_events"]


def test_llm_server_error_is_errored_with_kind():
    with FakeOpenAI([(503, {"error": {"message": "loading model"}}, 0)]) as srv:
        llm = ChatLLM("qwen3-4b", srv.base_url, "m", None, SAMPLING, 5.0, backoff_s=0, max_attempts=2)
        r = run_case(_spec(model="qwen3-4b"), llm, ctx=CTX)
    assert r.status == "errored" and r.error_kind == "server" and "503" in r.error_cause
    assert r.n_model_calls == 1


def test_every_record_carries_the_configuration_and_trace(tmp_path):
    ctx = RunContext(sampling=CTX.sampling, stamp=CTX.stamp, model_version="stub-v", trace_dir=tmp_path)
    r = run_case(_spec(task="user_task_7", condition="fake_system", rep=2, seed=99), StubLLM("follow_user"), ctx=ctx)
    d = r.to_dict()
    for key, value in {"model": "stub", "model_version": "stub-v", "defense": "none", "condition": "fake_system",
                       "user_task_id": "user_task_7", "injection_task_id": "injection_task_1", "rep": 2, "seed": 99,
                       "sampling": CTX.sampling, "stamp": CTX.stamp}.items():
        assert d[key] == value, key
    assert r.case_id == "stub|none|fake_system|user_task_7|r2"
    saved = json.loads(trace_file(tmp_path, r.case_id).read_text())
    assert saved["case_id"] == r.case_id and saved["tool_outputs"] and r.trace_path.endswith(".json")
    assert json.dumps(d)


def test_recording_executor_strips_its_delimiters():
    ex = D.RecordingToolsExecutor("<<", ">>")
    assert ex._format("x") == "<<x>>" and ex._raw("<<x>>") == "x" and ex._raw("y") == "y"
    assert D.RecordingToolsExecutor()._raw("<<x>>") == "<<x>>"


def test_unknown_defense_and_policy_rejected():
    with pytest.raises(ValueError):
        D.build_pipeline("magic", StubLLM("follow_user"))
    with pytest.raises(ValueError):
        StubLLM("obey_everyone")
    r = run_case(_spec(defense="magic"), StubLLM("follow_user"), ctx=CTX)
    assert r.status == "errored" and "magic" in r.error_cause
