"""Stub_Model: a deterministic, network-free pipeline element driven by ground truth (task 3.4; Req 8.3, 8.6).

Policies:
- `follow_user` replays the user task's ground-truth tool calls (computed from the environment at the
  first call), one per turn, then answers with the task's GROUND_TRUTH_OUTPUT.
- `follow_injection` does the same, except that right after the first Tool_Output it receives that
  contains the planted text, it inserts the goal's ground-truth calls before the remaining ones.
- `answer_only` answers with GROUND_TRUTH_OUTPUT at once and calls no tool (used by tests).

The stub reads the case from `extra_args` (user task id, goal id, injections) and keeps its plan
there too, so one instance can serve any number of cases and never carries state between them.
It sees tool outputs after the defenses, so a filtered output hides the injection from it.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionCall, FunctionsRuntime
from agentdojo.types import ChatAssistantMessage, ChatMessage, get_text_content_as_str, text_content_block_from_string

from aib.attacks import contains_injection
from aib.goals import get_goal
from aib.suite import get_user_task

POLICIES = ("follow_user", "follow_injection", "answer_only")


def _plain(call: FunctionCall) -> dict:
    return {"function": call.function, "args": copy.deepcopy(dict(call.args))}


def _trailing_tool_texts(messages: Sequence[ChatMessage]) -> list[str]:
    texts = []
    for m in reversed(messages):
        if m["role"] != "tool":
            break
        texts.append(get_text_content_as_str(m["content"] or []))
    return texts


class StubLLM(BasePipelineElement):
    def __init__(self, policy: str, name: str = "stub") -> None:
        if policy not in POLICIES:
            raise ValueError(f"unknown stub policy {policy!r}; known: {POLICIES}")
        self.policy = policy
        self.name = f"{name}-{policy}"

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        extra = dict(extra_args)
        user_task = get_user_task(extra["aib_user_task_id"])
        state = copy.deepcopy(extra.get("aib_stub"))
        if state is None:
            plan = [] if self.policy == "answer_only" else [_plain(c) for c in user_task.ground_truth(env)]
            state = {"queue": plan, "inserted": False, "turn": 0}
        injections = extra.get("aib_injections") or {}
        goal_id = extra.get("aib_injection_task_id")
        if (self.policy == "follow_injection" and not state["inserted"] and injections and goal_id
                and any(contains_injection(t, injections) for t in _trailing_tool_texts(messages))):
            state["queue"] = [_plain(c) for c in get_goal(goal_id).ground_truth(env)] + state["queue"]
            state["inserted"] = True
        turn = state["turn"]
        if state["queue"]:
            nxt = state["queue"].pop(0)
            call = FunctionCall(function=nxt["function"], args=copy.deepcopy(nxt["args"]), id=f"stub_{turn}")
            reply = ChatAssistantMessage(role="assistant", content=[text_content_block_from_string("")], tool_calls=[call])
        else:
            reply = ChatAssistantMessage(role="assistant",
                                         content=[text_content_block_from_string(user_task.GROUND_TRUTH_OUTPUT)],
                                         tool_calls=None)
        state["turn"] = turn + 1
        record = {"turn": turn, "seed": int(extra.get("aib_seed", 0)), "latency_s": 0.0, "attempts": 1,
                  "prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0,
                  "finish_reason": "tool_calls" if reply["tool_calls"] else "stop", "response_model": self.name,
                  "n_tool_calls": len(reply["tool_calls"] or []), "malformed_args": [], "cost_usd": None, "error": None}
        extra["aib_stub"] = state
        extra["aib_calls"] = [*extra.get("aib_calls", []), record]
        return query, runtime, env, [*messages, reply], extra
