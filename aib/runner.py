"""Case runner: one (Model, Defense_Condition, condition, User_Task, Repetition) case end to end (task 3.4; Req 3.6, 5.3, 5.5, 5.6).

plant -> init env -> pipeline -> both checks on the same case -> status -> one Results_Store record.
A per-case failure becomes status `errored` with its cause and the sweep continues; a context
overflow ends the case early, like the iteration cap, and the checks run on the state reached.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import FunctionCall, FunctionsRuntime
from agentdojo.task_suite.task_suite import functions_stack_trace_from_messages
from agentdojo.types import get_text_content_as_str

from aib.attacks import contains_injection, plant
from aib.defenses import MAX_ITERS, build_pipeline
from aib.goals import classify, get_goal, injection_success_check
from aib.ledger import SpendCapExceeded
from aib.llm import ContextOverflowError, LLMError, MissingSecretError
from aib.suite import get_user_task, init_case_environment, load_workspace_suite, task_success_check

INFRA_ERROR_KINDS = ("connection", "timeout", "server")


@dataclass(frozen=True)
class CaseSpec:
    model: str
    defense: str
    condition: str
    user_task_id: str
    injection_task_id: str | None
    rep: int
    seed: int

    @property
    def case_id(self) -> str:
        return f"{self.model}|{self.defense}|{self.condition}|{self.user_task_id}|r{self.rep}"


@dataclass
class CaseResult:
    """One Results_Store record (Req 5.5)."""

    case_id: str
    model: str
    model_version: str
    defense: str
    condition: str
    user_task_id: str
    injection_task_id: str | None
    rep: int
    seed: int
    sampling: dict
    status: str
    exposed: bool | None
    injection_success: bool | None
    task_success: bool | None
    unmet: str | None
    error_cause: str | None
    n_model_calls: int
    tool_calls: list[str]
    hit_max_iters: bool
    context_overflow: bool
    final_output_empty: bool
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int
    duration_s: float
    filter_decisions: list[dict]
    trace_path: str
    stamp: dict
    error_kind: str | None = None
    injection_check_value: bool | None = None  # the predicate's value even when unclassified (not used in rates)
    malformed_args: int = 0
    response_models: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RunContext:
    sampling: dict
    stamp: dict
    defense_cfg: object = None
    model_version: str = ""
    trace_dir: Path | None = None
    max_iters: int = MAX_ITERS


def _final_output(messages) -> tuple[str, bool]:
    """(final text, hit_max_iters). The loop ended on an assistant message that still asked for tools
    when the iteration cap was reached."""
    if not messages or messages[-1]["role"] != "assistant":
        return "", False
    last = messages[-1]
    text = get_text_content_as_str(last["content"]) if last.get("content") else ""
    return text, bool(last.get("tool_calls"))


def _jsonable(obj):
    if isinstance(obj, FunctionCall):
        return {"function": obj.function, "args": _jsonable(dict(obj.args)), "id": obj.id}
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def trace_file(trace_dir: Path, case_id: str) -> Path:
    return trace_dir / (re.sub(r"[^A-Za-z0-9_.-]+", "__", case_id) + ".json")


def run_case(spec: CaseSpec, llm: BasePipelineElement, classifier=None, ctx: RunContext | None = None) -> CaseResult:
    ctx = ctx or RunContext(sampling={}, stamp={})
    start = time.monotonic()
    user_task = get_user_task(spec.user_task_id)
    injected = spec.condition != "clean"
    goal = get_goal(spec.injection_task_id) if injected else None
    injections = plant(spec.condition, user_task, goal) if injected else {}
    env = init_case_environment(user_task, injections, spec.seed)
    pre_env = env.model_copy(deep=True)
    extra_args = {"aib_seed": spec.seed, "aib_injections": injections, "aib_user_task_id": spec.user_task_id,
                  "aib_injection_task_id": spec.injection_task_id, "aib_calls": [], "aib_tool_outputs": [],
                  "aib_filter": []}
    messages: list = []
    error_cause = error_kind = None
    overflow = False
    try:
        pipeline = build_pipeline(spec.defense, llm, ctx.defense_cfg, classifier, ctx.max_iters)
        runtime = FunctionsRuntime(load_workspace_suite().tools)
        _, _, env, messages, extra_args = pipeline.query(user_task.PROMPT, runtime, env, [], extra_args)
    except ContextOverflowError as exc:  # the model's behavior, not an execution failure
        overflow = True
        messages, extra_args = exc.messages, {**extra_args, **exc.extra_args}
    except (SpendCapExceeded, MissingSecretError):
        raise
    except LLMError as exc:
        error_kind, error_cause = exc.kind, f"{exc.kind}: {exc}"[:1000]
        messages, extra_args = exc.messages, {**extra_args, **exc.extra_args}
    except Exception as exc:  # a per-case failure never stops the sweep (Req 5.6)
        error_kind, error_cause = "exception", f"{type(exc).__name__}: {exc}"[:1000]

    errored = error_cause is not None
    output, hit_max = ("", False) if (errored or overflow) else _final_output(messages)
    trace = functions_stack_trace_from_messages(messages)
    raw_outputs = [o["raw"] for o in extra_args.get("aib_tool_outputs", [])]
    exposed = any(contains_injection(o, injections) for o in raw_outputs) if injected and not errored else None
    status = classify(spec.condition, errored, bool(exposed))
    task_success = unmet = injection_success = check_value = None
    if not errored:
        outcome = task_success_check(user_task, output, pre_env, env, trace)
        task_success, unmet = outcome.passed, outcome.unmet
        if injected:
            check_value = injection_success_check(goal, output, pre_env, env, trace)
            injection_success = check_value if status == "classified" else None

    calls = extra_args.get("aib_calls", [])
    trace_path = ""
    if ctx.trace_dir is not None:
        path = trace_file(Path(ctx.trace_dir), spec.case_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"case_id": spec.case_id, "messages": _jsonable(list(messages)),
                                    "tool_outputs": extra_args.get("aib_tool_outputs", []), "calls": calls,
                                    "filter": extra_args.get("aib_filter", []), "error": error_cause}, indent=1),
                        encoding="utf-8")
        trace_path = str(path)
    return CaseResult(
        case_id=spec.case_id, model=spec.model, model_version=ctx.model_version, defense=spec.defense,
        condition=spec.condition, user_task_id=spec.user_task_id, injection_task_id=spec.injection_task_id,
        rep=spec.rep, seed=spec.seed, sampling=dict(ctx.sampling), status=status, exposed=exposed,
        injection_success=injection_success, task_success=task_success, unmet=unmet, error_cause=error_cause,
        n_model_calls=len(calls), tool_calls=[c.function for c in trace], hit_max_iters=hit_max,
        context_overflow=overflow, final_output_empty=not output.strip(),
        prompt_tokens=sum(int(c.get("prompt_tokens") or 0) for c in calls),
        cached_tokens=sum(int(c.get("cached_tokens") or 0) for c in calls),
        completion_tokens=sum(int(c.get("completion_tokens") or 0) for c in calls),
        duration_s=round(time.monotonic() - start, 3), filter_decisions=list(extra_args.get("aib_filter", [])),
        trace_path=trace_path, stamp=dict(ctx.stamp), error_kind=error_kind, injection_check_value=check_value,
        malformed_args=sum(len(c.get("malformed_args") or []) for c in calls),
        response_models=sorted({c["response_model"] for c in calls if c.get("response_model")}),
    )
