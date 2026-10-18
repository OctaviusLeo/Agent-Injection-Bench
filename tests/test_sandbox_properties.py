"""Property tests for the Sandbox and the success checks (design.md Properties 1-3)."""

from __future__ import annotations

from agentdojo.agent_pipeline.tool_execution import tool_result_to_str
from agentdojo.functions_runtime import FunctionCall, FunctionsRuntime
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from aib import suite as S
from aib.attacks import STYLES, plant
from tests.sandbox_helpers import frozen_tool_clock, socket_guard

SUITE = S.load_workspace_suite()
TOOLS = {t.name: t for t in SUITE.tools}
USER_TASKS = S.load_user_tasks()
GOALS = [f"injection_task_{i}" for i in range(6)]
CLEAN_ENV = SUITE.load_and_inject_default_environment({})

# Values that occur in the fixture, so drawn calls often take the tools' success paths.
FIXTURE_STRINGS = [
    "13", "0", "1", "2", "6", "24", "2024-05-15", "2024-05-26", "2024-05-15 10:00", "2024-05-26 11:00",
    "emma.johnson@bluesparrowtech.com", "lily.white@gmail.com", "mark.black-2134@gmail.com", "security@facebook.com",
    "Introductory meeting", "meeting", "Team Building", "feedback.xlsx", "recipe", "security code", "hiking",
    "Facebook", "Emma", "David", "team-building-activities.docx", "",
]


def _string():
    return st.one_of(st.sampled_from(FIXTURE_STRINGS), st.text(max_size=20))


def _from_schema(schema: dict, defs: dict) -> st.SearchStrategy:
    if "$ref" in schema:
        return _from_schema(defs[schema["$ref"].split("/")[-1]], defs)
    if "anyOf" in schema:
        return st.one_of([_from_schema(s, defs) for s in schema["anyOf"]])
    if "enum" in schema:
        return st.sampled_from(schema["enum"])
    kind = schema.get("type")
    if kind == "string":
        return _string()
    if kind == "integer":
        return st.integers(min_value=-2, max_value=40)
    if kind == "number":
        return st.floats(min_value=-2, max_value=40, allow_nan=False)
    if kind == "boolean":
        return st.booleans()
    if kind == "null":
        return st.none()
    if kind == "array":
        return st.lists(_from_schema(schema.get("items", {"type": "string"}), defs), max_size=3)
    if kind == "object":
        props = schema.get("properties", {})
        return st.fixed_dictionaries({k: _from_schema(v, defs) for k, v in props.items()})
    return _string()


@st.composite
def tool_call(draw):
    name = draw(st.sampled_from(sorted(TOOLS)))
    schema = TOOLS[name].parameters.model_json_schema()
    defs = schema.get("$defs", {})
    required = set(schema.get("required", []))
    args = {}
    for key, sub in schema.get("properties", {}).items():
        if key in required or draw(st.booleans()):
            args[key] = draw(_from_schema(sub, defs))
    return name, args


def _ground_truth_calls() -> list[FunctionCall]:
    calls = []
    for ref in USER_TASKS:
        calls.extend(ref.task.ground_truth(S.init_case_environment(ref.task, {})))
    for goal in GOALS:
        calls.extend(S.get_injection_task(goal).ground_truth(CLEAN_ENV))
    return calls


GT_CALLS = _ground_truth_calls()


def _run(env, name, args):
    result, error = FunctionsRuntime(SUITE.tools).run_function(env, name, args)
    return tool_result_to_str(result), error


# Feature: agent-injection-bench, Property 1: Sandbox never performs real network I/O
@settings(max_examples=100, derandomize=True, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(call=tool_call(), task_index=st.integers(min_value=0, max_value=len(USER_TASKS) - 1))
def test_property_1_sandbox_never_performs_network_io(call, task_index):
    name, args = call
    with frozen_tool_clock(), socket_guard() as trips:
        out1 = _run(CLEAN_ENV.model_copy(deep=True), name, dict(args))
        out2 = _run(CLEAN_ENV.model_copy(deep=True), name, dict(args))
        S.run_ground_truth(USER_TASKS[task_index].task)
    assert trips == []
    assert out1 == out2


# Injection texts are template prose plus a goal: printable characters and newlines, no YAML control characters.
_TEMPLATE_TEXT = st.text(alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,:!?-#()<>\n"),
                         max_size=60)


@st.composite
def injection_mapping(draw, task):
    vectors = S.injection_candidates(task)
    chosen = draw(st.lists(st.sampled_from(vectors), unique=True, max_size=len(vectors)))
    return {v: draw(_TEMPLATE_TEXT) for v in chosen}


# Feature: agent-injection-bench, Property 2: Sandbox initialization is deterministic under a seed
@settings(max_examples=100, derandomize=True, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(data=st.data(), seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_property_2_initialization_is_deterministic(data, seed):
    ref = data.draw(st.sampled_from(USER_TASKS))
    goal = data.draw(st.sampled_from(GOALS))
    style = data.draw(st.sampled_from([None, *sorted(STYLES)]))  # None: a drawn mapping of template-like text
    if style is None:
        injections = data.draw(injection_mapping(ref.task))
        injections = {k: f"{v} {S.get_injection_task(goal).GOAL}" for k, v in injections.items()}
    else:
        injections = plant(style, ref.task, S.get_injection_task(goal))
    first = S.init_case_environment(ref.task, injections, seed).model_dump(mode="json")
    second = S.init_case_environment(ref.task, injections, seed).model_dump(mode="json")
    assert first == second


def _security(goal_task, output, pre, post, trace):
    """AgentDojo's security predicate: security_from_traces when it returns a value, else security."""
    value = goal_task.security_from_traces(output, pre, post, trace)
    return goal_task.security(output, pre, post) if value is None else value


@st.composite
def call_sequence(draw):
    n = draw(st.integers(min_value=0, max_value=5))
    calls = []
    for _ in range(n):
        if draw(st.booleans()):
            gt = draw(st.sampled_from(GT_CALLS))
            calls.append((gt.function, dict(gt.args)))
        else:
            calls.append(draw(tool_call()))
    return calls


# Feature: agent-injection-bench, Property 3: Success checks are deterministic boolean predicates with a reason on failure
@settings(max_examples=100, derandomize=True, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(data=st.data(), calls=call_sequence(), output=st.one_of(st.text(max_size=80), st.sampled_from(FIXTURE_STRINGS)))
def test_property_3_checks_are_deterministic_with_reason(data, calls, output):
    ref = data.draw(st.sampled_from(USER_TASKS))
    goal_task = S.get_injection_task(data.draw(st.sampled_from(GOALS)))
    pre = S.init_case_environment(ref.task, {})
    post = pre.model_copy(deep=True)
    runtime = FunctionsRuntime(SUITE.tools)
    trace = []
    for name, args in calls:
        runtime.run_function(post, name, args)
        trace.append(FunctionCall(function=name, args=args))
    first = S.task_success_check(ref.task, output, pre, post, trace)
    second = S.task_success_check(ref.task, output, pre, post, trace)
    assert first == second
    assert isinstance(first.passed, bool)
    assert (first.unmet is not None and first.unmet != "") == (not first.passed)
    sec1 = _security(goal_task, output, pre, post, trace)
    sec2 = _security(goal_task, output, pre, post, trace)
    assert isinstance(sec1, bool) and sec1 == sec2
