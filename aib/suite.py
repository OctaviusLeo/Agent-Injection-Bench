"""The Sandbox: AgentDojo's workspace suite at v1.2.2 with the v1.1.2 task fixes (task 2.1; Req 2, 0.14).

Every Simulated_Tool is a Python function over an in-memory `WorkspaceEnvironment` loaded from
AgentDojo's local YAML fixture; nothing opens a socket. User_Tasks are the suite's 40 workspace
user tasks; each Task_Success_Check is the task's own deterministic AgentDojo predicate, and a
failed check reports what was not satisfied.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from agentdojo.agent_pipeline.ground_truth_pipeline import GroundTruthPipeline
from agentdojo.base_tasks import BaseInjectionTask, BaseUserTask
from agentdojo.functions_runtime import FunctionCall, FunctionsRuntime, TaskEnvironment
from agentdojo.task_suite.task_suite import TaskSuite, functions_stack_trace_from_messages
from agentdojo.types import get_text_content_as_str
from deepdiff import DeepDiff

BENCHMARK_VERSION = (1, 2, 2)
MAX_DIFF_PATHS = 20
_NUMBER = re.compile(r"_(\d+)$")


@lru_cache(maxsize=1)
def load_workspace_suite() -> TaskSuite:
    """Build the suite in the only order that works in agentdojo 0.1.35 (see design.md "Benchmark substrate")."""
    from agentdojo.task_suite.load_suites import get_suite  # noqa: F401  registers the standard versions

    import agentdojo.default_suites.v1_1_2  # noqa: F401  registers the v1.1.2 fixes (load_suites never imports them)
    from agentdojo.default_suites.v1.workspace import workspace_task_suite

    return workspace_task_suite.get_new_version(BENCHMARK_VERSION)


def task_number(task_id: str) -> int:
    m = _NUMBER.search(task_id)
    if not m:
        raise ValueError(f"not a numbered task id: {task_id!r}")
    return int(m.group(1))


@dataclass(frozen=True)
class UserTaskRef:
    id: str
    number: int
    task: BaseUserTask
    module: str  # the AgentDojo version module the task class comes from


def load_user_tasks(suite: TaskSuite | None = None) -> list[UserTaskRef]:
    """The suite's user tasks ordered by number (40 for the workspace suite, Req 2.3)."""
    suite = suite or load_workspace_suite()
    refs = [UserTaskRef(tid, task_number(tid), t, type(t).__module__) for tid, t in suite.user_tasks.items()]
    return sorted(refs, key=lambda r: r.number)


def get_user_task(task_id: str, suite: TaskSuite | None = None) -> BaseUserTask:
    return (suite or load_workspace_suite()).get_user_task_by_id(task_id)


def get_injection_task(task_id: str, suite: TaskSuite | None = None) -> BaseInjectionTask:
    return (suite or load_workspace_suite()).get_injection_task_by_id(task_id)


def init_case_environment(user_task: BaseUserTask, injections: dict[str, str], seed: int = 0,
                          suite: TaskSuite | None = None) -> TaskEnvironment:
    """Deterministic case state (Req 2.5). The fixture has no randomness: `seed` is recorded, not used."""
    suite = suite or load_workspace_suite()
    del seed
    return user_task.init_environment(suite.load_and_inject_default_environment(injections))


@dataclass(frozen=True)
class GroundTruthRun:
    output: str
    pre_env: TaskEnvironment
    post_env: TaskEnvironment
    trace: list[FunctionCall]
    tool_outputs: list[str]


def run_ground_truth(task: BaseUserTask | BaseInjectionTask, injections: dict[str, str] | None = None,
                     env: TaskEnvironment | None = None, suite: TaskSuite | None = None) -> GroundTruthRun:
    """Execute a task's ground-truth tool calls in a fresh (or given) environment and collect the trace."""
    suite = suite or load_workspace_suite()
    if env is None:
        env = (init_case_environment(task, injections or {}, suite=suite) if isinstance(task, BaseUserTask)
               else suite.load_and_inject_default_environment(injections or {}))
    pre = env.model_copy(deep=True)
    prompt = task.PROMPT if isinstance(task, BaseUserTask) else task.GOAL
    _, _, env, messages, _ = GroundTruthPipeline(task).query(prompt, FunctionsRuntime(suite.tools), env)
    outputs = [get_text_content_as_str(m["content"]) for m in messages if m["role"] == "tool"]
    return GroundTruthRun(task.GROUND_TRUTH_OUTPUT, pre, env, functions_stack_trace_from_messages(messages), outputs)


def canaries(suite: TaskSuite | None = None) -> dict[str, str]:
    suite = suite or load_workspace_suite()
    return {vector: f"CANARY_{vector}_END" for vector in suite.get_injection_vector_defaults()}


def _candidates(user_task: BaseUserTask, suite: TaskSuite) -> tuple[str, ...]:
    marks = canaries(suite)
    run = run_ground_truth(user_task, marks, suite=suite)
    text = "\n".join(run.tool_outputs)
    return tuple(vector for vector, mark in marks.items() if mark in text)


@lru_cache(maxsize=None)
def _candidates_default(task_id: str) -> tuple[str, ...]:
    suite = load_workspace_suite()
    return _candidates(suite.get_user_task_by_id(task_id), suite)


def injection_candidates(user_task: BaseUserTask, suite: TaskSuite | None = None) -> list[str]:
    """Placeholders whose canary appears in a ground-truth Tool_Output (the vectors the task reads).
    Cached per task for the default suite, since the ground truth is deterministic."""
    if suite is None or suite is load_workspace_suite():
        return list(_candidates_default(user_task.ID))
    return list(_candidates(user_task, suite))


@dataclass(frozen=True)
class CheckOutcome:
    passed: bool
    unmet: str | None  # set exactly when passed is False (Req 2.6)

    def __post_init__(self) -> None:
        if self.passed == (self.unmet is not None):
            raise ValueError("unmet must be set exactly when passed is False")


def env_change_paths(pre_env: TaskEnvironment, post_env: TaskEnvironment, limit: int = MAX_DIFF_PATHS) -> list[str]:
    diff = DeepDiff(pre_env.model_dump(mode="json"), post_env.model_dump(mode="json"), ignore_order=False)
    return sorted(diff.affected_paths)[:limit]


def task_success_check(user_task: BaseUserTask, output: str, pre_env: TaskEnvironment, post_env: TaskEnvironment,
                       trace: list[FunctionCall]) -> CheckOutcome:
    """The task's AgentDojo predicate, no LLM (Req 2.4): `utility_from_traces` when it returns a value, else `utility`."""
    predicate = "utility_from_traces"
    try:
        result = user_task.utility_from_traces(output, pre_env, post_env, trace)
        if result is None:
            predicate = "utility"
            result = user_task.utility(output, pre_env, post_env)
        error = None
    except Exception as exc:  # a predicate that cannot evaluate the reached state counts as not satisfied
        result, error = False, f"{type(exc).__name__}: {exc}"
    if result:
        return CheckOutcome(True, None)
    changes = env_change_paths(pre_env, post_env)
    parts = [f"{user_task.ID}.{predicate} ({type(user_task).__module__}) " +
             (f"raised {error}" if error else "returned False")]
    parts.append("environment changes: " + (", ".join(changes) if changes else "none"))
    parts.append(f"output: {output[:200]!r}")
    return CheckOutcome(False, "; ".join(parts))


@dataclass
class CatalogReport:
    total: int
    tools: int
    failures: list[tuple[str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict:
        return {"total": self.total, "tools": self.tools, "ok": self.ok,
                "failures": [{"task": t, "reason": r} for t, r in self.failures], "failure_count": len(self.failures)}


def _has_predicate(task: BaseUserTask) -> bool:
    utility = getattr(type(task), "utility", None)
    return callable(utility) and not getattr(utility, "__isabstractmethod__", False)


def validate_catalog(suite: TaskSuite | None = None) -> CatalogReport:
    """Every user task needs a resolvable check, a ground truth that passes it, and at least one
    injection placeholder that its ground truth reads (Req 2.7)."""
    suite = suite or load_workspace_suite()
    refs = load_user_tasks(suite)
    report = CatalogReport(total=len(refs), tools=len(suite.tools))
    for ref in refs:
        if not _has_predicate(ref.task):
            report.failures.append((ref.id, "no resolvable Task_Success_Check (utility predicate)"))
            continue
        try:
            run = run_ground_truth(ref.task, suite=suite)
            outcome = task_success_check(ref.task, run.output, run.pre_env, run.post_env, run.trace)
            if not outcome.passed:
                report.failures.append((ref.id, f"ground truth fails its check: {outcome.unmet}"))
                continue
            if not injection_candidates(ref.task, suite):
                report.failures.append((ref.id, "ground truth reads no injection placeholder"))
        except Exception as exc:
            report.failures.append((ref.id, f"{type(exc).__name__}: {exc}"))
    return report
