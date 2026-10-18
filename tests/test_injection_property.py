"""Property test for injection placement (design.md Property 4)."""

from __future__ import annotations

from deepdiff import DeepDiff
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from aib import suite as S
from aib.attacks import STYLES, contains_injection, normalize, plant

USER_TASKS = S.load_user_tasks()
GOALS = [f"injection_task_{i}" for i in range(6)]


# Feature: agent-injection-bench, Property 4: Injection is planted only in Tool_Output, never in the User_Task
@settings(max_examples=100, derandomize=True, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(style=st.sampled_from(sorted(STYLES)), goal=st.sampled_from(GOALS),
       task_index=st.integers(min_value=0, max_value=len(USER_TASKS) - 1))
def test_property_4_injection_only_in_tool_output(style, goal, task_index):
    task = USER_TASKS[task_index].task
    prompt_before = task.PROMPT
    injections = plant(style, task, S.get_injection_task(goal))
    assert task.PROMPT == prompt_before
    assert injections and set(injections) <= set(S.load_workspace_suite().get_injection_vector_defaults())
    run = S.run_ground_truth(task, injections)
    assert any(contains_injection(out, injections) for out in run.tool_outputs)
    clean = S.init_case_environment(task, {}).model_dump(mode="json")
    injected = S.init_case_environment(task, injections).model_dump(mode="json")
    diff = DeepDiff(clean, injected)
    assert set(diff) <= {"values_changed"}, diff.keys()
    planted = {normalize(t) for t in injections.values()}
    changed = diff.get("values_changed", {})
    for path, change in changed.items():
        if path.endswith("['size']"):
            # AgentDojo's CloudDrive derives a file's size from its content: allowed only beside a content change
            assert path[: -len("['size']")] + "['content']" in changed, path
            continue
        assert any(p in normalize(str(change["new_value"])) for p in planted), path
