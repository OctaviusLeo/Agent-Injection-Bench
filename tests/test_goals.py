"""Unit tests for aib.goals."""

from __future__ import annotations

import pytest

from aib import goals as G
from aib import suite as S
from aib.attacks import plant

USER_TASKS = S.load_user_tasks()


def _check(goal_task, run):
    return G.injection_success_check(goal_task, run.output, run.pre_env, run.post_env, run.trace)


@pytest.mark.parametrize("goal_id", G.GOALS)
def test_goal_ground_truth_passes_in_a_fresh_environment(goal_id):
    goal = G.get_goal(goal_id)
    assert _check(goal, S.run_ground_truth(goal))


@pytest.mark.parametrize("ref", USER_TASKS, ids=lambda r: r.id)
def test_paired_user_task_ground_truth_alone_does_not_pass(ref):
    goal = G.get_goal(G.goal_for(ref.id))
    assert not _check(goal, S.run_ground_truth(ref.task))
    injections = plant("important_instructions", ref.task, goal)  # the goal text is planted, but not acted on
    assert not _check(goal, S.run_ground_truth(ref.task, injections))


def test_goal_assignment_is_k_mod_6():
    assert [G.goal_for(f"user_task_{k}") for k in (0, 5, 6, 13, 39)] == [
        "injection_task_0", "injection_task_5", "injection_task_0", "injection_task_1", "injection_task_3"]
    counts = {g: sum(G.goal_for(r.id) == g for r in USER_TASKS) for g in G.GOALS}
    assert counts == {"injection_task_0": 7, "injection_task_1": 7, "injection_task_2": 7, "injection_task_3": 7,
                      "injection_task_4": 6, "injection_task_5": 6}
    with pytest.raises(ValueError):
        G.goal_for("injection_task_1")


@pytest.mark.parametrize("bad", ["injection_task_6", "injection_task_13", "user_task_0", ""])
def test_any_other_goal_id_is_rejected(bad):
    with pytest.raises(ValueError):
        G.get_goal(bad)


def test_check_rejects_unconfigured_goal_and_tolerates_predicate_errors(monkeypatch):
    other = S.get_injection_task("injection_task_7")
    run = S.run_ground_truth(S.get_user_task("user_task_0"))
    with pytest.raises(ValueError):
        G.injection_success_check(other, run.output, run.pre_env, run.post_env, run.trace)
    goal = G.get_goal("injection_task_2")
    monkeypatch.setattr(type(goal), "security", lambda self, *a: 1 / 0)
    assert G.injection_success_check(goal, "", run.pre_env, run.post_env, []) is False


def test_classify_returns_each_status():
    assert G.classify("direct", errored=True, exposed=True) == "errored"
    assert G.classify("clean", errored=False, exposed=False) == "clean"
    assert G.classify("hidden_field", errored=False, exposed=False) == "unclassified_not_exposed"
    assert G.classify("fake_system", errored=False, exposed=True) == "classified"
    assert set(G.STATUSES) == {"errored", "clean", "unclassified_not_exposed", "classified"}
