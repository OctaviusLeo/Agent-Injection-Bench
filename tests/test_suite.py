"""Unit tests for aib.suite: version mapping, catalog size, ground-truth checks, catalog validation."""

from __future__ import annotations

import pytest
from agentdojo.base_tasks import BaseUserTask

from aib import suite as S

USER_TASKS = S.load_user_tasks()


def test_version_mapping_includes_the_v1_1_2_fixes():
    suite = S.load_workspace_suite()
    inj = {k: type(v).__module__ for k, v in suite.injection_tasks.items()}
    usr = {k: type(v).__module__ for k, v in suite.user_tasks.items()}
    for k in ("injection_task_0", "injection_task_4", "injection_task_5"):
        assert inj[k] == "agentdojo.default_suites.v1_1_2.workspace.injection_tasks"
    assert inj["injection_task_3"] == "agentdojo.default_suites.v1_2.workspace.injection_tasks"
    assert inj["injection_task_1"] == inj["injection_task_2"] == "agentdojo.default_suites.v1.workspace.injection_tasks"
    for k in ("user_task_13", "user_task_33"):
        assert usr[k] == "agentdojo.default_suites.v1_1_2.workspace.user_tasks"
    for k in ("user_task_0", "user_task_17", "user_task_18", "user_task_31", "user_task_32"):
        assert usr[k] == "agentdojo.default_suites.v1_2.workspace.user_tasks"
    assert usr["user_task_16"] == "agentdojo.default_suites.v1_2_2.workspace.user_tasks"
    assert suite.benchmark_version == (1, 2, 2)


def test_forty_user_tasks_ordered_and_twenty_four_tools():
    assert [r.id for r in USER_TASKS] == [f"user_task_{i}" for i in range(40)]
    assert len(S.load_workspace_suite().tools) == 24
    names = {t.name for t in S.load_workspace_suite().tools}
    assert {"send_email", "search_emails", "create_calendar_event", "list_files", "delete_file"} <= names


@pytest.mark.parametrize("ref", USER_TASKS, ids=lambda r: r.id)
def test_ground_truth_passes_and_empty_run_fails_with_reason(ref):
    run = S.run_ground_truth(ref.task)
    assert S.task_success_check(ref.task, run.output, run.pre_env, run.post_env, run.trace) == S.CheckOutcome(True, None)
    env = S.init_case_environment(ref.task, {})
    empty = S.task_success_check(ref.task, "", env, env.model_copy(deep=True), [])
    assert empty.passed is False
    assert empty.unmet and ref.id in empty.unmet and "environment changes: none" in empty.unmet
    assert 1 <= len(S.injection_candidates(ref.task)) <= 4


def test_check_reports_environment_changes_and_predicate_errors(monkeypatch):
    task = S.get_user_task("user_task_0")
    run = S.run_ground_truth(S.get_injection_task("injection_task_1"))  # deletes cloud file 13
    out = S.task_success_check(task, "", run.pre_env, run.post_env, run.trace)
    assert not out.passed and "cloud_drive" in out.unmet
    monkeypatch.setattr(type(task), "utility", lambda self, *a, **k: 1 / 0)
    broken = S.task_success_check(task, "x", run.pre_env, run.pre_env, [])
    assert not broken.passed and "raised ZeroDivisionError" in broken.unmet


def test_check_outcome_invariant():
    with pytest.raises(ValueError):
        S.CheckOutcome(True, "reason")
    with pytest.raises(ValueError):
        S.CheckOutcome(False, None)


def test_validate_catalog_passes_on_the_suite():
    report = S.validate_catalog()
    assert report.ok and report.total == 40 and report.tools == 24
    assert report.to_dict()["failure_count"] == 0


def test_validate_catalog_reports_a_task_without_check(monkeypatch):
    task = S.get_user_task("user_task_5")
    monkeypatch.setattr(type(task), "utility", BaseUserTask.utility)  # abstract: no resolvable predicate
    report = S.validate_catalog()
    assert report.total == 40
    assert report.failures == [("user_task_5", "no resolvable Task_Success_Check (utility predicate)")]
    assert report.to_dict()["failure_count"] == 1 and not report.ok


def test_validate_catalog_reports_failing_ground_truth_and_no_placeholder(monkeypatch):
    t7 = S.get_user_task("user_task_7")
    monkeypatch.setattr(type(t7), "utility", lambda self, *a, **k: False)
    monkeypatch.setattr(type(t7), "utility_from_traces", lambda self, *a, **k: None)
    monkeypatch.setattr(S, "injection_candidates", lambda task, suite=None: [] if task.ID == "user_task_8" else ["x"])
    reasons = dict(S.validate_catalog().failures)
    assert reasons["user_task_7"].startswith("ground truth fails its check")
    assert reasons["user_task_8"] == "ground truth reads no injection placeholder"
    assert len(reasons) == 2


def test_validate_catalog_reports_exceptions(monkeypatch):
    t9 = S.get_user_task("user_task_9")
    monkeypatch.setattr(type(t9), "ground_truth", lambda self, env: (_ for _ in ()).throw(RuntimeError("boom")))
    reasons = dict(S.validate_catalog().failures)
    assert reasons["user_task_9"] == "RuntimeError: boom"


def test_injection_candidates_with_an_explicit_suite_copy():
    suite = S.load_workspace_suite().get_new_version((1, 2, 2))
    task = suite.get_user_task_by_id("user_task_13")
    assert S.injection_candidates(task, suite) == ["drive_feedback_injection", "drive_team_building_injection"]


def test_task_number_rejects_bad_ids():
    assert S.task_number("user_task_12") == 12
    with pytest.raises(ValueError):
        S.task_number("user_task")
