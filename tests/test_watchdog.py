"""Unit tests for aib.watchdog: escalation logic with fakes, and real bg/child launches in a throwaway tree."""

from __future__ import annotations

import shutil
import time

import pytest

from aib import watchdog as W
from aib.repro import REPO_ROOT


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


class Script:
    """Feeds one list of observations per attempt; records launches and stops."""

    def __init__(self, per_attempt, total=100, complete_after=None):
        self.per_attempt = [list(a) for a in per_attempt]
        self.launches, self.stops = [], 0
        self.total = total
        self.current: list = []
        self.done = 0
        self.complete_after = complete_after

    def launch(self, method, workers):
        self.launches.append((method, workers))
        self.current = self.per_attempt.pop(0) if self.per_attempt else []

    def observe(self):
        if self.current:
            state, code, done, healthy = self.current.pop(0)
            self.done = done
            return W.Observation(state, code, done, healthy)
        return W.Observation("NONE", None, self.done, None)

    def stop(self):
        self.stops += 1
        self.current = []  # a real stop ends the job

    def complete(self):
        return [] if self.done >= self.total else [f"{self.total - self.done} grid cases missing"]


def _dog(script, **kw):
    clock = Clock()
    logs = []
    dog = W.Watchdog(script.observe, script.launch, script.stop, script.complete, script.total, poll_s=60,
                     stall_s=1800, slow_window_s=1800, log=logs.append, sleep=clock.sleep, clock=clock, **kw)
    return dog, logs


R = "RUNNING"


def test_success_on_first_attempt():
    s = Script([[(R, None, 50, True), ("DONE", 0, 100, None)]])
    dog, logs = _dog(s)
    assert dog.run() == (True, "complete after 1 attempt(s)")
    assert s.launches == [("bg", None)] and s.stops == 0


def test_lost_job_moves_to_the_next_method_and_resumes():
    s = Script([[(R, None, 30, True), ("LOST", None, 40, None)], [(R, None, 70, True), ("DONE", 0, 100, None)]])
    dog, logs = _dog(s)
    ok, reason = dog.run()
    assert ok and s.launches == [("bg", None), ("bg", None)] and s.stops == 1
    assert dog.attempts[0].outcome == "lost" and dog.attempts[1].done_at_start == 40


def test_stall_is_detected_and_escalates_to_systemd_with_one_worker():
    stalled = [(R, None, 10, True)] * 40  # no progress for 40 polls of 60 s
    s = Script([stalled, stalled, [(R, None, 60, True), ("DONE", 0, 100, None)]])
    dog, _ = _dog(s)
    ok, _ = dog.run()
    assert ok and [a.outcome for a in dog.attempts[:2]] == ["stalled", "stalled"]
    assert s.launches[2] == ("systemd", 1) and s.stops == 2


def test_unhealthy_server_triggers_restart():
    s = Script([[(R, None, i, False) for i in range(1, 10)], [("DONE", 0, 100, None)]])
    dog, _ = _dog(s)
    assert dog.run()[0] and dog.attempts[0].outcome == "unhealthy"


def test_slowdown_triggers_restart():
    fast = [(R, None, i * 10, True) for i in range(1, 61)]  # 10 cases per poll for an hour
    slow = [(R, None, 600 + i, True) for i in range(1, 31)]  # then 1 case per poll
    s = Script([fast + slow, [("DONE", 0, 1000, None)]], total=1000)
    dog, _ = _dog(s)
    assert dog.run()[0] and dog.attempts[0].outcome == "slow"


def test_crash_exit_is_retried_but_config_and_cap_exits_are_not():
    s = Script([[("DONE", 1, 20, None)], [("DONE", 0, 100, None)]])
    dog, _ = _dog(s)
    assert dog.run()[0] and dog.attempts[0].outcome == "exit:1" and s.stops == 1
    for code in (2, 3):
        s2 = Script([[("DONE", code, 20, None)]])
        dog2, _ = _dog(s2)
        ok, reason = dog2.run()
        assert not ok and "not retried" in reason and len(s2.launches) == 1


def test_done_but_incomplete_store_is_retried():
    s = Script([[("DONE", 0, 90, None)], [("DONE", 0, 100, None)]])
    dog, _ = _dog(s)
    assert dog.run()[0] and dog.attempts[0].outcome.startswith("done but incomplete")


def test_every_method_failing_reports_all_attempts_and_launch_errors():
    s = Script([[("LOST", None, 0, None)]] * 5)
    original = s.launch

    def flaky(method, workers):
        if method == "tmux":
            raise OSError("tmux missing")
        original(method, workers)

    dog, _ = _dog(s)
    dog.launch = flaky
    ok, reason = dog.run()
    assert not ok and reason.startswith("every launch method failed")
    assert [m for m, _ in s.launches] == ["bg", "bg", "systemd", "child"]
    assert "launch failed: OSError: tmux missing" in reason


def test_running_job_is_adopted_not_killed():
    s = Script([], total=10)
    s.current = [(R, None, 5, True), (R, None, 8, True), ("DONE", 0, 10, None)]
    dog, logs = _dog(s)
    assert dog.run() == (True, "complete (adopted job)") and s.stops == 0 and s.launches == []
    assert "adopting the running job at 5/10 cases" in logs[0]


def test_adopted_job_that_stalls_escalates():
    s = Script([[("DONE", 0, 10, None)]], total=10)
    s.current = [(R, None, 5, True)] * 40
    dog, _ = _dog(s)
    assert dog.run()[0] and dog.attempts[0].outcome == "stalled" and s.stops == 1 and s.launches == [("bg", None)]


def test_already_complete_store():
    s = Script([], total=10)
    s.done = 10
    dog, _ = _dog(s)
    assert dog.run() == (True, "already complete") and s.launches == []


def test_record_blocker(tmp_path):
    attempts = [W.Attempt("bg", None, 0.0, outcome="stalled"), W.Attempt("child", 1, 0.0, outcome="lost")]
    W.record_blocker(tmp_path, "4.7", "bash scripts/run_benchmark.sh llama3.1-8b", "every launch method failed", attempts,
                     "runs/logs/run_llama3.1-8b.log")
    W.record_blocker(tmp_path, "4.7", "cmd", "again", attempts, "log")
    text = (tmp_path / "docs" / "BLOCKERS.md").read_text()
    assert text.startswith("# Blockers") and text.count("## 4.7:") == 2 and "child (workers 1): lost" in text


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    shutil.copy(REPO_ROOT / "scripts" / "bg.sh", root / "scripts" / "bg.sh")
    (root / "scripts" / "wsl_run.sh").write_text('#!/usr/bin/env bash\ncd "$(dirname "$(readlink -f "$0")")/.."\nexec "$@"\n')
    (root / "scripts" / "serve.sh").write_text("#!/usr/bin/env bash\nexit 0\n")
    return root


def _wait_done(job, limit=10.0):
    end = time.time() + limit
    while time.time() < end:
        state, code = job.state()
        if state == "DONE":
            return code
        time.sleep(0.1)
    return None


@pytest.mark.parametrize("method", ["bg", "child"])
def test_real_launch_reports_exit_codes(tree, method):
    job = W.JobControl(tree, f"t_{method}", ["bash", "-c", 'exit "${@: -1}"', "_"])
    job.launch(method, 0)  # appends "--workers 0": the command exits with the last argument
    assert _wait_done(job) == 0
    assert job.pid() is not None


def test_real_stop_kills_the_tree_and_state_becomes_lost(tree):
    job = W.JobControl(tree, "t_sleep", ["sleep", "60"])
    job.launch("child", None)
    assert job.state()[0] == "RUNNING"
    job.stop()
    if job.child is not None:
        job.child.wait(timeout=10)
    assert job.state()[0] in ("LOST", "DONE")
    assert W.JobControl(tree, "never_started", ["true"]).state() == ("NONE", None)
    with pytest.raises(ValueError):
        job.launch("teleport", None)


@pytest.mark.parametrize("outcome", [(True, "complete after 1 attempt(s)"), (False, "every launch method failed: x")])
def test_main_wires_store_checks_and_writes_a_blocker_on_failure(tmp_path, monkeypatch, outcome):
    root = tmp_path / "repo"
    shutil.copytree(REPO_ROOT / "config", root / "config")
    seen = {}

    class FakeDog:
        def __init__(self, observe, launch, stop, complete, total, **kw):
            seen["obs"] = observe()
            seen["problems"] = complete()
            seen["total"] = total
            self.attempts = [W.Attempt("bg", None, 0.0, outcome="lost")]

        def run(self):
            return outcome

    monkeypatch.setattr(W, "REPO_ROOT", root)
    monkeypatch.setattr(W, "Watchdog", FakeDog)
    code = W.main(["--model", "qwen3-4b", "--task", "4.6", "--poll-seconds", "1"])
    assert code == (0 if outcome[0] else 1)
    assert seen["total"] == 2880 and seen["obs"].state == "NONE" and seen["obs"].done_cases == 0
    assert "results/raw/qwen3-4b.jsonl is missing" in seen["problems"][0]
    assert (root / "runs" / "logs" / "watchdog-run_qwen3-4b.log").read_text().count("watchdog finished") == 1
    assert (root / "docs" / "BLOCKERS.md").exists() == (not outcome[0])


def test_server_healthy_is_false_without_a_server():
    assert W.server_healthy("http://127.0.0.1:9", timeout=0.5) is False
