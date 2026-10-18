"""Watchdog for long benchmark runs: detect crashes, vanished jobs, stalls, and slowdowns, then retry with a
different launch method until the store is complete and correct (owner request, 2026-10-18).

The job always runs through the same wrapper as scripts/bg.sh (pid file, exit file, log under runs/logs/),
so its state is read the same way whatever launched it. Methods, tried in order:

1. `bg`      - scripts/bg.sh (setsid nohup), the configured workers
2. `bg`      - the same again (a plain resume after a transient failure)
3. `systemd` - systemd-run --user, one worker
4. `tmux`    - a detached tmux session, one worker
5. `child`   - a direct child of the watchdog, one worker

Every retry first stops what is left of the job and this project's model server (`serve.sh down`), then
resumes: the runner skips case ids already in the store, and the store refuses a different commit. The
watchdog never commits, never changes the configuration, and never touches other projects' containers.
A run ends successfully only when the store holds every grid case exactly once (aib.gates.check_store).
When every method has failed, it appends a blocker to docs/BLOCKERS.md and exits non-zero.

Usage: python -m aib.watchdog --model <key> [--name run_<key>] [--stall-minutes 30] [--poll-seconds 60]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from aib.repro import REPO_ROOT

METHODS: tuple[tuple[str, int | None], ...] = (("bg", None), ("bg", None), ("systemd", 1), ("tmux", 1), ("child", 1))
NO_RETRY_EXITS = {2: "configuration or usage error", 3: "the Spend_Cap stopped the run"}


@dataclass
class Observation:
    state: str  # RUNNING | DONE | LOST | NONE
    exit_code: int | None
    done_cases: int
    healthy: bool | None  # model server /health, None when not checked


@dataclass
class Attempt:
    method: str
    workers: int | None
    started: float
    outcome: str = ""
    done_at_start: int = 0
    rates: list[float] = field(default_factory=list)


class JobControl:
    """Launch, observe, and stop one named job on this machine (all methods share the bg.sh wrapper)."""

    def __init__(self, root: Path, name: str, command: list[str]) -> None:
        self.root, self.name, self.command = root, name, command
        self.logs = root / "runs" / "logs"
        self.child: subprocess.Popen | None = None
        self.launches = 0

    @property
    def unit(self) -> str:
        return f"aib-{self.name}".replace(".", "-")

    def wrapper(self, cmd: list[str]) -> list[str]:
        """The bg.sh wrapper: write the pid, run the command through wsl_run.sh, write the exit status."""
        inner = ('n="$1"; shift; echo "$$" > "runs/logs/$n.pid"; bash scripts/wsl_run.sh "$@"; '
                 'echo "$?" > "runs/logs/$n.exit"')
        return ["bash", "-c", inner, "_", self.name, *cmd]

    def launch(self, method: str, workers: int | None) -> None:
        self.logs.mkdir(parents=True, exist_ok=True)
        self.launches += 1
        log = self.logs / f"{self.name}.log"
        if log.exists():  # bg.sh truncates the log: keep the previous attempt's copy
            log.replace(self.logs / f"{self.name}.before-attempt{self.launches}.log")
        for suffix in ("exit", "pid"):
            (self.logs / f"{self.name}.{suffix}").unlink(missing_ok=True)
        cmd = [*self.command, *(["--workers", str(workers)] if workers is not None else [])]
        wrapper = self.wrapper(cmd)
        if method == "bg":
            subprocess.run(["bash", "scripts/bg.sh", self.name, *cmd], cwd=self.root, check=True, capture_output=True)
        elif method == "systemd":
            subprocess.run(["systemctl", "--user", "reset-failed", self.unit], capture_output=True)
            # user units start with a minimal PATH (no /usr/lib/wsl/lib, so no nvidia-smi) and without the docker
            # group, so pass this process's PATH and run under `sg docker`
            subprocess.run(["systemd-run", "--user", f"--unit={self.unit}", "--collect", "--quiet",
                            f"--working-directory={self.root}", f"--setenv=PATH={os.environ.get('PATH', '')}",
                            f"--setenv=HOME={os.environ.get('HOME', '')}", f"--property=StandardOutput=append:{log}",
                            f"--property=StandardError=append:{log}", "sg", "docker", "-c",
                            " ".join(shlex.quote(a) for a in wrapper)], cwd=self.root, check=True, capture_output=True)
        elif method == "tmux":
            session = self.unit
            subprocess.run(["tmux", "kill-session", "-t", session], capture_output=True)
            line = " ".join(shlex.quote(a) for a in wrapper) + f" >> {shlex.quote(str(log))} 2>&1"
            subprocess.run(["tmux", "new-session", "-d", "-s", session, "-c", str(self.root), line], check=True,
                           capture_output=True)
        elif method == "child":
            with log.open("a") as f:
                self.child = subprocess.Popen(wrapper, cwd=self.root, stdout=f, stderr=subprocess.STDOUT,
                                              start_new_session=True)
        else:
            raise ValueError(f"unknown launch method {method!r}")
        for _ in range(50):  # wait for the wrapper to write its pid
            if (self.logs / f"{self.name}.pid").exists():
                break
            time.sleep(0.1)

    def pid(self) -> int | None:
        try:
            return int((self.logs / f"{self.name}.pid").read_text().strip())
        except (OSError, ValueError):
            return None

    def state(self) -> tuple[str, int | None]:
        exit_file = self.logs / f"{self.name}.exit"
        if exit_file.exists():
            try:
                return "DONE", int(exit_file.read_text().strip() or "1")
            except ValueError:
                return "DONE", 1
        pid = self.pid()
        if pid is None:
            return "NONE", None
        try:
            stat = Path(f"/proc/{pid}/stat").read_text().split()
            if stat[2] != "Z":
                return "RUNNING", None
        except OSError:
            pass
        return ("DONE", int(exit_file.read_text().strip())) if exit_file.exists() else ("LOST", None)

    def _descendants(self, pid: int) -> list[int]:
        out = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True).stdout.split()
        kids = [int(p) for p in out]
        return kids + [d for k in kids for d in self._descendants(k)]

    def stop(self) -> None:
        """Stop the job's whole process tree, then this project's model server."""
        pid = self.pid()
        if pid is not None:
            tree = [pid, *self._descendants(pid)]
            for sig in (signal.SIGTERM, signal.SIGKILL):
                for p in tree:
                    try:
                        os.kill(p, sig)
                    except OSError:
                        pass
                time.sleep(5)
        for unit_cmd in (["systemctl", "--user", "stop", self.unit], ["tmux", "kill-session", "-t", self.unit]):
            subprocess.run(unit_cmd, capture_output=True)
        subprocess.run(["bash", "scripts/serve.sh", "down"], cwd=self.root, capture_output=True)


def server_healthy(base_url: str, timeout: float = 5.0) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


class Watchdog:
    def __init__(self, observe: Callable[[], Observation], launch: Callable[[str, int | None], None],
                 stop: Callable[[], None], complete: Callable[[], list[str]], total: int, *,
                 stall_s: float = 1800, poll_s: float = 60, slow_window_s: float = 1800, slow_fraction: float = 0.25,
                 unhealthy_polls: int = 5, methods=METHODS, log: Callable[[str], None] = print,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic) -> None:
        self.observe, self.launch, self.stop, self.complete = observe, launch, stop, complete
        self.total, self.stall_s, self.poll_s = total, stall_s, poll_s
        self.slow_window_s, self.slow_fraction, self.unhealthy_polls = slow_window_s, slow_fraction, unhealthy_polls
        self.methods, self.log, self.sleep, self.clock = methods, log, sleep, clock
        self.attempts: list[Attempt] = []

    def _watch(self, attempt: Attempt) -> str:
        """Poll one attempt until it ends; return 'done', 'exit:<n>', 'lost', 'stalled', 'unhealthy', or 'slow'."""
        last_count, last_change = attempt.done_at_start, self.clock()
        window_start, window_count = self.clock(), attempt.done_at_start
        bad_health, seen_healthy = 0, False
        while True:
            self.sleep(self.poll_s)
            obs = self.observe()
            now = self.clock()
            if obs.done_cases > last_count:
                last_count, last_change = obs.done_cases, now
            if obs.state == "DONE":
                return "done" if obs.exit_code == 0 else f"exit:{obs.exit_code}"
            if obs.state in ("LOST", "NONE"):
                return "lost"
            if now - last_change >= self.stall_s:
                return "stalled"
            seen_healthy = seen_healthy or obs.healthy is True
            bad_health = bad_health + 1 if (obs.healthy is False and seen_healthy) else 0
            if bad_health >= self.unhealthy_polls:
                return "unhealthy"
            if now - window_start >= self.slow_window_s:
                rate = (obs.done_cases - window_count) / (now - window_start)
                attempt.rates.append(rate)
                earlier = sorted(attempt.rates[:-1])
                if earlier and rate < self.slow_fraction * earlier[len(earlier) // 2]:
                    return "slow"
                window_start, window_count = now, obs.done_cases

    def run(self) -> tuple[bool, str]:
        obs = self.observe()
        if obs.state == "RUNNING":  # a restarted watchdog adopts the job instead of killing a healthy run
            attempt = Attempt("adopted", None, self.clock(), done_at_start=obs.done_cases)
            self.attempts.append(attempt)
            self.log(f"adopting the running job at {obs.done_cases}/{self.total} cases")
            attempt.outcome = self._watch(attempt)
            self.log(f"adopted job ended: {attempt.outcome}")
            if attempt.outcome == "done" and not self.complete():
                return True, "complete (adopted job)"
            self.stop()
        for method, workers in self.methods:
            obs = self.observe()
            if obs.state == "RUNNING":
                self.log("a job with this name is still running; stopping it before a controlled restart")
                self.stop()
                obs = self.observe()
            if obs.done_cases >= self.total and not self.complete():
                self.log(f"store already complete and correct ({obs.done_cases} cases)")
                return True, "already complete"
            attempt = Attempt(method, workers, self.clock(), done_at_start=obs.done_cases)
            self.attempts.append(attempt)
            self.log(f"attempt {len(self.attempts)}: method={method} workers={workers or 'configured'} "
                     f"starting at {obs.done_cases}/{self.total} cases")
            try:
                self.launch(method, workers)
            except Exception as exc:  # the launcher itself failed: try the next method
                attempt.outcome = f"launch failed: {type(exc).__name__}: {exc}"
                self.log(attempt.outcome)
                continue
            attempt.outcome = self._watch(attempt)
            self.log(f"attempt {len(self.attempts)} ended: {attempt.outcome}")
            if attempt.outcome == "done":
                problems = self.complete()
                if not problems:
                    return True, f"complete after {len(self.attempts)} attempt(s)"
                attempt.outcome = f"done but incomplete: {problems[:3]}"
                self.log(attempt.outcome)
            code = int(attempt.outcome.split(":")[1]) if attempt.outcome.startswith("exit:") else None
            if code in NO_RETRY_EXITS:
                return False, f"exit {code}: {NO_RETRY_EXITS[code]}; not retried"
            if attempt.outcome in ("stalled", "unhealthy", "slow", "lost") or code is not None:
                self.stop()
        return False, "every launch method failed: " + "; ".join(f"{a.method}: {a.outcome}" for a in self.attempts)


def record_blocker(root: Path, task: str, command: str, reason: str, attempts: list[Attempt], log_path: str) -> None:
    path = root / "docs" / "BLOCKERS.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [] if path.exists() else ["# Blockers", ""]
    lines += [f"## {task}: {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}", "",
              f"- Command: `{command}`", f"- Outcome: {reason}", f"- Log: `{log_path}`",
              "- Attempts: " + "; ".join(f"{a.method} (workers {a.workers or 'configured'}): {a.outcome}" for a in attempts),
              "- Suspected cause: see the log; the watchdog retried with every launch method.", ""]
    with path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    from aib.config import load_all
    from aib.experiments import ResultsStore, build_grid
    from aib.gates import check_store

    p = argparse.ArgumentParser(description="Run one model's benchmark under a crash and stall watchdog.")
    p.add_argument("--model", required=True)
    p.add_argument("--name")
    p.add_argument("--task", default="benchmark run")
    p.add_argument("--stall-minutes", type=float, default=30)
    p.add_argument("--poll-seconds", type=float, default=60)
    args = p.parse_args(argv)
    cfg = load_all(REPO_ROOT / "config")
    m = cfg.models.models[args.model]
    hosted = m.provider == "openai"
    specs = build_grid(args.model, cfg.experiment, hosted=hosted, hosted_conditions=cfg.hosted.conditions,
                       hosted_defenses=cfg.hosted.defenses)
    ids = {s.case_id for s in specs}
    store_path = REPO_ROOT / "results" / "raw" / f"{args.model}.jsonl"
    name = args.name or f"run_{args.model}"
    command = ["bash", "scripts/run_benchmark.sh", args.model]
    job = JobControl(REPO_ROOT, name, command)
    wlog = REPO_ROOT / "runs" / "logs" / f"watchdog-{name}.log"
    wlog.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        line = f"{dt.datetime.now().isoformat(timespec='seconds')} {msg}"
        print(line, flush=True)
        with wlog.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def observe() -> Observation:
        state, code = job.state()
        done = len(ResultsStore(store_path).completed_ids() & ids) if store_path.exists() else 0
        healthy = None if hosted or state != "RUNNING" else server_healthy(cfg.models.server.base_url)
        return Observation(state, code, done, healthy)

    dog = Watchdog(observe, job.launch, job.stop, lambda: check_store(store_path, ids), len(ids),
                   stall_s=args.stall_minutes * 60, poll_s=args.poll_seconds, log=log)
    ok, reason = dog.run()
    log(f"watchdog finished: {'OK' if ok else 'FAILED'}: {reason}")
    if not ok:
        record_blocker(REPO_ROOT, args.task, " ".join(command), reason, dog.attempts, f"runs/logs/{name}.log")
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
