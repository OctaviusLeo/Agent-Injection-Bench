"""Grid runner and Results_Store (task 4.4; Req 5.1, 5.3-5.7, 0.2).

The grid for a local model is all four defenses x six conditions x 40 tasks x 3 Repetitions, in
(defense, condition, task, rep) order; the hosted grid is the hosted conditions in Repetition-major
order, so a Spend_Cap stop leaves every Cell with the same number of complete Repetitions, give or
take one. Each case's seed comes from the base seed and the case id. The store is
results/raw/<model>.jsonl: a header line with the run's stamp and configuration, then one line per
case, each written with flush and fsync. A run resumes only on the commit that started it.

CLI: python -m aib.experiments run --model <key> [--pilot] [--workers N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from pathlib import Path

from aib.config import AllConfig, ExperimentConfig, load_all
from aib.goals import goal_for
from aib.ledger import SpendCapExceeded
from aib.llm import MissingSecretError
from aib.repro import REPO_ROOT, make_stamp, require_clean
from aib.runner import INFRA_ERROR_KINDS, CaseResult, CaseSpec, RunContext, run_case

MAX_CONSECUTIVE_INFRA = 5


class StoreError(RuntimeError):
    pass


def case_seed(base_seed: int, spec_key: str) -> int:
    """First 8 bytes of SHA-256(f"{base_seed}|{key}") as an integer, mod 2**31."""
    digest = hashlib.sha256(f"{base_seed}|{spec_key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


def _spec(model: str, defense: str, condition: str, task: str, rep: int, base_seed: int) -> CaseSpec:
    key = f"{model}|{defense}|{condition}|{task}|r{rep}"
    goal = None if condition == "clean" else goal_for(task)
    return CaseSpec(model, defense, condition, task, goal, rep, case_seed(base_seed, key))


def build_grid(model: str, cfg: ExperimentConfig, *, hosted: bool = False, hosted_conditions: list[str] | None = None,
               hosted_defenses: list[str] | None = None, pilot: bool = False) -> list[CaseSpec]:
    if pilot:
        tasks, reps = cfg.pilot.user_tasks, cfg.pilot.repetitions
    else:
        tasks, reps = cfg.user_tasks, cfg.repetitions
    if hosted and not pilot:
        conditions = hosted_conditions or cfg.conditions
        defenses = hosted_defenses or cfg.defenses
        return [_spec(model, d, c, t, r, cfg.base_seed)
                for r in range(reps) for d in defenses for c in conditions for t in tasks]
    return [_spec(model, d, c, t, r, cfg.base_seed)
            for d in cfg.defenses for c in cfg.conditions for t in tasks for r in range(reps)]


class ResultsStore:
    """results/raw/<model>.jsonl: one header line, then one CaseResult per line."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def _lines(self) -> list[dict]:
        if not self.path.exists():
            return []
        rows = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:  # a line cut off by a crash: ignored, the case reruns
                    continue
        return rows

    def header(self) -> dict | None:
        rows = self._lines()
        return rows[0] if rows and rows[0].get("header") else None

    def records(self) -> list[dict]:
        return [r for r in self._lines() if not r.get("header")]

    def completed_ids(self) -> set[str]:
        return {r["case_id"] for r in self.records()}

    def _write(self, obj: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def open(self, stamp: dict, config: dict) -> bool:
        """Write the header for a new store, or check the commit of an existing one. Returns True on resume."""
        with self._lock:
            existing = self.header()
            if existing is None:
                if self.path.exists() and self.path.stat().st_size > 0:
                    raise StoreError(f"{self.path} has no header line; move it aside before running")
                self._write({"header": True, "stamp": stamp, "config": config})
                return False
            if existing["stamp"].get("commit") != stamp.get("commit"):
                raise StoreError(
                    f"{self.path} was started on commit {existing['stamp'].get('commit')}, not {stamp.get('commit')}: "
                    "move the partial store to ~/aib-runs/discarded/<timestamp>/ and start the model over")
            return True

    def append(self, result: CaseResult) -> None:
        with self._lock:
            self._write(result.to_dict())


@dataclass
class RunSummary:
    model: str
    planned: int
    skipped: int = 0
    recorded: int = 0
    errored: int = 0
    infra_failures: int = 0
    stopped: bool = False
    stop_reason: str | None = None
    stopped_at_cap: bool = False
    statuses: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def run_grid(model: str, specs: list[CaseSpec], workers: int, store: ResultsStore,
             llm_factory: Callable[[CaseSpec], object], ctx: RunContext,
             classifier_factory: Callable[[], object] | None = None,
             on_result: Callable[[CaseResult], None] | None = None) -> RunSummary:
    """Run every spec not already in the store with `workers` threads. An infrastructure failure (5 consecutive
    cases failing with connection errors, timeouts, or 5xx after retries) stops the run without recording
    those cases; a Spend_Cap refusal stops it too; any other per-case failure is recorded as errored."""
    done = store.completed_ids()
    todo = [s for s in specs if s.case_id not in done]
    summary = RunSummary(model=model, planned=len(specs), skipped=len(specs) - len(todo))
    stop = threading.Event()
    lock = threading.Lock()
    consecutive = 0
    classifier_box: list = []

    def classifier_for(spec: CaseSpec):
        if spec.defense != "filter" or classifier_factory is None:
            return None
        with lock:
            if not classifier_box:
                classifier_box.append(classifier_factory())
            return classifier_box[0]

    def one(spec: CaseSpec):
        if stop.is_set():
            return spec, None, None
        try:
            return spec, run_case(spec, llm_factory(spec), classifier_for(spec), ctx), None
        except (SpendCapExceeded, MissingSecretError) as exc:
            return spec, None, exc

    pending = iter(todo)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = set()
        for _ in range(max(1, workers)):
            spec = next(pending, None)
            if spec is not None:
                futures.add(pool.submit(one, spec))
        while futures:
            finished, futures = wait(futures, return_when=FIRST_COMPLETED)
            for fut in finished:
                spec, result, exc = fut.result()
                if exc is not None:
                    stop.set()
                    summary.stopped = True
                    summary.stopped_at_cap = isinstance(exc, SpendCapExceeded)
                    summary.stop_reason = f"{type(exc).__name__}: {exc}"
                    continue
                if result is None:
                    continue
                if result.error_kind in INFRA_ERROR_KINDS:
                    summary.infra_failures += 1
                    consecutive += 1
                    if consecutive >= MAX_CONSECUTIVE_INFRA and not stop.is_set():
                        stop.set()
                        summary.stopped = True
                        summary.stop_reason = (f"infrastructure failure: {consecutive} consecutive cases failed "
                                               f"({result.error_cause})")
                    continue  # infrastructure failures are not recorded; a restart reruns them
                consecutive = 0
                store.append(result)
                summary.recorded += 1
                summary.errored += result.status == "errored"
                summary.statuses[result.status] = summary.statuses.get(result.status, 0) + 1
                if on_result is not None:
                    on_result(result)
            if not stop.is_set():
                while len(futures) < max(1, workers):
                    spec = next(pending, None)
                    if spec is None:
                        break
                    futures.add(pool.submit(one, spec))
    if not summary.stopped and summary.infra_failures:
        summary.stopped = True
        summary.stop_reason = f"{summary.infra_failures} cases hit infrastructure failures and were not recorded"
    return summary


# ---------------------------------------------------------------- CLI
def _sampling_record(cfg: AllConfig, model: str) -> dict:
    exp, m = cfg.experiment, cfg.models.models[model]
    if m.provider == "openai":
        accepted = cfg.hosted.accepted_params or []
        base = {"temperature": exp.sampling.temperature, "top_p": exp.sampling.top_p}
        rec = {k: v for k, v in base.items() if k in accepted}
        rec.update({"seed": "per request" if "seed" in accepted else None, "reasoning_effort": cfg.hosted.reasoning_effort,
                    "max_completion_tokens": cfg.hosted.max_completion_tokens, "accepted_params": accepted})
        return rec
    return {"temperature": exp.sampling.temperature, "top_p": exp.sampling.top_p, **exp.sampling.local_extra.model_dump(),
            "max_tokens": exp.max_tokens, "seed": "per request"}


def model_version(cfg: AllConfig, model: str) -> str:
    m = cfg.models.models[model]
    if m.provider == "openai":
        return f"{m.name} (response model field recorded per case)"
    return f"{m.repo}@{m.revision}/{m.files[0].name}"


def run_cli(model: str, pilot: bool, workers: int | None, root: Path = REPO_ROOT) -> int:
    from aib.classifier import InjectionClassifier
    from aib.hosted import make_hosted_llm, make_ledger
    from aib.llm import ChatLLM, Sampling

    require_clean(cwd=root)
    cfg = load_all(root / "config")
    m = cfg.models.models.get(model)
    if m is None or not m.enabled:
        print(f"model {model!r} is not an enabled model in config/models.yaml")
        return 2
    hosted = m.provider == "openai"
    if hosted and cfg.hosted.accepted_params is None:
        print("config/hosted.yaml has no accepted_params: run scripts/hosted_check.py first")
        return 2
    command = f"bash scripts/run_benchmark.sh {model}" + (" --pilot" if pilot else "")
    stamp = make_stamp(command, cfg.experiment.base_seed, cwd=root).to_dict()
    specs = build_grid(model, cfg.experiment, hosted=hosted, hosted_conditions=cfg.hosted.conditions,
                       hosted_defenses=cfg.hosted.defenses, pilot=pilot)
    workers = workers or m.workers
    store = ResultsStore(root / ("results/pilot" if pilot else "results/raw") / f"{model}.jsonl")
    config = {"experiment": cfg.experiment.model_dump(mode="json"), "model": {model: m.model_dump(mode="json")},
              "defenses": cfg.defenses.model_dump(mode="json"), "grid_size": len(specs), "workers": workers,
              "pilot": pilot, "sampling": _sampling_record(cfg, model)}
    if hosted:
        config["hosted"] = cfg.hosted.model_dump(mode="json")
    resumed = store.open(stamp, config)
    runs = Path(os.environ.get("AIB_RUNS", str(Path.home() / "aib-runs")))
    ctx = RunContext(sampling=_sampling_record(cfg, model), stamp=stamp, defense_cfg=cfg.defenses,
                     model_version=model_version(cfg, model),
                     trace_dir=runs / "traces" / (f"pilot-{model}" if pilot else model), max_iters=cfg.experiment.max_iters)
    ledger = make_ledger(cfg.hosted, root) if hosted else None
    if hosted:
        shared = make_hosted_llm(cfg.models, cfg.hosted, cfg.experiment, ledger, purpose=f"benchmark:{model}")
    else:
        sampling = Sampling(cfg.experiment.sampling.temperature, cfg.experiment.sampling.top_p, cfg.experiment.max_tokens,
                            cfg.experiment.sampling.local_extra.model_dump())
        shared = ChatLLM(model, f"{cfg.models.server.base_url}/v1", model, None, sampling, m.timeout_s)
    progress = {"n": 0}

    def report(result: CaseResult) -> None:
        progress["n"] += 1
        if progress["n"] % 20 == 0:
            print(f"[{model}] {progress['n']} recorded this session; last {result.case_id} -> {result.status}", flush=True)

    print(f"[{model}] {'resuming' if resumed else 'starting'} {len(specs)} cases with {workers} workers "
          f"(store {store.path.relative_to(root)})", flush=True)
    summary = run_grid(model, specs, workers, store, lambda spec: shared, ctx,
                       classifier_factory=lambda: InjectionClassifier(window_tokens=cfg.defenses.window_tokens,
                                                                      overlap=cfg.defenses.window_overlap_tokens,
                                                                      injection_label=cfg.defenses.injection_label),
                       on_result=report)
    total = len(store.completed_ids() & {s.case_id for s in specs})
    print(json.dumps({**summary.to_dict(), "complete_in_store": total}, indent=2), flush=True)
    if hosted and not pilot:
        write_hosted_summary(root, store, specs, summary, ledger.total_usd, cfg.hosted.spend_cap_usd, stamp)
    if summary.stopped_at_cap:
        return 3
    return 0 if total == len(specs) else 1


def write_hosted_summary(root: Path, store: ResultsStore, specs: list[CaseSpec], summary: RunSummary,
                         spend: float, cap: float, stamp: dict) -> dict:
    """results/hosted_run.json: spend reached, the cap, whether the run stopped at the cap, completed cases and Cells."""
    ids = {s.case_id for s in specs}
    records = [r for r in store.records() if r["case_id"] in ids]
    cells: dict[str, dict] = {}
    for r in records:
        key = f"{r['defense']}|{r['condition']}"
        cell = cells.setdefault(key, {"cases": 0, "reps": set()})
        cell["cases"] += 1
        cell["reps"].add(r["rep"])
    per_cell_tasks = len({s.user_task_id for s in specs})
    out = {
        "stamp": stamp, "spend_usd": spend, "spend_cap_usd": cap, "stopped_at_cap": summary.stopped_at_cap,
        "stop_reason": summary.stop_reason, "planned_cases": len(specs), "completed_cases": len(records),
        "cells": {k: {"cases": v["cases"], "complete_repetitions": sum(
            1 for rep in v["reps"] if sum(1 for r in records if f"{r['defense']}|{r['condition']}" == k and r["rep"] == rep) == per_cell_tasks)}
            for k, v in sorted(cells.items())},
    }
    path = root / "results" / "hosted_run.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one model's benchmark grid.")
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("run")
    p.add_argument("--model", required=True)
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--workers", type=int)
    args = parser.parse_args(argv)
    return run_cli(args.model, args.pilot, args.workers)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
