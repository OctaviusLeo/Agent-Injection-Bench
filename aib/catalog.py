"""M1 catalog evidence: every User_Task with its ground truth, placeholders, and check result (task 2.3; Req 2.1, 2.3, 2.7).

Usage: python -m aib.catalog [--out results/m1_catalog.json]
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import sys

from aib.repro import REPO_ROOT, make_stamp, require_clean
from aib.suite import (
    BENCHMARK_VERSION,
    injection_candidates,
    load_user_tasks,
    load_workspace_suite,
    run_ground_truth,
    task_success_check,
    task_number,
    validate_catalog,
)

GOAL_IDS = tuple(f"injection_task_{i}" for i in range(6))


def build_catalog() -> dict:
    """The catalog content (no stamp); deterministic, so regenerating it gives identical content."""
    suite = load_workspace_suite()
    tasks = []
    for ref in load_user_tasks(suite):
        run = run_ground_truth(ref.task, suite=suite)
        outcome = task_success_check(ref.task, run.output, run.pre_env, run.post_env, run.trace)
        tasks.append({
            "id": ref.id,
            "prompt": ref.task.PROMPT,
            "ground_truth_calls": [{"function": c.function, "args": json.loads(json.dumps(dict(c.args), default=str))}
                                   for c in run.trace],
            "injection_placeholders": injection_candidates(ref.task, suite),
            "ground_truth_passes": outcome.passed,
            "version_module": ref.module,
        })
    goals = sorted(GOAL_IDS, key=task_number)
    return {
        "suite": suite.name,
        "agentdojo_version": md.version("agentdojo"),
        "benchmark_version": list(BENCHMARK_VERSION),
        "register_v1_1_2": True,
        "user_task_count": len(tasks),
        "tools": [t.name for t in suite.tools],
        "tool_count": len(suite.tools),
        "version_mapping": {
            "user_tasks": {t["id"]: t["version_module"] for t in tasks},
            "injection_tasks": {g: type(suite.get_injection_task_by_id(g)).__module__ for g in goals},
        },
        "validate_catalog": validate_catalog(suite).to_dict(),
        "user_tasks": tasks,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the M1 catalog evidence.")
    parser.add_argument("--out", default="results/m1_catalog.json")
    args = parser.parse_args(argv)
    require_clean()
    command = "python scripts/catalog.py" + ("" if args.out == "results/m1_catalog.json" else f" --out {args.out}")
    stamp = make_stamp(command, 0)
    catalog = build_catalog()
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"stamp": stamp.to_dict(), **catalog}, indent=2) + "\n", encoding="utf-8")
    report = catalog["validate_catalog"]
    print(f"{catalog['user_task_count']} user tasks, {catalog['tool_count']} tools, "
          f"{report['failure_count']} catalog failures; wrote {args.out}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
