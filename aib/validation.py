"""M2 stub validation: every task under every style with both Stub_Model policies (task 3.5; Req 3.2, 3.4, 3.6).

Usage: python -m aib.validation [--out results/m2_stub_validation.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from aib.attacks import STYLES
from aib.goals import goal_for
from aib.repro import REPO_ROOT, make_stamp, require_clean
from aib.runner import CaseSpec, RunContext, run_case
from aib.stub_model import StubLLM
from aib.suite import load_user_tasks

POLICIES = ("follow_user", "follow_injection")


def run_stub_validation(stamp: dict, task_ids: list[str] | None = None, styles: list[str] | None = None) -> dict:
    task_ids = task_ids or [r.id for r in load_user_tasks()]
    styles = styles or list(STYLES)
    ctx = RunContext(sampling={"policy": "stub"}, stamp=stamp, model_version="stub")
    cases, summary = [], {}
    for policy in POLICIES:
        llm = StubLLM(policy)
        for style in styles:
            counts: Counter = Counter()
            for tid in task_ids:
                spec = CaseSpec(f"stub-{policy}", "none", style, tid, goal_for(tid), 0, 0)
                r = run_case(spec, llm, ctx=ctx)
                cases.append({"policy": policy, "style": style, "user_task_id": tid, "injection_task_id": r.injection_task_id,
                              "status": r.status, "exposed": r.exposed, "task_success": r.task_success,
                              "injection_success": r.injection_success, "error_cause": r.error_cause})
                counts[f"status:{r.status}"] += 1
                counts[f"task_success:{r.task_success}"] += 1
                counts[f"injection_success:{r.injection_success}"] += 1
                counts[f"exposed:{r.exposed}"] += 1
            summary.setdefault(policy, {})[style] = dict(sorted(counts.items()))
    return {"cases": cases, "summary": summary, "n_cases": len(cases), "styles": styles,
            "goals": sorted({c["injection_task_id"] for c in cases}, key=lambda g: int(g.rsplit("_", 1)[1]))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the M2 stub validation.")
    parser.add_argument("--out", default="results/m2_stub_validation.json")
    args = parser.parse_args(argv)
    require_clean()
    stamp = make_stamp("python scripts/stub_validation.py", 0).to_dict()
    result = run_stub_validation(stamp)
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"stamp": stamp, **result}, indent=2) + "\n", encoding="utf-8")
    for policy, by_style in result["summary"].items():
        for style, counts in by_style.items():
            print(f"{policy:16s} {style:24s} {counts}")
    print(f"{result['n_cases']} cases; wrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
