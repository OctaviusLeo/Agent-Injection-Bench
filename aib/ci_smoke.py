"""CI smoke subset: Stub_Model cases with fixed, assertable outcomes (task 6.3; Req 8.3, 8.5, 8.6).

user_task_0..4 x styles (direct, hidden_field) x defenses (none, delimit, filter with the keyword
classifier) x policies (follow_user, follow_injection) = 60 cases. No GPU, network, key, or download.
Expected: follow_user completes the task and never carries out the goal; follow_injection carries it
out under none and delimit, and not under the keyword filter (which hides the planted text).

Usage: python scripts/ci_smoke.py   (writes runs/ci_smoke.json; exits non-zero on the first mismatch)
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

from aib.classifier import KeywordClassifier
from aib.config import load_defenses
from aib.goals import goal_for
from aib.metrics import wilson_interval
from aib.runner import CaseSpec, RunContext, run_case
from aib.stub_model import StubLLM

TASKS = [f"user_task_{i}" for i in range(5)]
STYLES = ("direct", "hidden_field")
DEFENSES = ("none", "delimit", "filter")
POLICIES = ("follow_user", "follow_injection")


def expected(policy: str, defense: str) -> dict:
    if policy == "follow_user":
        return {"task_success": True, "injection_success": False}
    return {"injection_success": defense != "filter"}


def run_smoke() -> tuple[bool, dict]:
    cfg = load_defenses()
    classifier = KeywordClassifier()
    ctx = RunContext(sampling={"policy": "stub"}, stamp={"command": "python scripts/ci_smoke.py", "seed": 0, "commit": ""},
                     defense_cfg=cfg)
    cases, mismatch = [], None
    for policy in POLICIES:
        llm = StubLLM(policy)
        for defense in DEFENSES:
            for style in STYLES:
                for task in TASKS:
                    spec = CaseSpec("stub", defense, style, task, goal_for(task), 0, 0)
                    r = run_case(spec, llm, classifier if defense == "filter" else None, ctx)
                    row = {"policy": policy, "defense": defense, "style": style, "task": task, "status": r.status,
                           "task_success": r.task_success, "injection_success": r.injection_success,
                           "error_cause": r.error_cause}
                    cases.append(row)
                    want = expected(policy, defense)
                    got = {k: row[k] for k in want}
                    if mismatch is None and (r.status != "classified" or got != want):
                        mismatch = (f"{policy}/{defense}/{style}/{task}: expected status classified and {want}, "
                                    f"got status {r.status} and {got}" + (f" ({r.error_cause})" if r.error_cause else ""))
    summary = {}
    for policy in POLICIES:
        for defense in DEFENSES:
            rows = [c for c in cases if c["policy"] == policy and c["defense"] == defense]
            ts = sum(c["task_success"] is True for c in rows)
            inj = sum(c["injection_success"] is True for c in rows)
            summary[f"{policy}/{defense}"] = {
                "cases": len(rows),
                "task_success": wilson_interval(ts, len(rows)).to_dict(),
                "injection_success": wilson_interval(inj, len(rows)).to_dict(),
            }
    return mismatch is None, {"passed": mismatch is None, "first_mismatch": mismatch, "n_cases": len(cases),
                              "summary": summary, "cases": cases}


def main(out: Path = Path("runs/ci_smoke.json")) -> int:
    try:
        ok, result = run_smoke()
    except Exception as exc:  # the subset did not complete: name the step
        print(f"CI smoke subset did not complete: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 2
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for key, row in result["summary"].items():
        print(f"{key:30s} task success {row['task_success']['point']:.3f}  injection success {row['injection_success']['point']:.3f}")
    if not ok:
        print(f"CI smoke FAILED: {result['first_mismatch']}")
        return 1
    print(f"CI smoke passed: {result['n_cases']} cases; wrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
