"""Filter calibration check on ground-truth Tool_Outputs (task 4.3; Req 4.4, 4.5).

Runs every ground-truth Tool_Output of the 40 tasks, under each Injection_Style and the clean
condition, through the classifier at the fixed threshold. This checks the wiring and measures CPU
time; it does not tune anything (the threshold stays 0.5 whatever the result).

Usage: python -m aib.calibration [--out results/m3_filter_calibration.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from aib.attacks import STYLES, contains_injection, plant
from aib.classifier import InjectionClassifier
from aib.config import load_defenses
from aib.goals import get_goal, goal_for
from aib.metrics import rate_or_not_computed
from aib.repro import REPO_ROOT, make_stamp, require_clean
from aib.suite import load_user_tasks, run_ground_truth


def calibrate(classifier, threshold: float, task_ids: list[str] | None = None, conditions: list[str] | None = None) -> dict:
    task_ids = task_ids or [r.id for r in load_user_tasks()]
    conditions = conditions or ["clean", *STYLES]
    per_condition = {}
    cpu_start, wall_start = time.process_time(), time.monotonic()
    for condition in conditions:
        injected = flagged = flagged_injected = total = 0
        for tid in task_ids:
            ref = next(r for r in load_user_tasks() if r.id == tid)
            injections = {} if condition == "clean" else plant(condition, ref.task, get_goal(goal_for(tid)))
            for out in run_ground_truth(ref.task, injections).tool_outputs:
                is_injected = contains_injection(out, injections)
                is_flagged = classifier.score(out) >= threshold
                total += 1
                injected += is_injected
                flagged += is_flagged
                flagged_injected += is_injected and is_flagged
        row = {
            "outputs": total, "injected_outputs": injected, "clean_outputs": total - injected,
            "flagged": flagged, "flagged_injected": flagged_injected, "flagged_clean": flagged - flagged_injected,
            "precision": rate_or_not_computed(flagged_injected, flagged, "no flagged outputs").to_dict(),
            "recall": rate_or_not_computed(flagged_injected, injected, "no injected outputs").to_dict(),
            "false_positive_rate": rate_or_not_computed(flagged - flagged_injected, total - injected, "no clean outputs").to_dict(),
        }
        per_condition[condition] = row
    return {"threshold": threshold, "per_condition": per_condition,
            "cpu_seconds": round(time.process_time() - cpu_start, 2), "wall_seconds": round(time.monotonic() - wall_start, 2),
            "unique_texts_scored": getattr(classifier, "calls", None)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Classifier check on ground-truth Tool_Outputs.")
    parser.add_argument("--out", default="results/m3_filter_calibration.json")
    args = parser.parse_args(argv)
    require_clean()
    stamp = make_stamp("python scripts/filter_calibration.py", 0).to_dict()
    cfg = load_defenses()
    clf = InjectionClassifier(window_tokens=cfg.window_tokens, overlap=cfg.window_overlap_tokens,
                              injection_label=cfg.injection_label)
    result = calibrate(clf, cfg.threshold)
    result["classifier"] = {"repo": cfg.classifier_repo, "revision": cfg.classifier_revision,
                            "window_tokens": cfg.window_tokens, "window_overlap_tokens": cfg.window_overlap_tokens}
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"stamp": stamp, **result}, indent=2) + "\n", encoding="utf-8")
    for cond, row in result["per_condition"].items():
        p, r = row["precision"], row["recall"]
        print(f"{cond:24s} outputs={row['outputs']:3d} injected={row['injected_outputs']:3d} flagged={row['flagged']:3d} "
              f"precision={p.get('point', 'n/a')} recall={r.get('point', 'n/a')}")
    print(f"cpu {result['cpu_seconds']} s; wrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
