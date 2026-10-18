"""README generator: README.md from docs/README.template.md and the results files (task 6.6; Req 9).

The Results_Table section is copied from results/summary.md, so every rate keeps its interval and every
number traces to a generated file.

Usage: python -m aib.docs_build [--template docs/README.template.md] [--out README.md]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from aib.report import LIMITATIONS, render

_STAMP = re.compile(r"^- (Table|Metrics): command `(?P<command>[^`]*)`, seed (?P<seed>\S+), commit `(?P<commit>[0-9a-f]{40})`\.$",
                    re.M)


def summary_section(summary: str, heading: str) -> str:
    """The body of one `## heading` section of results/summary.md (without the heading)."""
    m = re.search(rf"^## {re.escape(heading)}\n(.*?)(?=^## |\Z)", summary, re.S | re.M)
    if not m:
        raise ValueError(f"results/summary.md has no section {heading!r}")
    return m.group(1).strip()


def summary_stamps(summary: str) -> dict[str, dict]:
    return {m.group(1).lower(): {"command": m.group("command"), "seed": m.group("seed"), "commit": m.group("commit")}
            for m in _STAMP.finditer(summary)}


def build_context(root: Path, stamp: dict) -> dict:
    from aib.config import load_all

    summary = (root / "results" / "summary.md").read_text(encoding="utf-8")
    metrics = json.loads((root / "results" / "metrics.json").read_text(encoding="utf-8"))
    calib_path = root / "results" / "m3_filter_calibration.json"
    calib = json.loads(calib_path.read_text(encoding="utf-8")) if calib_path.exists() else {}
    cfg = load_all(root / "config")
    facts = metrics.get("run_facts") or {}
    stamps = summary_stamps(summary)
    if "table" not in stamps or "metrics" not in stamps:
        raise ValueError("results/summary.md is missing its stamp lines")
    limitations = "\n".join(line.format(n_user_tasks=facts.get("n_user_tasks"), hosted_conditions=len(cfg.hosted.conditions),
                                        n_conditions=facts.get("n_conditions"), threshold=calib.get("threshold"))
                            for line in LIMITATIONS)
    return {
        "readme_command": stamp["command"], "readme_seed": stamp["seed"], "readme_commit": stamp["commit"],
        "summary_command": stamps["table"]["command"], "summary_seed": stamps["table"]["seed"],
        "summary_commit": stamps["table"]["commit"],
        "metrics_command": stamps["metrics"]["command"], "metrics_seed": stamps["metrics"]["seed"],
        "metrics_commit": stamps["metrics"]["commit"],
        "run_facts": summary_section(summary, "Run facts"),
        "pooled_table": summary_section(summary, "Pooled over the Injection_Styles"),
        "limitations": limitations,
    }


def main(argv: list[str] | None = None) -> int:
    from aib.repro import REPO_ROOT, make_stamp, require_clean

    parser = argparse.ArgumentParser(description="Render README.md from the template and the results files.")
    parser.add_argument("--template", default="docs/README.template.md")
    parser.add_argument("--out", default="README.md")
    args = parser.parse_args(argv)
    require_clean()
    seed = json.loads((REPO_ROOT / "results" / "metrics.json").read_text(encoding="utf-8")).get("settings", {}).get("seed", 0)
    stamp = make_stamp(f"python -m aib.docs_build --template {args.template} --out {args.out}", seed).to_dict()
    text = render((REPO_ROOT / args.template).read_text(encoding="utf-8"), build_context(REPO_ROOT, stamp))
    (REPO_ROOT / args.out).write_text(text, encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
