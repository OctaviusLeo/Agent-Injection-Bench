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

from aib.report import LIMITATIONS, _computed, render
from aib.results_table import fmt_rate

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


def _excludes_zero(entry: dict) -> bool:
    return entry["lower"] > 0 or entry["upper"] < 0


def headline(metrics: dict) -> str:
    """The README's opening findings, from the pooled paired comparisons; numbers are formatted as summary.md formats them."""
    out = []
    base = [p for p in metrics.get("pooled", []) if p["defense"] == "none" and _computed(p["asr"])]
    if base:
        out.append("- **Undefended attack success:** " + ", ".join(f"`{p['model']}` {fmt_rate(p['asr'])}" for p in base) + ".")
    pooled = [c for c in metrics.get("defense_cost", []) if c["condition"] == "pooled"]
    filt = [c for c in pooled if c["defense"] == "filter" and _computed(c["defense_cost"]) and _computed(c["asr_reduction"])]
    if filt:
        lo = min(filt, key=lambda c: c["defense_cost"]["point"])
        hi = max(filt, key=lambda c: c["defense_cost"]["point"])
        best = max(filt, key=lambda c: c["asr_reduction"]["point"])
        out.append(f"- **DeBERTa filter:** it lowered task success by {fmt_rate(lo['defense_cost'])} (`{lo['model']}`) to "
                   f"{fmt_rate(hi['defense_cost'])} (`{hi['model']}`); its largest drop in attack success was "
                   f"{fmt_rate(best['asr_reduction'])} (`{best['model']}`).")
    prompts = [c for c in pooled if c["defense"] in ("delimit", "hierarchy")
               and _computed(c["defense_cost"]) and _computed(c["asr_reduction"])]
    if prompts:
        cost_moved = [c for c in prompts if _excludes_zero(c["defense_cost"])]
        asr_moved = [c for c in prompts if _excludes_zero(c["asr_reduction"])]
        line = (f"- **Delimiter and hierarchy prompts:** the paired interval for the change in task success includes zero in "
                f"{len(prompts) - len(cost_moved)} of {len(prompts)} model-defense pairs, and for the change in attack success "
                f"in {len(prompts) - len(asr_moved)} of {len(prompts)}")
        moved = [f"`{c['model']}` under `{c['defense']}`, task-success cost {fmt_rate(c['defense_cost'])}" for c in cost_moved]
        moved += [f"`{c['model']}` under `{c['defense']}`, attack-success reduction {fmt_rate(c['asr_reduction'])}"
                  + (" (attack success rose)" if c["asr_reduction"]["upper"] < 0 else "") for c in asr_moved]
        out.append(line + ("; the exceptions: " + "; ".join(moved) if moved else "") + ".")
    return "\n".join(out) if out else "- No pooled comparison had enough classified cases for a finding."


def hosted_dates(ledger: Path, model: str) -> str:
    """The UTC date range of the hosted benchmark requests, from the spend ledger."""
    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()] if ledger.exists() else []
    days = sorted({r["ts"][:10] for r in rows if r.get("purpose") == f"benchmark:{model}"})
    if not days:
        return "on the dates recorded in `results/hosted_ledger.jsonl`"
    return f"on {days[0]} (UTC)" if days[0] == days[-1] else f"from {days[0]} to {days[-1]} (UTC)"


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
        "pooled_table": summary_section(summary, "Pooled over the injection styles"),
        "limitations": limitations,
        "headline": headline(metrics),
        "hosted_model": cfg.hosted.model,
        "hosted_styles": len([c for c in cfg.hosted.conditions if c != "clean"]),
        "hosted_dates": hosted_dates(root / "results" / "hosted_ledger.jsonl", cfg.hosted.model),
        **{k: facts.get(k) for k in ("n_user_tasks", "n_styles", "n_defenses", "n_models", "n_records")},
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
