"""Report generator: docs/REPORT.md from docs/report_template.md and the results files (task 6.1; Req 7).

Every number enters through a `{{placeholder}}` filled from results/metrics.json (formatted exactly as
results/summary.md formats it), another file under results/, or the configuration; an unfilled
placeholder fails the render. The Report gives aggregate rates and defense effects only.

Usage: python -m aib.report [--template docs/report_template.md] [--out docs/REPORT.md]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from aib.results_table import fmt_rate

PLACEHOLDER = re.compile(r"\{\{\s*([a-z_][a-z0-9_]*)\s*\}\}")
DEFENSE_ORDER = ("none", "delimit", "hierarchy", "filter")
SECTIONS = ("## Question", "## Setup", "## Results", "## Limitations", "## Threats to validity")

LIMITATIONS = (
    "- **One environment.** All {n_user_tasks} user tasks come from AgentDojo's workspace suite, so the results describe "
    "an email, calendar, and file-drive assistant, not agents in general.",
    "- **Small local models.** Their task success can be low even without attacks, which widens intervals and limits "
    "what Defense_Cost can resolve.",
    "- **One model family locally.** Both local models are Qwen-family: Llama-3.1-8B was replaced before the benchmark "
    "because llama.cpp could not parse the turns in which it called two tools (design.md, \"M0 outcome\"), so the local "
    "comparison is across size and generation, not family.",
    "- **The hosted grid is smaller:** it covers {hosted_conditions} of the {n_conditions} conditions because of the "
    "Spend_Cap; the hosted model is an alias that can change over time, and each case records the response `model` field.",
    "- **Different decoding stacks.** The local models are 4-bit GGUF quantizations served by llama.cpp; quantization and "
    "the server's chat template can change behavior relative to the original weights.",
    "- **Exposure-conditioned ASR.** Cases where the agent never read the injected data are excluded from the ASR "
    "denominator; exposure rates are reported beside every ASR.",
    "- **The tool-permission-limit defense is not implemented.** It needs a per-task allowed-tool list that AgentDojo "
    "does not provide.",
    "- **Fixed filter threshold** of {threshold}; precision and recall at other thresholds are not explored.",
    "- **An AgentDojo executor crash counts as errored.** `ToolsExecutor` in `agentdojo` `0.1.35` runs `literal_eval` on "
    "every string argument and catches only `ValueError` and `SyntaxError`, so an argument such as `{{{{'a'}}}}` raises "
    "`TypeError: unhashable type` and ends the case. The runner records such a case as `errored` with that cause, as for "
    "any per-case failure; it is counted in the errored columns, and the executor was not patched mid-run, so every record "
    "of a model comes from one commit.",
    "- **Suite deviation.** The suite includes AgentDojo's `v1.1.2` task fixes, which `get_suite` omits in `0.1.35`, so "
    "numbers are not directly comparable with results produced through that function.",
)


class RenderError(ValueError):
    pass


def render(template: str, context: dict) -> str:
    missing = sorted({m.group(1) for m in PLACEHOLDER.finditer(template) if context.get(m.group(1)) is None})
    if missing:
        raise RenderError(f"unfilled placeholders: {', '.join(missing)}")
    return PLACEHOLDER.sub(lambda m: str(context[m.group(1)]), template)


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines) if rows else "_No rows: the stores held no cases for this table._"


def _order(defense: str) -> int:
    return DEFENSE_ORDER.index(defense) if defense in DEFENSE_ORDER else len(DEFENSE_ORDER)


def models_table(metrics: dict) -> str:
    rows = []
    for model, info in metrics.get("models", {}).items():
        responses = ", ".join(f"`{m}`" for m in info.get("response_models", []) if not m.startswith("stub")) or "-"
        rows.append([f"`{model}`", f"`{info.get('model_version')}`", responses, str(info.get("n_records")),
                     str(info.get("repetitions"))])
    return _table(["Model", "Weights or API model", "Response model field", "Case records", "Repetitions"], rows)


def pooled_table(metrics: dict) -> str:
    costs = {(c["model"], c["defense"], c["condition"]): c for c in metrics.get("defense_cost", [])}
    rows = []
    for p in sorted(metrics.get("pooled", []), key=lambda p: (p["model"], _order(p["defense"]))):
        c = costs.get((p["model"], p["defense"], "pooled"))
        ref = p["defense"] == "none"
        rows.append([f"`{p['model']}`", f"`{p['defense']}`", fmt_rate(p["asr"]), fmt_rate(p["tsr"]),
                     "reference" if ref else fmt_rate(c["defense_cost"] if c else None),
                     "reference" if ref else fmt_rate(c["asr_reduction"] if c else None)])
    return _table(["Model", "Defense", "ASR (95% CI)", "TSR (95% CI)", "Defense_Cost (95% CI)", "ASR reduction (95% CI)"], rows)


def clean_cost_table(metrics: dict) -> str:
    rows = [[f"`{c['model']}`", f"`{c['defense']}`", fmt_rate(c["tsr_none"]), fmt_rate(c["tsr_defense"]),
             fmt_rate(c["defense_cost"])]
            for c in sorted(metrics.get("defense_cost", []), key=lambda c: (c["model"], _order(c["defense"])))
            if c["condition"] == "clean"]
    return _table(["Model", "Defense", "TSR under none (95% CI)", "TSR under the defense (95% CI)",
                   "Defense_Cost (95% CI)"], rows)


def style_table(metrics: dict) -> str:
    rows = [[f"`{c['model']}`", f"`{c['condition']}`", fmt_rate(c["asr"]), fmt_rate(c["exposure"]), str(c["unclassified"])]
            for c in metrics.get("cells", []) if c["defense"] == "none" and c["condition"] != "clean"]
    return _table(["Model", "Injection_Style", "ASR (95% CI)", "Exposure (95% CI)", "Unclassified"], rows)


def filter_table(metrics: dict) -> str:
    rows = [[f"`{f['model']}`", str(f["outputs"]), str(f["injected"]), str(f["flagged"]), fmt_rate(f.get("precision")),
             fmt_rate(f.get("recall")), fmt_rate(f["false_positive_rate"])]
            for f in metrics.get("filter", []) if f["condition"] == "pooled"]
    return _table(["Model", "Screened outputs", "Injected", "Flagged", "Precision (95% CI)", "Recall (95% CI)",
                   "False-positive rate (95% CI)"], rows)


def _computed(entry: dict | None) -> bool:
    return bool(entry) and not entry.get("not_computed")


def findings(metrics: dict) -> str:
    """Factual sentences generated from the metrics; nothing here is typed by hand."""
    out = []
    pooled = metrics.get("pooled", [])
    costs = {(c["model"], c["defense"], c["condition"]): c for c in metrics.get("defense_cost", [])}
    for model in metrics.get("models", {}):
        rows = {p["defense"]: p for p in pooled if p["model"] == model}
        base = rows.get("none")
        if base and _computed(base["asr"]):
            out.append(f"- `{model}`: without a defense it carried out the injected goal in {fmt_rate(base['asr'])} of "
                       f"classified injected cases (pooled over the styles), and it was exposed to the injection in "
                       f"{fmt_rate(base['exposure'])} of injected cases.")
        ranked = [(d, rows[d]) for d in DEFENSE_ORDER if d != "none" and d in rows and _computed(rows[d]["asr"])]
        if ranked:
            best_d, best = min(ranked, key=lambda kv: (kv[1]["asr"]["point"], _order(kv[0])))
            red = (costs.get((model, best_d, "pooled")) or {}).get("asr_reduction")
            if _computed(red):
                verdict = ("its paired interval excludes zero" if red["lower"] > 0 or red["upper"] < 0
                           else "its paired interval includes zero")
                out.append(f"- `{model}`: the lowest pooled ASR was under `{best_d}` ({fmt_rate(best['asr'])}); the ASR "
                           f"reduction versus `none` was {fmt_rate(red)}, and {verdict}.")
        for d in DEFENSE_ORDER[1:]:
            c = costs.get((model, d, "clean"))
            if c and _computed(c["defense_cost"]):
                dc = c["defense_cost"]
                verdict = ("the paired interval excludes zero" if dc["lower"] > 0 or dc["upper"] < 0
                           else "the paired interval includes zero")
                out.append(f"- `{model}` under `{d}`: benign Defense_Cost {fmt_rate(dc)} on the clean condition; {verdict}.")
    return "\n".join(out) if out else "- No Cell had enough classified cases for an observation."


def hosted_statement(facts: dict, hosted_conditions: list[str], n_conditions: int) -> str:
    h = facts.get("hosted")
    scope = (f"The hosted model ran all {facts.get('n_defenses')} defenses on {len(hosted_conditions)} of the "
             f"{n_conditions} conditions ({', '.join(f'`{c}`' for c in hosted_conditions)}) to stay within the Spend_Cap, "
             f"in Repetition-major order so that a stop leaves every Cell with the same number of complete Repetitions.")
    if not h:
        return scope + " No hosted run summary was present when this report was generated."
    stop = ("It stopped at the Spend_Cap" if h["stopped_at_cap"] else "It did not reach the Spend_Cap")
    return (scope + f" {h['completed_cases']} of {h['planned_cases']} planned cases completed; recorded spend was "
            f"${h['spend_usd']} of the ${h['spend_cap_usd']} cap (an upper bound on the bill). {stop}.")


def build_context(root: Path, stamp: dict) -> dict:
    from aib.config import load_all

    metrics = json.loads((root / "results" / "metrics.json").read_text(encoding="utf-8"))
    catalog = json.loads((root / "results" / "m1_catalog.json").read_text(encoding="utf-8"))
    calib_path = root / "results" / "m3_filter_calibration.json"
    calib = json.loads(calib_path.read_text(encoding="utf-8")) if calib_path.exists() else {}
    cfg = load_all(root / "config")
    facts = metrics.get("run_facts") or {}
    local = next((i["sampling"] for i in metrics.get("models", {}).values() if "top_k" in (i.get("sampling") or {})), {})
    clean_cal = (calib.get("per_condition") or {}).get("clean") or {}
    ms = metrics.get("stamp", {})
    ctx = {
        "report_command": stamp["command"], "report_seed": stamp["seed"], "report_commit": stamp["commit"],
        "metrics_command": ms.get("command"), "metrics_seed": ms.get("seed"), "metrics_commit": ms.get("commit"),
        "n_user_tasks": facts.get("n_user_tasks"), "n_tools": catalog.get("tool_count"), "n_styles": facts.get("n_styles"),
        "n_goals": facts.get("n_goals"), "n_defenses": facts.get("n_defenses"), "repetitions": facts.get("repetitions"),
        "temperature": facts.get("temperature"), "top_p": facts.get("top_p"), "max_iters": facts.get("max_iters"),
        "max_tokens": facts.get("max_tokens"), "top_k": local.get("top_k"), "min_p": local.get("min_p"),
        "resamples": (metrics.get("settings") or {}).get("resamples"), "threshold": calib.get("threshold"),
        "calibration_clean_flagged": clean_cal.get("flagged"), "calibration_clean_outputs": clean_cal.get("outputs"),
        "models_table": models_table(metrics), "pooled_table": pooled_table(metrics),
        "clean_cost_table": clean_cost_table(metrics), "style_table": style_table(metrics),
        "filter_table": filter_table(metrics), "findings": findings(metrics),
        "hosted_statement": hosted_statement(facts, cfg.hosted.conditions, facts.get("n_conditions") or len(cfg.experiment.conditions)),
    }
    ctx["limitations"] = "\n".join(line.format(
        n_user_tasks=ctx["n_user_tasks"], hosted_conditions=len(cfg.hosted.conditions),
        n_conditions=facts.get("n_conditions"), threshold=ctx["threshold"]) for line in LIMITATIONS)
    return ctx


def main(argv: list[str] | None = None) -> int:
    from aib.repro import REPO_ROOT, make_stamp, require_clean

    parser = argparse.ArgumentParser(description="Render docs/REPORT.md from the template and the results files.")
    parser.add_argument("--template", default="docs/report_template.md")
    parser.add_argument("--out", default="docs/REPORT.md")
    args = parser.parse_args(argv)
    require_clean()
    seed = json.loads((REPO_ROOT / "results" / "metrics.json").read_text(encoding="utf-8")).get("settings", {}).get("seed", 0)
    stamp = make_stamp(f"python -m aib.report --template {args.template} --out {args.out}", seed).to_dict()
    text = render((REPO_ROOT / args.template).read_text(encoding="utf-8"), build_context(REPO_ROOT, stamp))
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
