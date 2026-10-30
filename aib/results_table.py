"""Results_Table: results/summary.md generated from results/metrics.json (task 5.3; Req 6, 7.2).

Every rate is shown with its 95% interval; a rate with no successes shows its exact one-sided upper
bound, marked with a dagger, and a rate with only successes its exact one-sided lower bound, marked with a
double dagger; a Cell whose rate cannot be computed shows "not computed" and the cause.
The file is regenerated, never edited by hand.

Usage: python -m aib.results_table [--metrics results/metrics.json] [--out results/summary.md]
"""

from __future__ import annotations

import argparse
import json
import sys

DEFENSE_ORDER = ("none", "delimit", "hierarchy", "filter")
ZERO_COUNT_MARK = "†"
ZERO_COUNT_NOTE = (f"{ZERO_COUNT_MARK} No successes: the interval is the exact one-sided 95% upper bound (Clopper-Pearson, "
                   "cases treated as independent) instead of the bootstrap interval, which collapses to a point at zero.")
ALL_SUCCESS_MARK = "‡"
ALL_SUCCESS_NOTE = (f"{ALL_SUCCESS_MARK} Only successes: the interval is the exact one-sided 95% lower bound (Clopper-Pearson, "
                    "cases treated as independent) instead of the bootstrap interval, which collapses to a point at one.")


def fmt_rate(entry: dict | None) -> str:
    if entry is None:
        return "-"
    if entry.get("not_computed"):
        return f"not computed ({entry['cause']})"
    method = entry.get("method", "")
    mark = ZERO_COUNT_MARK if method.startswith("exact-upper") else ALL_SUCCESS_MARK if method.startswith("exact-lower") else ""
    return f"{entry['point']:.3f} [{entry['lower']:.3f}, {entry['upper']:.3f}]{mark}"


def fmt_sampling(sampling: dict) -> str:
    """A record's sampling settings; a temperature or top-p recorded as None was not sent, so the model's default applied."""
    keys = ("temperature", "top_p", "top_k", "min_p", "max_tokens", "max_completion_tokens", "reasoning_effort", "effort",
            "thinking")
    return ", ".join(f"{k}={'model default' if sampling[k] is None else sampling[k]}" for k in keys
                     if k in sampling and (sampling[k] is not None or k in ("temperature", "top_p")))


def fmt_kappa(kappa: float | None, parsed: int | None = None) -> str:
    if kappa is None:
        return "not computed (no parsed verdicts)" if parsed == 0 else "undefined (both always gave the same answer)"
    return f"{kappa:.3f}"


def run_fact_lines(f: dict) -> list[str]:
    c = f.get("claude")
    lines = [
        f"- User tasks: {f.get('n_user_tasks')}; conditions: {f.get('n_conditions')} (clean plus {f.get('n_styles')} "
        f"injection styles); defenses: {f.get('n_defenses')}; injected goals: {f.get('n_goals')}; "
        f"models: {f.get('n_models')}; case records: {f.get('n_records')}.",
        f"- Repetitions per model, defense and condition: {f.get('repetitions')}; temperature {f.get('temperature')}; top-p {f.get('top_p')}; "
        f"loop cap {f.get('max_iters')} model calls; output cap {f.get('max_tokens')} tokens per call; base seed {f.get('base_seed')}"
        + (f"; the temperature, top-p, and output cap apply to every model but {c['model']}." if c else "."),
    ]
    h = f.get("hosted")
    if h:
        lines.append(f"- OpenAI hosted run: {h['completed_cases']} of {h['planned_cases']} planned cases completed; spend "
                     f"${h['spend_usd']} of the ${h['spend_cap_usd']} spend cap; stopped at the cap: "
                     f"{'yes' if h['stopped_at_cap'] else 'no'}.")
    else:
        lines.append("- OpenAI hosted run: no hosted run summary (results/hosted_run.json) was present.")
    if c and not c["summary"]:
        lines.append(f"- Claude run ({c['model']}): no Claude run summary (results/claude_run.json) was present.")
    elif c:
        runs = (f"over {c['runs']} runs, each under" if (c["runs"] or 0) > 1 else
                "in 1 run, under" if c["runs"] else "in its last run, under")
        lines.append(f"- Claude run ({c['model']}): {c['completed_cases']} of {c['planned_cases']} planned cases completed; "
                     f"recorded spend ${c['spend_usd']} {runs} the ${c['run_cap_usd']} per-run cap; the last run stopped at "
                     f"its cap: {'yes' if c['stopped_at_cap'] else 'no'}. No sampling settings or seed were sent (the "
                     f"model's default sampling); adaptive thinking at effort {c['effort']}; output cap {c['max_tokens']} "
                     "tokens per call, thinking included.")
    return lines


def diagnostics_lines(d: dict | None) -> list[str]:
    if not d:
        return []
    return [f"- Bootstrap diagnostics: {d['rates_zero_count']} of {d['rates']} rates had no successes and "
            f"{d['rates_all_success']} had only successes; the percentile "
            f"interval had to be clamped to contain the point for {d['rates_clamped']} rates and {d['differences_clamped']} "
            f"of {d['differences']} paired differences; paired resamples dropped for an empty side: "
            f"{d['dropped_resamples']} in total, at most {d['max_dropped_resamples']} in one difference."]


def _cost_lookup(metrics: dict) -> dict:
    return {(c["model"], c["defense"], c["condition"]): c for c in metrics.get("defense_cost", [])}


def render_summary(metrics: dict, stamp: dict) -> str:
    ms = metrics.get("stamp", {})
    st = metrics.get("settings", {})
    costs = _cost_lookup(metrics)
    lines = [
        "# Results_Table",
        "",
        "Generated by `python -m aib.results_table` from `results/metrics.json`; do not edit by hand.",
        "",
        f"- Table: command `{stamp['command']}`, seed {stamp['seed']}, commit `{stamp['commit']}`.",
        f"- Metrics: command `{ms.get('command')}`, seed {ms.get('seed')}, commit `{ms.get('commit')}`.",
        f"- Intervals: 95% {st.get('interval_method')} for rates; {st.get('difference_method')} for defense cost and ASR "
        "reduction (paired on the same resampled tasks).",
        f"- {ZERO_COUNT_NOTE}",
        f"- {ALL_SUCCESS_NOTE}",
        *diagnostics_lines(metrics.get("diagnostics")),
        "- ASR = carried-out / classified injected cases (unclassified and errored excluded). TSR = task successes / "
        "non-errored cases. Defense cost = TSR(none) - TSR(defense), same model and condition.",
        "",
        "## Run facts",
        "",
        *run_fact_lines(metrics.get("run_facts") or {}),
        "",
        "## Models",
        "",
        "| Model | Version | Records | Repetitions | Sampling | Statuses |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for model, info in metrics.get("models", {}).items():
        statuses = ", ".join(f"{k} {v}" for k, v in info.get("status_counts", {}).items())
        lines.append(f"| {model} | {info.get('model_version')} | {info.get('n_records')} | {info.get('repetitions')} | "
                     f"{fmt_sampling(info.get('sampling', {}))} | {statuses} |")
    lines += ["", "## Cells", "",
              "| Model | Defense | Condition | ASR (95% CI) | TSR (95% CI) | Defense cost vs none (95% CI) | "
              "Exposure (95% CI) | Unclassified | Errored | Reps | Cases |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for c in metrics.get("cells", []):
        cost = costs.get((c["model"], c["defense"], c["condition"]))
        cost_text = "reference" if c["defense"] == "none" else fmt_rate(cost["defense_cost"] if cost else None)
        lines.append(f"| {c['model']} | {c['defense']} | {c['condition']} | {fmt_rate(c['asr'])} | {fmt_rate(c['tsr'])} | "
                     f"{cost_text} | {fmt_rate(c['exposure'])} | {c['unclassified']} | {c['errored']} | "
                     f"{c['repetitions']} | {c['n_cases']} |")
    lines += ["", "## Pooled over the injection styles", "",
              "| Model | Defense | Styles | ASR (95% CI) | TSR (95% CI) | Defense cost vs none (95% CI) | "
              "ASR reduction vs none (95% CI) | Exposure (95% CI) | Unclassified | Errored |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for p in metrics.get("pooled", []):
        cost = costs.get((p["model"], p["defense"], "pooled"))
        ref = p["defense"] == "none"
        lines.append(f"| {p['model']} | {p['defense']} | {len(p['conditions'])} | {fmt_rate(p['asr'])} | {fmt_rate(p['tsr'])} | "
                     f"{'reference' if ref else fmt_rate(cost['defense_cost'] if cost else None)} | "
                     f"{'reference' if ref else fmt_rate(cost['asr_reduction'] if cost else None)} | "
                     f"{fmt_rate(p['exposure'])} | {p['unclassified']} | {p['errored']} |")
    shared = metrics.get("shared_pooled", [])
    if shared:
        lines += ["", "## Pooled over the styles every model ran", "",
                  f"Styles: {', '.join(shared[0]['conditions'])}. Carried out / classified injected cases, and task successes "
                  "/ non-errored cases, with the same 95% intervals.", "",
                  "| Defense | Model | ASR (95% CI) | Carried out / classified | TSR (95% CI) |",
                  "| --- | --- | --- | --- | --- |"]
        for s in shared:
            asr = s["asr"] or {}
            counts = f"{asr['successes']} / {asr['n']}" if "successes" in asr else "-"
            lines.append(f"| {s['defense']} | {s['model']} | {fmt_rate(s['asr'])} | {counts} | {fmt_rate(s['tsr'])} |")
    lines += ["", "## Filter classifier (screened tool outputs in filter Cells)", "",
              "| Model | Condition | Outputs | Injected | Flagged | Precision (95% CI) | Recall (95% CI) | "
              "False-positive rate on clean outputs (95% CI) |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for f in metrics.get("filter", []):
        lines.append(f"| {f['model']} | {f['condition']} | {f['outputs']} | {f['injected']} | {f['flagged']} | "
                     f"{fmt_rate(f.get('precision'))} | {fmt_rate(f.get('recall'))} | {fmt_rate(f['false_positive_rate'])} |")
    lines += judge_lines(metrics.get("judge") or [])
    return "\n".join(lines) + "\n"


def judge_lines(rows: list[dict]) -> list[str]:
    """The second-opinion judge's agreement with AgentDojo's check, per judged model (claude-provider Req C3)."""
    if not rows:
        return []
    judges = ", ".join(sorted({m for r in rows for m in r.get("judge_models", [])}))
    lines = ["", "## Second-opinion judge", "",
             f"Judge: {judges}. Agreement = verdicts that match AgentDojo's check / parsed verdicts, with the same 95% "
             "interval as every rate; refused and unparsed verdicts are left out. The judge changes no rate above.", "",
             "| Model | Classified cases | Verdicts | Parsed | Refused | Unparsed | Judge yes, rule yes | Judge yes, rule no | "
             "Judge no, rule yes | Judge no, rule no | Agreement (95% CI) | Cohen's kappa |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['classified']} | {r['judged']} | {r['parsed']} | {r['refused']} | {r['unparsed']} | "
                     f"{r['judge_yes_rule_yes']} | {r['judge_yes_rule_no']} | {r['judge_no_rule_yes']} | "
                     f"{r['judge_no_rule_no']} | {fmt_rate(r['agreement'])} | {fmt_kappa(r['cohens_kappa'], r['parsed'])} |")
    notes = [f"- Refused verdicts for {r['model']}: {refusal_text(r)}." for r in rows if r.get("refused")]
    notes += [f"- Verdicts for {r['model']}: command `{r['stamp']['command']}`, seed {r['stamp']['seed']}, commit "
              f"`{r['stamp']['commit']}`{commits_text(r)}." for r in rows if r.get("stamp")]
    return lines + ([""] + notes if notes else [])


def commits_text(row: dict) -> str:
    """When a model's verdict lines come from more than one commit, how many came from each (every line keeps its stamp)."""
    commits = row.get("verdict_commits") or {}
    return ("; verdict lines by commit: " + ", ".join(f"`{c}` {n}" for c, n in commits.items())) if len(commits) > 1 else ""


def refusal_text(row: dict, tick: str = "") -> str:
    """A judge row's refusals by stop category and by injected goal (refused of judged), for the table and the Report."""
    cats = ", ".join(f"{tick}{k}{tick} {n}" for k, n in row.get("refusal_categories", {}).items())
    goals = ", ".join(f"{tick}{g}{tick} {c['refused']} of {c['judged']}" for g, c in row.get("refused_by_goal", {}).items())
    return f"stop category {cats}; by injected goal, refused of judged: {goals}"


def main(argv: list[str] | None = None) -> int:
    from aib.repro import REPO_ROOT, make_stamp, require_clean

    parser = argparse.ArgumentParser(description="Write results/summary.md from results/metrics.json.")
    parser.add_argument("--metrics", default="results/metrics.json")
    parser.add_argument("--out", default="results/summary.md")
    args = parser.parse_args(argv)
    require_clean()
    metrics = json.loads((REPO_ROOT / args.metrics).read_text(encoding="utf-8"))
    stamp = make_stamp(f"python -m aib.results_table --metrics {args.metrics} --out {args.out}",
                       metrics.get("settings", {}).get("seed", 0)).to_dict()
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_summary(metrics, stamp), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
