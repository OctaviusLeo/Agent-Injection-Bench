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

from aib.results_table import fmt_kappa, fmt_rate, refusal_text

PLACEHOLDER = re.compile(r"\{\{\s*([a-z_][a-z0-9_]*)\s*\}\}")
DEFENSE_ORDER = ("none", "delimit", "hierarchy", "filter")
SECTIONS = ("## Question", "## Setup", "## Results", "## Limitations", "## Threats to validity")

LIMITATIONS = (
    "- **One environment.** All {n_user_tasks} user tasks come from AgentDojo's workspace suite, so the results describe "
    "an email, calendar, and file-drive assistant, not agents in general.",
    "- **Small local models.** Their task success can be low even without attacks, which widens intervals and limits "
    "what the defense-cost comparison can resolve.",
    "- **One model family locally.** Both local models are Qwen-family: Llama-3.1-8B was replaced before the benchmark "
    "because llama.cpp could not parse the turns in which it called two tools (the model checks are recorded in "
    "`.kiro/specs/agent-injection-bench/design.md`), so the local "
    "comparison is across size and generation, not family.",
    "- **The hosted grid is smaller:** it covers {hosted_conditions} of the {n_conditions} conditions because of the "
    "spend cap. `{hosted_model}` is OpenAI's API model ID, an alias the provider can repoint; the response `model` field "
    "recorded on every case echoed the alias, so no dated snapshot was recorded.",
    "- **Floor effect.** Undefended attack success was at most {max_none_asr} pooled over a model's styles, so no defense "
    "could lower it by more than that; the ASR-reduction intervals are narrow mainly because there was little to reduce.",
    "- **Different decoding stacks.** The local models are 4-bit GGUF quantizations served by llama.cpp; quantization and "
    "the server's chat template can change behavior relative to the original weights.",
    "- **Exposure-conditioned ASR.** Cases where the agent never read the injected data are excluded from the ASR "
    "denominator; exposure rates are reported beside every ASR.",
    "- **The tool-permission-limit defense is not implemented.** It needs a per-task allowed-tool list that AgentDojo "
    "does not provide.",
    "- **Fixed filter threshold** of {threshold}; precision and recall at other thresholds are not explored.",
    "- **Zero-success intervals.** A rate with no successes reports the exact one-sided 95% upper bound, which treats "
    "cases as independent and so can be narrower than a task-clustered bound; a paired difference between two "
    "zero-success rates still shows a zero-width bootstrap interval.",
    "- **Goal confounding.** Each user task is always paired with the same injected goal (user task k with goal k mod 6), "
    "and the bootstrap resamples tasks only, so task and goal effects cannot be separated and the intervals do not cover "
    "other pairings.",
    "- **Predicate exceptions.** An AgentDojo injection predicate that raises is scored as not carried out, and the "
    "committed records do not say whether that happened; runs made after this was found record the exception in "
    "`security_check_error`.",
    "- **An AgentDojo executor crash counts as errored.** `ToolsExecutor` in `agentdojo` `0.1.35` runs `literal_eval` on "
    "every string argument and catches only `ValueError` and `SyntaxError`, so a nested set literal (a set written inside "
    "a set) raises `TypeError: unhashable type` and ends the case. The runner records such a case as `errored` with that cause, as for "
    "any per-case failure; it is counted in the errored columns, and the executor was not patched mid-run, so every record "
    "of a model comes from one commit.",
    "- **Suite deviation.** The suite includes AgentDojo's `v1.1.2` task fixes, which `get_suite` omits in `0.1.35`, so "
    "numbers are not directly comparable with results produced through that function.",
)


def limitation_lines(metrics: dict, cfg, calib: dict) -> str:
    """The Known limitations list shared by the Report and the README, filled from the results and the config."""
    facts = metrics.get("run_facts") or {}
    base = [p["asr"] for p in metrics.get("pooled", []) if p["defense"] == "none" and _computed(p["asr"])]
    max_none_asr = f"{max(a['point'] for a in base):.3f}" if base else "not computed"
    lines = [line.format(n_user_tasks=facts.get("n_user_tasks"), hosted_conditions=len(cfg.hosted.conditions),
                         n_conditions=facts.get("n_conditions"), threshold=calib.get("threshold"),
                         hosted_model=cfg.hosted.model, max_none_asr=max_none_asr) for line in LIMITATIONS]
    at = next((i + 1 for i, line in enumerate(lines) if line.startswith("- **The hosted grid is smaller")), len(lines))
    lines[at:at] = comparability_lines(metrics, cfg.hosted.model)
    return "\n".join(lines)


def default_sampling_models(metrics: dict) -> dict[str, dict]:
    """Models whose records sent no temperature (recorded as None): the Claude target, which accepts only its default."""
    return {m: i["sampling"] for m, i in metrics.get("models", {}).items()
            if "temperature" in (i.get("sampling") or {}) and i["sampling"]["temperature"] is None}


def comparability_lines(metrics: dict, hosted_model: str) -> list[str]:
    """The comparability limit between a default-sampling hosted model and the OpenAI model (claude-provider design.md)."""
    other = (metrics.get("models", {}).get(hosted_model) or {}).get("sampling")
    if not other:
        return []
    out = []
    for model, s in default_sampling_models(metrics).items():
        ids = [m for m in metrics["models"][model].get("response_models", []) if not m.startswith("stub")]
        recorded = ("echoed the requested ID, so no dated snapshot was recorded" if ids == [model]
                    else "recorded " + ", ".join(f"`{m}`" for m in ids) if ids else "was not recorded")
        out.append(f"- **The two hosted models are not a controlled comparison.** `{model}` sent no sampling settings and no "
                   f"seed, so it used the model's default sampling, and it thought adaptively at effort `{s.get('effort')}`; "
                   f"`{hosted_model}` ran at temperature {other.get('temperature')} and top-p {other.get('top_p')} with a "
                   f"per-request seed and reasoning effort `{other.get('reasoning_effort')}`. A difference between the two mixes "
                   f"the model with its decoding settings. The response `model` field on `{model}`'s cases {recorded}.")
    return out


def sampling_statement(metrics: dict, facts: dict) -> str:
    """The sampling settings per model: the experiment's for every model that took them, the default for any that did not."""
    local = next((i["sampling"] for i in metrics.get("models", {}).values() if "top_k" in (i.get("sampling") or {})), {})
    default = default_sampling_models(metrics)
    who = "Every model" if not default else "Every model but " + ", ".join(f"`{m}`" for m in default)
    text = (f"{who} ran with temperature {facts.get('temperature')} and top-p {facts.get('top_p')}, with at most "
            f"{facts.get('max_tokens')} output tokens per call; the local models also used top-k {local.get('top_k')} and "
            f"min-p {local.get('min_p')}, so only temperature and top-p shape their sampling. Each "
            f"{'request' if not default else 'of their requests'} carried a seed derived from the case seed.")
    for model, s in default.items():
        text += (f" `{model}` sent no sampling settings and no seed ({s.get('note')}); it thought adaptively at effort "
                 f"`{s.get('effort')}`, with at most {s.get('max_tokens')} output tokens per call, thinking included.")
    return text


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


def hosted_statement(facts: dict, hosted_conditions: list[str], n_conditions: int, hosted_model: str = "",
                     claude_conditions: list[str] | None = None) -> str:
    h = facts.get("hosted")
    name = f"The OpenAI model `{hosted_model}`" if hosted_model else "The hosted model"
    scope = (f"{name} ran all {facts.get('n_defenses')} defenses on {len(hosted_conditions)} of the "
             f"{n_conditions} conditions ({', '.join(f'`{c}`' for c in hosted_conditions)}) to stay within the Spend_Cap, "
             f"in Repetition-major order so that a stop leaves every Cell with the same number of complete Repetitions.")
    if not h:
        text = scope + " No hosted run summary was present when this report was generated."
    else:
        stop = ("It stopped at the Spend_Cap" if h["stopped_at_cap"] else "It did not reach the Spend_Cap")
        text = (scope + f" {h['completed_cases']} of {h['planned_cases']} planned cases completed; recorded spend was "
                f"${h['spend_usd']} of the ${h['spend_cap_usd']} cap (an upper bound on the bill). {stop}.")
    c = facts.get("claude")
    if not c:
        return text
    if not c["summary"]:
        return text + f" No run summary for the Claude target `{c['model']}` was present when this report was generated."
    grid =("the same conditions in the same order" if claude_conditions in (None, hosted_conditions) else
            f"{len(claude_conditions)} of the conditions ({', '.join(f'`{x}`' for x in claude_conditions)}) in the same order")
    runs = f" in {c['runs']} run{'' if c['runs'] == 1 else 's'}" if c["runs"] else ""
    stop = "Its last run stopped at the Run_Cap" if c["stopped_at_cap"] else "Its last run ended below the Run_Cap"
    return (text + f" The Claude target `{c['model']}` ran {grid}, under a cap of ${c['run_cap_usd']} per run: "
            f"{c['completed_cases']} of {c['planned_cases']} planned cases completed{runs}; recorded spend was "
            f"${c['spend_usd']} (an upper bound on the bill: uncached prompt tokens are priced at the cache-write rate). {stop}.")


def judge_section(metrics: dict) -> str:
    """The second-opinion judge's agreement with AgentDojo's check, copied as results/summary.md formats it."""
    rows = metrics.get("judge") or []
    if not rows:
        return "No second-opinion verdicts were present when this report was generated."
    judges = ", ".join(f"`{m}`" for m in sorted({m for r in rows for m in r.get("judge_models", [])}))
    table = _table(["Model under test", "Classified cases", "Verdicts", "Parsed", "Refused", "Unparsed", "Judge yes, rule yes",
                    "Judge yes, rule no", "Judge no, rule yes", "Judge no, rule no", "Agreement (95% CI)", "Cohen's kappa"],
                   [[f"`{r['model']}`", str(r["classified"]), str(r["judged"]), str(r["parsed"]), str(r["refused"]),
                     str(r["unparsed"]),
                     str(r["judge_yes_rule_yes"]), str(r["judge_yes_rule_no"]), str(r["judge_no_rule_yes"]),
                     str(r["judge_no_rule_no"]), fmt_rate(r["agreement"]), fmt_kappa(r["cohens_kappa"])] for r in rows])
    notes = [f"- `{r['model']}` refused verdicts: {refusal_text(r, '`')}." for r in rows if r.get("refused")]
    notes += [f"- `{r['model']}` verdicts: command `{r['stamp']['command']}`, seed {r['stamp']['seed']}, commit "
              f"`{r['stamp']['commit']}`." for r in rows if r.get("stamp")]
    stamps = "\n".join(notes)
    models = ", ".join(f"`{r['model']}`" for r in rows)
    cats = sorted({k for r in rows for k in r.get("refusal_categories", {})})
    goals = sorted({g for r in rows for g in r.get("refused_by_goal", {})})
    refusals = (f" Every refused verdict had stop category {', '.join(f'`{k}`' for k in cats)} and came from injected goal "
                f"{' or '.join(f'`{g}`' for g in goals)}; refused verdicts are left out of the agreement, so those goals rest "
                "on fewer verdicts (counts below the table)." if goals else "")
    return (f"{judges} read classified attack cases of {models} (the injected goal, "
            "the user's task, and the transcript, which it was told is data that may contain instructions addressed to it) "
            "and said whether the agent's tool calls carried out the injected goal; the table gives how many of each model's "
            "classified cases it judged. Its verdicts change no rate in this "
            "report; they measure how often an LLM judge agrees with AgentDojo's rule-based check. Agreement is over the "
            "parsed verdicts, with a 95% task-clustered interval. Cohen's kappa corrects it for the agreement expected by "
            "chance, which is high when nearly every case is a rule-negative; it is undefined when both always gave the "
            f"same answer.{refusals}\n\n{table}\n\n{stamps}\n\nEach verdict line in `results/judge/<model>.jsonl` keeps the "
            "stamp of the run that wrote it.")


def external_validity(metrics: dict, cfg) -> str:
    """The models the results describe, counted from the stores."""
    words = {1: "one", 2: "two", 3: "three", 4: "four"}
    local = sum(1 for m in metrics.get("models", {}) if cfg.models.models.get(m) and cfg.models.models[m].provider == "llamacpp")
    hosted = len(metrics.get("models", {})) - local
    return (f"{words.get(local, local)} small quantized local model{'' if local == 1 else 's'}, and "
            f"{words.get(hosted, hosted)} hosted model ID{'' if hosted == 1 else 's'} at one point in time")


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
        "hosted_statement": hosted_statement(facts, cfg.hosted.conditions, facts.get("n_conditions") or len(cfg.experiment.conditions),
                                             cfg.hosted.model, cfg.claude.conditions),
        "sampling_statement": sampling_statement(metrics, facts), "judge_section": judge_section(metrics),
        "external_validity": external_validity(metrics, cfg),
    }
    ctx["limitations"] = limitation_lines(metrics, cfg, calib)
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
