"""Pilot summary: results/pilot.json from the pilot store (task 4.5; Req 5.1, 5.6).

Per case: duration, status, error cause, token counts, exposure, and context overflow. Projection of
each Local_Model's full grid: the 4B model's mean measured seconds per case, divided by its worker
count, times the grid size; for any other local model, the same per-case time scaled by the ratio of
the decode rates measured in the M0 smoke tests (4B rate / that model's rate) and divided by its own
worker count. A projection is a planning estimate, not a result.

Usage: python -m aib.pilot [--model qwen3-4b]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from aib.metrics import load_store


def summarize(records: list[dict], grid_sizes: dict[str, int], workers: dict[str, int], decode_rates: dict[str, float],
              pilot_model: str) -> dict:
    cases = [{
        "case_id": r["case_id"], "status": r["status"], "duration_s": r["duration_s"], "error_cause": r.get("error_cause"),
        "prompt_tokens": r["prompt_tokens"], "cached_tokens": r["cached_tokens"], "completion_tokens": r["completion_tokens"],
        "n_model_calls": r["n_model_calls"], "exposed": r.get("exposed"), "context_overflow": r.get("context_overflow"),
        "hit_max_iters": r.get("hit_max_iters"), "task_success": r.get("task_success"),
        "injection_success": r.get("injection_success"),
    } for r in sorted(records, key=lambda r: r["case_id"])]
    n = len(cases)
    mean_s = sum(c["duration_s"] for c in cases) / n if n else 0.0
    projections = {}
    for model, size in grid_sizes.items():
        scale = 1.0 if model == pilot_model else decode_rates[pilot_model] / decode_rates[model]
        seconds = mean_s * scale * size / workers[model]
        projections[model] = {"grid_cases": size, "workers": workers[model], "decode_rate_scale": round(scale, 4),
                              "projected_hours": round(seconds / 3600, 2)}
    statuses: dict[str, int] = {}
    for c in cases:
        statuses[c["status"]] = statuses.get(c["status"], 0) + 1
    return {
        "pilot_model": pilot_model, "n_cases": n, "statuses": dict(sorted(statuses.items())),
        "errored": statuses.get("errored", 0), "context_overflows": sum(bool(c["context_overflow"]) for c in cases),
        "hit_max_iters": sum(bool(c["hit_max_iters"]) for c in cases),
        "mean_case_seconds": round(mean_s, 3), "total_case_seconds": round(sum(c["duration_s"] for c in cases), 1),
        "decode_rates_tokens_per_s": {k: round(v, 2) for k, v in decode_rates.items()},
        "projection_method": "mean seconds per pilot case x decode-rate scale x grid cases / workers",
        "projections": projections, "cases": cases,
    }


def main(argv: list[str] | None = None) -> int:
    from aib.config import load_all
    from aib.experiments import build_grid
    from aib.repro import REPO_ROOT, make_stamp, require_clean

    parser = argparse.ArgumentParser(description="Summarize the pilot store and project the full-grid run time.")
    parser.add_argument("--model", default="qwen3-4b")
    args = parser.parse_args(argv)
    require_clean()
    cfg = load_all(REPO_ROOT / "config")
    stamp = make_stamp(f"python -m aib.pilot --model {args.model}", cfg.experiment.base_seed).to_dict()
    _, records = load_store(REPO_ROOT / "results" / "pilot" / f"{args.model}.jsonl")
    local = cfg.models.local_models()
    rates = {k: json.loads((REPO_ROOT / "results" / f"m0_smoke_{k}.json").read_text())["predicted_per_second"] for k in local}
    summary = summarize(records, {k: len(build_grid(k, cfg.experiment)) for k in local},
                        {k: cfg.models.models[k].workers for k in local}, rates, args.model)
    out = REPO_ROOT / "results" / "pilot.json"
    out.write_text(json.dumps({"stamp": stamp, **summary}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "cases"}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
