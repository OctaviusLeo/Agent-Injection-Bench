"""Plots generated from results/metrics.json, each with its 95% intervals (task 5.3; Req 7.3).

- results/plots/asr_by_defense.png: Attack_Success_Rate pooled over the Injection_Styles, per Model and defense.
- results/plots/defense_cost.png: Defense_Cost (TSR(none) - TSR(defense)) per Model and defense, for the clean
  condition (the benign cost) and pooled over the styles, with paired interval bars.

Usage: python -m aib.plots [--metrics results/metrics.json] [--out results/plots]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

DEFENSE_ORDER = ("none", "delimit", "hierarchy", "filter")
MODEL_ORDER = ("qwen3-4b", "llama3.1-8b", "qwen2.5-7b", "gpt-6-luna")
# Categorical slots 1-3 of the validated reference palette (blue, orange, aqua). Color follows the model, never its
# rank: the 4B model is slot 1, the 8B slot (llama3.1-8b or its fallback qwen2.5-7b, never both) slot 2, the hosted
# model slot 3. Aqua is below 3:1 contrast on the light surface, so the legend and results/summary.md carry identity.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
MODEL_COLORS = {"qwen3-4b": SERIES[0], "llama3.1-8b": SERIES[1], "qwen2.5-7b": SERIES[1], "gpt-6-luna": SERIES[2]}
OTHER = "#898781"
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def model_colors(models: list[str]) -> dict[str, str]:
    return {m: MODEL_COLORS.get(m, OTHER) for m in models}


def _legend(ax, colors: dict[str, str]) -> None:
    handles = [Patch(facecolor=c, edgecolor=SURFACE, label=m) for m, c in colors.items()]
    ax.legend(handles=handles, frameon=False, fontsize=9, labelcolor=INK, loc="upper right")


def _ordered_models(metrics: dict) -> list[str]:
    present = list(metrics.get("models", {}))
    return sorted(present, key=lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else len(MODEL_ORDER))


def _style(ax, title: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK_2)


def _bars(ax, groups: list[str], series: dict[str, dict[str, dict | None]], colors: dict[str, str]) -> None:
    n = max(len(series), 1)
    width = 0.8 / n
    for i, (model, values) in enumerate(series.items()):
        xs, ys, lo, hi, missing = [], [], [], [], []
        for g_index, g in enumerate(groups):
            x = g_index - 0.4 + width * (i + 0.5)
            v = values.get(g)
            if v is None or v.get("not_computed"):
                missing.append(x)
                continue
            xs.append(x)
            ys.append(v["point"])
            lo.append(v["point"] - v["lower"])
            hi.append(v["upper"] - v["point"])
        ax.bar(xs, ys, width=width * 0.92, color=colors[model], label=model, edgecolor=SURFACE, linewidth=1)
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=INK, elinewidth=1, capsize=3)
        for x in missing:
            ax.text(x, 0, "n/c", ha="center", va="bottom", fontsize=7, color=INK_2)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)


def plot_asr_by_defense(metrics: dict, path: Path) -> Path:
    models = _ordered_models(metrics)
    colors = model_colors(models)
    series = {m: {p["defense"]: p["asr"] for p in metrics.get("pooled", []) if p["model"] == m} for m in models}
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=SURFACE)
    _bars(ax, list(DEFENSE_ORDER), series, colors)
    _style(ax, "Attack success rate by defense (pooled over injection styles, 95% CI)", "attack success rate")
    ax.set_ylim(0, 1.08)
    _legend(ax, colors)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE, metadata={"Software": None})
    plt.close(fig)
    return path


def plot_defense_cost(metrics: dict, path: Path) -> Path:
    models = _ordered_models(metrics)
    colors = model_colors(models)
    defenses = [d for d in DEFENSE_ORDER if d != "none"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), facecolor=SURFACE, sharey=True)
    extent = 0.1
    for ax, condition, title in ((axes[0], "clean", "Benign cost: clean condition"),
                                 (axes[1], "pooled", "Pooled over injection styles")):
        series = {m: {c["defense"]: c["defense_cost"] for c in metrics.get("defense_cost", [])
                      if c["model"] == m and c["condition"] == condition} for m in models}
        for values in series.values():
            for v in values.values():
                if v and not v.get("not_computed"):
                    extent = max(extent, abs(v["lower"]), abs(v["upper"]))
        _bars(ax, defenses, series, colors)
        _style(ax, title, "Defense_Cost = TSR(none) - TSR(defense)" if condition == "clean" else "")
        ax.axhline(0, color=INK_2, linewidth=1)
    limit = min(1.0, round(extent * 1.15 + 0.005, 2))
    axes[0].set_ylim(-limit, limit)
    _legend(axes[1], colors)
    fig.suptitle("Task-success cost of each defense (paired 95% CI)", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE, metadata={"Software": None})
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    from aib.repro import REPO_ROOT

    parser = argparse.ArgumentParser(description="Write the two result plots from results/metrics.json.")
    parser.add_argument("--metrics", default="results/metrics.json")
    parser.add_argument("--out", default="results/plots")
    args = parser.parse_args(argv)
    metrics = json.loads((REPO_ROOT / args.metrics).read_text(encoding="utf-8"))
    out = REPO_ROOT / args.out
    for p in (plot_asr_by_defense(metrics, out / "asr_by_defense.png"), plot_defense_cost(metrics, out / "defense_cost.png")):
        print(f"wrote {p.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
