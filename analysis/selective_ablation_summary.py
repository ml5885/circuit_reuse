"""Aggregate the shared-core / task-residual ablation into one readable panel.

The per-model grids answer the same question six times over.  What the argument
needs is whether the shared core carries the damage and whether the task-only
remainder is selective, aggregated over every model and task pair, with the
per-model spread shown rather than hidden.

This experiment exists only at the head-and-MLP granularity; the neuron analog
would need a fresh ablation sweep, so the figure is labelled accordingly.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "circuit_reuse_mplconfig"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd

from analysis.granularity_parity import (EXCLUDED_TASKS, GRAN_COLORS,
                                         SECONDARY, SECONDARY_LIGHT,
                                         _legend, _task_label)

CONDITIONS = [("shared_core", "Shared core\n$C_A \\cap C_B$"),
              ("residual_a", "Task-only\n$C_A \\setminus C_B$"),
              ("residual_b", "Other-only\n$C_B \\setminus C_A$"),
              ("random_control", "Random control\n(size-matched)")]
# Navy for the shared core, since that is literally the heads-and-MLP shared
# circuit the rest of the paper draws in navy; the two residuals take the
# secondary; the control stays neutral. Red is deliberately unused here, so it
# cannot be misread as the neuron granularity.
COLORS = {"shared_core": GRAN_COLORS["head_mlp"], "residual_a": SECONDARY,
          "residual_b": SECONDARY_LIGHT, "random_control": "#7F7F7F"}


def read(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.glob("selective_*.json")):
        d = json.loads(path.read_text())
        a = d["task_a"]
        if a in EXCLUDED_TASKS or d["task_b"] in EXCLUDED_TASKS:
            continue
        for condition, _ in CONDITIONS:
            cell = d["conditions"].get(condition)
            if not cell or not cell.get("size"):
                continue
            evals = {t: v["relative_drop_pct"] for t, v in cell.items()
                     if t != "size" and t not in EXCLUDED_TASKS}
            others = [v for t, v in evals.items() if t != a]
            rows.append({"model": d["model_name"], "task_a": a, "task_b": d["task_b"],
                         "condition": condition, "size": cell["size"],
                         "target": evals.get(a, np.nan),
                         "non_target": float(np.mean(others)) if others else np.nan})
    return pd.DataFrame(rows)


def make_figure(df: pd.DataFrame, out: Path):
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                         "axes.titlesize": 11, "axes.labelsize": 10,
                         "xtick.labelsize": 9, "ytick.labelsize": 9,
                         "legend.fontsize": 9, "figure.titlesize": 12})

    def style(ax):
        ax.grid(alpha=.25, linewidth=.6, axis="y")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    order = [c for c, _ in CONDITIONS]
    x = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6),
                             gridspec_kw={"width_ratios": [1.35, 1]})

    # Damage by condition, split into the task the circuit came from and the rest.
    ax = axes[0]
    for i, (col, offset, hatch, alpha) in enumerate(
            [("target", -.19, None, 1.0), ("non_target", .19, "///", .45)]):
        means = [df[df.condition == c][col].mean() for c in order]
        ax.bar(x + offset, means, .38, color=[COLORS[c] for c in order], alpha=alpha,
               hatch=hatch, edgecolor="0.25", linewidth=.7,
               label="Target task" if col == "target" else "Other tasks (mean)")
        for xi, c in enumerate(order):
            per_model = df[df.condition == c].groupby("model")[col].mean()
            ax.scatter(np.full(len(per_model), xi + offset), per_model.values, s=13,
                       color="0.15", alpha=.75, zorder=3, linewidth=0)
    ax.set_xticks(x, [label for _, label in CONDITIONS])
    ax.axhline(0, c="0.3", lw=.8)
    ax.set(ylabel="Relative accuracy drop (%)")
    _legend(ax)
    style(ax)

    # Selectivity, against the control the experiment size-matches to it. Raw
    # target-minus-other is confounded by how fragile a task is: a random
    # ablation of the same size looks just as selective on a brittle task.
    ax = axes[1]
    sel = df.assign(selectivity=df.target - df.non_target)
    wide = sel.pivot_table(index=["model", "task_a", "task_b"], columns="condition",
                           values="selectivity").dropna(subset=["residual_a",
                                                                "random_control"])
    pair = [("random_control", "Random control\n(size-matched)"),
            ("residual_a", "Task-only\n$C_A \\setminus C_B$")]
    for xi, (c, _) in enumerate(pair):
        per_model = wide.groupby("model")[c].mean()
        ax.scatter(np.full(len(per_model), xi), per_model.values, s=40, color=COLORS[c],
                   alpha=.85, zorder=3, edgecolor="white", linewidth=.7)
    for model, row in wide.groupby("model")[[c for c, _ in pair]].mean().iterrows():
        ax.plot([0, 1], row.values, color="0.55", lw=1, alpha=.7, zorder=2)
    means = [wide[c].mean() for c, _ in pair]
    ax.plot([0, 1], means, color="0.1", lw=2.6, marker="o", markersize=8,
            markeredgecolor="white", markeredgewidth=1.2, zorder=4)
    delta = wide.residual_a - wide.random_control
    boot = [delta.sample(len(delta), replace=True, random_state=s).mean()
            for s in range(2000)]
    ax.annotate(f"{delta.mean():+.1f} pp\n[{np.quantile(boot, .025):.1f}, "
                f"{np.quantile(boot, .975):.1f}]", (1, means[1]),
                textcoords="offset points", xytext=(-8, 14), ha="right", fontsize=9,
                fontweight="bold")
    ax.axhline(0, c="0.3", lw=.9)
    ax.set_xlim(-.35, 1.35)
    ax.set_xticks([0, 1], [label for _, label in pair])
    ax.set(ylabel="Target minus other-task drop (pp)")
    style(ax)

    fig.tight_layout()
    fig.savefig(out / "selective_ablation_summary.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="results/selective_ablation_k10")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--p", type=int, default=100)
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = read(Path(args.results_root))
    df.to_csv(out / "selective_ablation_tidy.csv", index=False)
    make_figure(df, out)
    summary = df.assign(selectivity=df.target - df.non_target).groupby("condition")[
        ["size", "target", "non_target", "selectivity"]].mean()
    print(summary.to_string())


if __name__ == "__main__":
    main()
