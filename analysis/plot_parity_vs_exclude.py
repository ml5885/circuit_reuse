"""Side-by-side comparison of C^3 Exclude vs C^3 Parity necessity scores."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from circuit_reuse.dataset import get_task_display_name, get_model_display_name

PARITY_DIRS = [
    Path("results/c3_parity/slurm_5385629"),
    Path("results/c3_parity/slurm_5386189"),
]
EXCLUDE_DIR = Path("results/c3_exclude/results/slurm_5387813")

SKIP_TASKS = {"mmlu", "addition"}


def load_all_metrics(results_dirs: list[Path]) -> list[dict]:
    rows = []
    for results_dir in results_dirs:
        for path in sorted(results_dir.rglob("metrics.json")):
            with path.open() as f:
                r = json.load(f)
            base_val = r["baseline_val_accuracy"]
            for k_str, block in r["by_k"].items():
                for p_str, tblock in block["thresholds"].items():
                    va = tblock["val"]
                    lift = (va["control_accuracy"] - va["ablation_accuracy"]) / base_val if base_val else float("nan")
                    rows.append({
                        "model": r["model_name"],
                        "task": r["task"],
                        "top_k": int(k_str),
                        "reuse_threshold": int(p_str),
                        "necessity": lift,
                    })
    return rows


def main():
    df_parity = pd.DataFrame(load_all_metrics(PARITY_DIRS))
    df_exclude = pd.DataFrame(load_all_metrics([EXCLUDE_DIR]))

    df_parity["condition"] = "Parity"
    df_exclude["condition"] = "Exclude"

    df = pd.concat([df_exclude, df_parity], ignore_index=True)
    df = df[~df["task"].isin(SKIP_TASKS)]
    df["task_display"] = df["task"].apply(get_task_display_name)
    df["model_display"] = df["model"].apply(get_model_display_name)

    # Fix reuse_threshold=100, average across K values
    df = df[df["reuse_threshold"] == 100]

    # Deduplicate: if both slurm dirs have the same model/task/K, keep the later one
    df = df.drop_duplicates(subset=["model", "task", "top_k", "condition"], keep="last")

    grouped = df.groupby(["model_display", "task_display", "condition"]).agg(
        mean=("necessity", "mean"),
        std=("necessity", "std"),
    ).reset_index()

    models = sorted(grouped["model_display"].unique())
    tasks = sorted(grouped["task_display"].unique())
    conditions = ["Exclude", "Parity"]
    pastel = sns.color_palette("pastel")
    colors = {"Exclude": pastel[0], "Parity": pastel[3]}

    plt.rcParams.update({"font.family": "serif", "font.size": 12})

    ncols = 3
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for idx, model in enumerate(models):
        ax = axes[idx // ncols][idx % ncols]
        sub = grouped[grouped["model_display"] == model]

        x = np.arange(len(tasks))
        width = 0.35

        for i, cond in enumerate(conditions):
            means, stds = [], []
            for task in tasks:
                row = sub[(sub["task_display"] == task) & (sub["condition"] == cond)]
                means.append(row["mean"].values[0] if len(row) else np.nan)
                stds.append(row["std"].values[0] if len(row) else 0)
            offset = (i - 0.5) * width
            ax.bar(x + offset, means, width, yerr=stds, capsize=3,
                   label=f"$C^3$ {cond}", color=colors[cond],
                   edgecolor="black", linewidth=0.5)

        ax.axhline(0, color="gray", linewidth=1, linestyle="--", alpha=0.7)
        ax.set_xticks(x)
        if idx // ncols == nrows - 1:
            ax.set_xticklabels(tasks, rotation=30, ha="right", fontsize=10)
        else:
            ax.set_xticklabels([])
        ax.set_title(model, fontsize=14)
        ax.grid(axis="y", linestyle="-", alpha=0.3)

        if idx % ncols == 0:
            ax.set_ylabel("Necessity", fontsize=16)

    for idx in range(len(models), nrows * ncols):
        fig.delaxes(axes[idx // ncols][idx % ncols])

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=2, fontsize=16, frameon=True)

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    out = Path("results/parity_vs_exclude_necessity.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
