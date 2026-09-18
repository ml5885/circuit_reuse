"""Neuron-granularity cross-task ablation for Qwen3-4B across top-K and reuse threshold P.

Rows are per-example circuit sizes K (% of components); columns are the reuse threshold
P (% of examples a component must appear in to enter the shared circuit). Each cell is
the cross-task ablation heatmap (own circuit = diagonal). Shows that at neuron
granularity the shared set is empty at high P regardless of K, and that raising K
reintroduces cross-task overlap once shared sets are non-empty.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from analysis.plot_cross_task_results import render_heatmap, exclude_tasks
from circuit_reuse.dataset import get_model_display_name

MODEL = "qwen3-4b"
KS = [1, 5, 10, 20, 30]
PS = [100, 90, 75, 50]
ROOT = Path("results")


def load_cell(k, p):
    path = ROOT / f"e2_cross_task_ablation_neuron_k{k}_p{p}" / f"cross_task_{MODEL}_K{k}_t{p}.json"
    d = json.loads(path.read_text())
    return exclude_tasks([d], {"mmlu"})[0]


def main():
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})

    grid = [[load_cell(k, p) for p in PS] for k in KS]
    tasks = grid[0][0]["tasks"]

    all_vals = [grid[i][j]["accuracy_drop_pp"][s][t]
                for i in range(len(KS)) for j in range(len(PS))
                for s in tasks for t in tasks]
    vmax = max(all_vals)
    cmap = LinearSegmentedColormap.from_list("white_coral", ["#FFFFFF", "#F2C0A0", "#E76F51", "#9E2A1E"], N=256)

    fig, axes = plt.subplots(len(KS), len(PS), figsize=(3.6 * len(PS), 3.6 * len(KS)), squeeze=False)
    im = None
    for i, k in enumerate(KS):
        for j, p in enumerate(PS):
            ax = axes[i][j]
            im = render_heatmap(
                grid[i][j]["accuracy_drop_pp"], tasks, "", ax,
                vmin=0, vmax=vmax, cmap=cmap,
                source_label="",
            )
            ax.set_xlabel("")
            ax.set_ylabel("")
            if i == 0:
                ax.set_title(f"$P = {p}\\%$", fontsize=16, pad=8)
            ax.tick_params(labelbottom=(i == len(KS) - 1), labelleft=(j == 0), labelsize=9)

    fig.subplots_adjust(left=0.21, right=0.87, bottom=0.10, top=0.93, wspace=0.08, hspace=0.12)
    fig.suptitle(
        f"{get_model_display_name(MODEL)}: neuron-level cross-task ablation across $K$ and $P$",
        fontsize=18, y=0.975,
    )

    for i, k in enumerate(KS):
        pos = axes[i][0].get_position()
        fig.text(0.085, (pos.y0 + pos.y1) / 2, f"$K = {k}\\%$", rotation=90,
                 va="center", ha="center", fontsize=16)
    fig.text(0.035, 0.5, "Source task (circuit ablated)", rotation=90,
             va="center", ha="center", fontsize=16)
    fig.supxlabel("Target task", fontsize=16, y=0.005)

    cax = fig.add_axes([0.89, 0.20, 0.015, 0.55])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Accuracy drop (pp)", fontsize=15)

    out = Path("new_plots") / f"neuron_topk_sweep_{MODEL}.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=200)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
