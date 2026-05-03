"""
Render Qwen3-4B neuron-granularity Spearman stability heatmaps from the NPZ
written by analysis/compute_spearman_neuron.py. Produces a 1x2 figure
(pairwise | reference) styled to match the head_mlp heatmaps.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "spearman_stability"
FIGS = OUT / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

CMAP = LinearSegmentedColormap.from_list(
    "sharp_rdbu",
    [(0.0, "#053061"), (0.40, "#4393c3"), (0.50, "#f7f7f7"),
     (0.60, "#d6604d"), (1.0, "#67001f")],
)


def main():
    data = np.load(OUT / "spearman_neuron_metrics.npz", allow_pickle=True)
    tasks = list(data["tasks"])
    K_pcts = list(data["K_pcts"])
    pairwise = data["pairwise"]
    reference = data["reference"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.87, bottom=0.22, wspace=0.40)
    panels = [
        (axes[0], pairwise, "Pairwise"),
        (axes[1], reference, "vs aggregate-ranking reference"),
    ]
    for ax, arr, label in panels:
        im = ax.imshow(arr, aspect="auto", vmin=-1.0, vmax=1.0, cmap=CMAP)
        ax.set_xticks(range(len(K_pcts)))
        ax.set_xticklabels([f"{int(k)}%" for k in K_pcts], fontsize=18)
        ax.set_yticks(range(len(tasks)))
        ax.set_yticklabels(tasks, fontsize=18)
        ax.set_title(label, fontsize=22)
        ax.set_xlabel("top-$K$", fontsize=20)
        for ti in range(len(tasks)):
            for ki in range(len(K_pcts)):
                v = arr[ti, ki]
                if np.isnan(v):
                    continue
                ax.text(ki, ti, f"{v:.2f}", ha="center", va="center",
                        color=("white" if abs(v) > 0.55 else "black"), fontsize=16)
    fig.suptitle("Spearman $\\rho$ stability — Qwen3-4B, neuron granularity (RelP, n=200)",
                 fontsize=24, y=0.96)
    cbar_ax = fig.add_axes([0.30, 0.10, 0.40, 0.022])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(r"Spearman $\rho$", fontsize=20)
    cbar.ax.tick_params(labelsize=17)
    out = FIGS / "spearman_relp_neuron_qwen3-4b_heatmap.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
