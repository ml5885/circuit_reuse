"""Zero vs mean ablation cross-task comparison.

Side-by-side figure mirroring the paper's Figure 3 layout: for each model,
six tasks with two bars per task (own circuit vs mean other circuit). The
left column of panels shows zero ablation (paper) and the right column
shows mean ablation (means computed over the corrupted distribution, per
Wang et al. 2022 / Miller et al. 2024).

Visual style matches the paper's cross_task_diagonal_vs_offdiag_K10.png.
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

REPO = Path("/Users/michaelli/Desktop/Research/circuit_reuse")
OUT_DIR = REPO / "results2" / "zero_vs_mean_ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TASKS = ["addition", "arc_challenge", "arc_easy", "boolean", "ioi", "mcqa"]
TASK_DISPLAY = {
    "addition": "Addition",
    "arc_challenge": "ARC (Chal.)",
    "arc_easy": "ARC (Easy)",
    "boolean": "Boolean",
    "ioi": "IOI",
    "mcqa": "CopyColors MCQA",
}

MODEL_FILE = {
    "Gemma 2 2B Instruct": "cross_task_google_gemma-2-2b-it",
    "Gemma 2 2B":          "cross_task_google_gemma-2-2b",
    "Llama-3.2-3B Instruct": "cross_task_meta-llama_Llama-3.2-3B-Instruct",
    "Llama-3.2-3B":        "cross_task_meta-llama_Llama-3.2-3B",
    "Qwen3-4B":            "cross_task_qwen3-4b",
    "Qwen3-8B":            "cross_task_qwen3-8b",
}

ZERO_DIR = REPO / "results" / "cross_task_ablation_k10"
MEAN_DIR = REPO / "results" / "cross_task_ablation_mean_k10"

COLOR_OWN = "#2B5F8C"
COLOR_OTHER = "#A6CEE3"


def compute_diag_offdiag(data):
    drops = data["accuracy_drop_pp"]
    diag = []
    offdiag = []
    for A in TASKS:
        diag.append(drops[A][A])
        others = [drops[A][B] for B in TASKS if B != A and B in drops[A]]
        offdiag.append(float(np.mean(others)) if others else float("nan"))
    return diag, offdiag


def draw_panel(ax, diag, offdiag, title, show_legend=False,
               y_lo=-20.0, y_hi=105.0):
    x = np.arange(len(TASKS))
    w = 0.35
    ax.bar(x - w / 2, diag, w,
           color=COLOR_OWN, edgecolor="none",
           label="Own circuit" if show_legend else None, zorder=2)
    ax.bar(x + w / 2, offdiag, w,
           color=COLOR_OTHER, edgecolor="none",
           label="Other circuits (mean)" if show_legend else None, zorder=2)
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_DISPLAY[t] for t in TASKS],
                       rotation=35, ha="right", fontsize=9)
    ax.set_ylim(y_lo, y_hi)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.6)
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=11, pad=4)


def main():
    nrows = 6  # one per model
    fig, axes = plt.subplots(nrows, 2, figsize=(11, 3.0 * nrows),
                             sharey=True)

    for r, (model, prefix) in enumerate(MODEL_FILE.items()):
        zero_path = ZERO_DIR / f"{prefix}_K10_t100.json"
        mean_path = MEAN_DIR / f"{prefix}_K10_t100_meanabl.json"

        with open(zero_path) as f:
            zero_data = json.load(f)
        with open(mean_path) as f:
            mean_data = json.load(f)

        zero_diag, zero_off = compute_diag_offdiag(zero_data)
        mean_diag, mean_off = compute_diag_offdiag(mean_data)

        ax_l = axes[r, 0]
        ax_r = axes[r, 1]
        title_l = f"{model}, zero ablation"
        title_r = f"{model}, mean ablation"
        draw_panel(ax_l, zero_diag, zero_off, title_l, show_legend=(r == 0))
        draw_panel(ax_r, mean_diag, mean_off, title_r)

        if r < nrows - 1:
            ax_l.tick_params(labelbottom=False)
            ax_r.tick_params(labelbottom=False)

    fig.supylabel("Accuracy Drop (pp)  $K=10$%", fontsize=13, x=0.04, y=0.5)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        fontsize=11,
        loc="lower center", bbox_to_anchor=(0.5, -0.01),
        ncol=2,
        frameon=True, edgecolor="grey", facecolor="white",
        fancybox=True, borderpad=0.4,
    )
    fig.tight_layout(rect=[0.04, 0.015, 1, 1.0])
    out = OUT_DIR / "zero_vs_mean_ablation_K10.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
