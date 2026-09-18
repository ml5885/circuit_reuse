"""Composition of the shared-core / task-specific / task-complement decomposition,
split by attention head vs. MLP block, for every (model, task) cell.

For a task pair (A, B) at the component granularity, C_A and C_B decompose into
a shared core (C_A n C_B), a task-specific residual (C_A \\ C_B), and a
task-complement residual (C_B \\ C_A). This mirrors selective_ablation_experiment.py,
but instead of just counting components we also split each group by kind (MLP vs.
attention head) and average over every other task B, giving one 3-bar panel per
(model, task_a) cell.

Run: python -m analysis.plot_composition_mlp_head
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from circuit_reuse.dataset import get_model_display_name, get_task_display_name
from cross_task_experiment import find_metrics_file, load_shared_components

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"

REPO = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO / "results" / "cross_task"
OUT = (REPO / "paper2" /
       "_NeurIPS_2026_InterpScience_Workshop__How_Much_Do_Circuits_Tell_Us__"
       "Measuring_the_Consistency_and_Specificity_of_Language_Model_Circuits" /
       "figures" / "within_task" / "composition_mlp_head.png")

MODEL_ORDER = ["google/gemma-2-2b", "google/gemma-2-2b-it", "meta-llama/Llama-3.2-3B",
               "meta-llama/Llama-3.2-3B-Instruct", "qwen3-4b", "qwen3-8b"]
TASK_ORDER = ["addition", "arc_challenge", "arc_easy", "boolean", "ioi", "mcqa"]
K, P = 10, 100

CONDITIONS = ["shared_core", "residual_a", "residual_b"]
COND_LABEL = {"shared_core": "Shared Core", "residual_a": "Task-Specific",
              "residual_b": "Task-Complement"}
COND_COLOR = {"shared_core": "tab:green", "residual_a": "tab:red", "residual_b": "tab:blue"}


def load_component_sets() -> dict[tuple[str, str], set]:
    sets = {}
    for model in MODEL_ORDER:
        for task in TASK_ORDER:
            path = find_metrics_file(RESULTS_DIR, model, None, task)
            sets[model, task] = set(load_shared_components(path, K, P))
    return sets


def decompose(sets: dict[tuple[str, str], set], model: str, task_a: str) -> dict[str, tuple[float, float]]:
    """Mean (mlp, head) counts for each condition, averaged over every task_b."""
    counts = {cond: {"mlp": [], "head": []} for cond in CONDITIONS}
    for task_b in TASK_ORDER:
        if task_b == task_a:
            continue
        c_a, c_b = sets[model, task_a], sets[model, task_b]
        groups = {"shared_core": c_a & c_b, "residual_a": c_a - c_b, "residual_b": c_b - c_a}
        for cond, components in groups.items():
            counts[cond]["mlp"].append(sum(c.kind == "mlp" for c in components))
            counts[cond]["head"].append(sum(c.kind == "head" for c in components))
    return {cond: (np.mean(v["mlp"]), np.mean(v["head"])) for cond, v in counts.items()}


def plot(sets: dict[tuple[str, str], set]) -> None:
    nrows, ncols = len(MODEL_ORDER), len(TASK_ORDER)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 2.0 * nrows), sharey="row")

    x = np.arange(len(CONDITIONS))
    for r, model in enumerate(MODEL_ORDER):
        for c, task in enumerate(TASK_ORDER):
            ax = axes[r][c]
            comp = decompose(sets, model, task)
            for i, cond in enumerate(CONDITIONS):
                mlp, head = comp[cond]
                ax.bar(i, mlp, color=COND_COLOR[cond], edgecolor="white", linewidth=.5)
                ax.bar(i, head, bottom=mlp, color=COND_COLOR[cond], alpha=.45,
                       hatch="////", edgecolor="white", linewidth=.5)
            ax.set_xlim(-.7, len(CONDITIONS) - .3)
            ax.set_xticks([])
            ax.grid(axis="y", alpha=.25, linewidth=.5)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if r == 0:
                ax.set_title(get_task_display_name(task), fontsize=13)
            if c > 0:
                ax.tick_params(labelleft=False)
            if c == ncols - 1:
                ax.yaxis.set_label_position("right")
                ax.set_ylabel(get_model_display_name(model), rotation=270, labelpad=14,
                              fontsize=11, va="center")

    fig.tight_layout(rect=[0.035, 0.038, 1, 1])
    fig.canvas.draw()
    left_edge = min(axes[r][0].get_position().x0 for r in range(nrows))
    fig.text(left_edge - 0.022, 0.5, "Number of Components", rotation=90,
              va="center", ha="right", fontsize=14)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COND_COLOR[c], edgecolor="white")
               for c in CONDITIONS]
    handles += [plt.Rectangle((0, 0), 1, 1, facecolor="0.5", edgecolor="white"),
                plt.Rectangle((0, 0), 1, 1, facecolor="0.5", alpha=.45, hatch="////",
                              edgecolor="white")]
    labels = [COND_LABEL[c] for c in CONDITIONS] + ["MLP (solid)", "Attention Head (hatched)"]
    leg = fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.53, -0.003),
                     fontsize=12, frameon=True, fancybox=True, framealpha=1,
                     edgecolor="0.2", borderpad=.5, labelspacing=.4, handlelength=1.8)
    leg.get_frame().set_linewidth(.7)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {OUT}")


def main():
    sets = load_component_sets()
    plot(sets)


if __name__ == "__main__":
    main()
