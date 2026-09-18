"""Per-model diagonal vs off-diagonal accuracy drop with variability.

Addresses ye6M Concern 3 / Q4: report uncertainty on the own-circuit drop
$\\Delta_A^A$ and the mean other-circuit drop $\\mathrm{mean}_{B \\neq A} \\Delta_A^B$
for each (model, task) at K=10%, P=100%.

We load the saved cross-task ablation JSONs (per-model accuracy matrices),
and for each target task A and each model:
  * diagonal bar: own-circuit drop, with 95% normal-approx CI on a difference
    of two proportions using n=100 eval examples.
  * off-diagonal bar: mean of 5 other-task circuit drops, with 95% t-CI across
    the 5 other tasks, plus the individual task drops shown as dots.

Also produces analogous plots for EAP-IG, RelP (addresses ye6M C2, ooA1 C4,
oP9Z C3), and a head+MLP vs neuron-granularity comparison on Qwen3-4B
(addresses ye6M C1, ooA1 C3).
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

matplotlib.use("Agg")
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

REPO = Path("/Users/michaelli/Desktop/Research/circuit_reuse")
OUT_DIR = REPO / "results2" / "own_vs_other_variability"
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
    "Gemma 2 2B Instruct": "cross_task_google_gemma-2-2b-it_K10_t100.json",
    "Gemma 2 2B": "cross_task_google_gemma-2-2b_K10_t100.json",
    "Llama-3.2-3B Instruct": "cross_task_meta-llama_Llama-3.2-3B-Instruct_K10_t100.json",
    "Llama-3.2-3B": "cross_task_meta-llama_Llama-3.2-3B_K10_t100.json",
    "Qwen3-4B": "cross_task_qwen3-4b_K10_t100.json",
    "Qwen3-8B": "cross_task_qwen3-8b_K10_t100.json",
}

# Colors match the paper's Figure 3 (cross_task_diagonal_vs_offdiag_K10.png) so
# this figure reads as an updated version of that figure with error bars.
COLOR_OWN = "#2B5F8C"
COLOR_OTHER = "#A6CEE3"
COLOR_DOTS = "#1F3A55"


def diff_props_ci95(p_baseline, p_ablated, n):
    """Approximate 95% CI half-width on (p_baseline - p_ablated) under the
    conservative independence assumption. p_baseline and p_ablated are fractions."""
    var = p_baseline * (1 - p_baseline) / n + p_ablated * (1 - p_ablated) / n
    return 1.96 * np.sqrt(max(var, 0.0))


def t_ci95(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return float("nan")
    return stats.t.ppf(0.975, len(arr) - 1) * arr.std(ddof=1) / np.sqrt(len(arr))


def clipped_yerr(means, errs, lo=0.0, hi=100.0):
    means = np.asarray(means, dtype=float)
    errs = np.asarray(errs, dtype=float)
    errs = np.where(np.isnan(errs), 0.0, errs)
    errs = np.maximum(errs, 0.0)
    room_below = np.maximum(means - lo, 0.0)
    room_above = np.maximum(hi - means, 0.0)
    lower = np.minimum(errs, room_below)
    upper = np.minimum(errs, room_above)
    return np.vstack([lower, upper])


def compute_bars(data, n_eval=100, tasks=TASKS):
    baseline = data["baseline_accuracy"]
    drops = data["accuracy_drop_pp"]
    ablated = data["ablated_accuracy_pct"]

    diag_vals, diag_errs = [], []
    other_vals, other_errs = [], []
    other_points = []
    for A in tasks:
        if A not in baseline or A not in drops:
            diag_vals.append(np.nan); diag_errs.append(np.nan)
            other_vals.append(np.nan); other_errs.append(np.nan)
            other_points.append([])
            continue
        ablated_A_by_A = ablated[A][A] / 100.0
        base_A = baseline[A]
        diag_vals.append(drops[A][A])
        diag_errs.append(100.0 * diff_props_ci95(base_A, ablated_A_by_A, n_eval))

        other_drops = [drops[A][B] for B in tasks if B != A and B in drops[A]]
        other_vals.append(float(np.mean(other_drops)) if other_drops else np.nan)
        other_errs.append(t_ci95(other_drops))
        other_points.append(other_drops)
    return diag_vals, diag_errs, other_vals, other_errs, other_points


def draw_own_other_bars(ax, diag_vals, diag_errs, other_vals, other_errs, other_points,
                       show_legend=True, w=0.35,
                       y_lo=-20.0, y_hi=105.0):
    x = np.arange(len(TASKS))
    ax.bar(
        x - w / 2, diag_vals, w,
        yerr=clipped_yerr(diag_vals, diag_errs, lo=y_lo, hi=y_hi), capsize=3,
        color=COLOR_OWN, edgecolor="none",
        label="Own circuit" if show_legend else None,
        error_kw={"elinewidth": 1.0, "ecolor": "#333"},
        zorder=2,
    )
    ax.bar(
        x + w / 2, other_vals, w,
        yerr=clipped_yerr(other_vals, other_errs, lo=y_lo, hi=y_hi), capsize=3,
        color=COLOR_OTHER, edgecolor="none",
        label="Other circuits (mean)" if show_legend else None,
        error_kw={"elinewidth": 1.0, "ecolor": "#333"},
        zorder=2,
    )
    for i, pts in enumerate(other_points):
        if not pts:
            continue
        jitter = np.random.default_rng(i).uniform(-0.06, 0.06, size=len(pts))
        ax.scatter(
            np.full(len(pts), x[i] + w / 2) + jitter, pts,
            s=14, color=COLOR_DOTS, alpha=0.7,
            zorder=4, edgecolor="white", linewidths=0.4,
            label="Individual other-task drop" if (show_legend and i == 0) else None,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_DISPLAY[t] for t in TASKS], rotation=35, ha="right", fontsize=10)
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.6)
    ax.set_ylim(y_lo, y_hi)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.6)
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=10)


def make_six_model_plot(src_dir: Path, out_path: Path, k_label: str = "10",
                        model_file_map=MODEL_FILE):
    ncols = 3
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 6.0), sharey=True)
    axes = np.atleast_2d(axes).reshape(nrows, ncols)
    axes_flat = axes.flatten()

    for idx, (model, fname) in enumerate(model_file_map.items()):
        ax = axes_flat[idx]
        path = src_dir / fname
        if not path.exists():
            ax.set_title(f"{model} (missing)", fontsize=12)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        with open(path) as f:
            data = json.load(f)
        bars = compute_bars(data)
        draw_own_other_bars(ax, *bars, show_legend=(idx == 0))
        ax.set_title(model, fontsize=12, pad=4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        row, col = divmod(idx, ncols)
        if row == 0:
            ax.tick_params(labelbottom=False)

    fig.supylabel(f"Accuracy drop (pp), K={k_label}%", fontsize=13, x=0.06, y=0.55)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        fontsize=11,
        loc="lower center", bbox_to_anchor=(0.5, -0.06),
        ncol=3,
        frameon=True, edgecolor="grey", facecolor="white",
        fancybox=True, borderpad=0.4,
    )
    fig.tight_layout(rect=[0.04, 0, 1, 1.0])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def make_granularity_plot(head_mlp_path: Path, neuron_path: Path,
                          out_path: Path, model_name="Qwen3-4B",
                          method_label="RelP", p_label="50"):
    """Side-by-side head+MLP vs neuron-granularity own-vs-other on one model."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True)

    for ax, (path, label) in zip(axes, [
        (head_mlp_path, "Head + MLP"),
        (neuron_path,   "Neuron"),
    ]):
        with open(path) as f:
            data = json.load(f)
        bars = compute_bars(data)
        draw_own_other_bars(ax, *bars, show_legend=(label == "Head + MLP"))
        ax.set_title(f"{label} granularity", fontsize=13, pad=4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Accuracy drop (pp), K=10%", fontsize=12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        fontsize=11,
        loc="lower center", bbox_to_anchor=(0.5, -0.08),
        ncol=3,
        frameon=True, edgecolor="grey", facecolor="white",
        fancybox=True, borderpad=0.4,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    # Existing EAP plot
    make_six_model_plot(
        src_dir=REPO / "results" / "cross_task_ablation_k10",
        out_path=OUT_DIR / "own_vs_other_K10_P100.png",
        k_label="10",
    )

    # EAP-IG
    make_six_model_plot(
        src_dir=REPO / "results_eap_ig" / "cross_task_ablation_eap_ig_k10_p100",
        out_path=OUT_DIR / "eap_ig_own_vs_other_K10_P100.png",
        k_label="10",
    )

    # Head+MLP vs Neuron on Qwen3-4B (both use RelP). Use P=50% so circuits
    # are non-trivial; the strict P=100% threshold produces near-empty RelP
    # shared sets and obscures the granularity comparison.
    make_granularity_plot(
        head_mlp_path=REPO / "results" / "e2_cross_task_ablation_head_mlp_k10_p50"
                      / "cross_task_qwen3-4b_K10_t50.json",
        neuron_path=REPO / "results" / "e2_cross_task_ablation_neuron_k10_p50"
                    / "cross_task_qwen3-4b_K10_t50.json",
        out_path=OUT_DIR / "granularity_qwen3-4b_K10_P50.png",
    )


if __name__ == "__main__":
    main()
