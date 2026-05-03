"""
Plot histograms of per-example |attribution score| from cached RelP and EAP
extractions.

For head_mlp: streams the local cache/ jsonls (n=1000 per task), bins on a
log-spaced grid, plots PDF on log-log axes — one panel per model, one line
per task. Two figures, one per method.

For neuron (Qwen3-4B, n=200): reads a precomputed NPZ produced by
analysis/compute_neuron_score_histogram.py (which runs on a babel compute
node since the cache is only mounted there).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "cache"
OUT = REPO / "results" / "score_distributions"
FIGS = OUT / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

MODELS = [
    ("google_gemma-2-2b", "Gemma-2-2B"),
    ("google_gemma-2-2b-it", "Gemma-2-2B-IT"),
    ("meta-llama_Llama-3.2-3B", "Llama-3.2-3B"),
    ("meta-llama_Llama-3.2-3B-Instruct", "Llama-3.2-3B-Instruct"),
    ("qwen3-4b", "Qwen3-4B"),
    ("qwen3-8b", "Qwen3-8B"),
]
TASKS = ["addition", "arc_challenge", "arc_easy", "boolean", "ioi", "mcqa"]
TASK_COLORS = {t: plt.cm.tab10.colors[i] for i, t in enumerate(TASKS)}

BINS = np.logspace(-6, 2, 80)
BIN_CENTERS = np.sqrt(BINS[:-1] * BINS[1:])


def cache_path(slug: str, task: str, method: str) -> Path:
    digit_slug = "d3" if task == "addition" else "dna"
    return CACHE / f"{slug}__none__{task}__{method}__n1000__{digit_slug}__s42.jsonl"


def histogram_for_task(path: Path, bins: np.ndarray) -> np.ndarray:
    """Stream the jsonl, bin |score| values into a fixed log-spaced grid."""
    counts = np.zeros(len(bins) - 1, dtype=np.int64)
    if not path.exists():
        return counts
    with path.open() as f:
        for line in f:
            ex = json.loads(line)
            vals = np.fromiter(
                (abs(c["score"]) for c in ex["components"]),
                dtype=np.float32,
                count=len(ex["components"]),
            )
            vals = vals[vals > 0]
            if vals.size == 0:
                continue
            h, _ = np.histogram(vals, bins=bins)
            counts += h
    return counts


def plot_method_panels(method: str) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.90, bottom=0.10, wspace=0.25, hspace=0.30)
    for mi, (slug, label) in enumerate(MODELS):
        ax = axes.flat[mi]
        any_data = False
        for task in TASKS:
            counts = histogram_for_task(cache_path(slug, task, method), BINS)
            total = counts.sum()
            if total == 0:
                continue
            density = counts / (total * np.diff(BINS))
            mask = density > 0
            ax.plot(BIN_CENTERS[mask], density[mask],
                    label=task, color=TASK_COLORS[task], linewidth=1.6)
            any_data = True
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(label, fontsize=18)
        if mi // 3 == 1:
            ax.set_xlabel(r"$|\mathrm{score}|$", fontsize=15)
        if mi % 3 == 0:
            ax.set_ylabel("density", fontsize=15)
        ax.tick_params(labelsize=12)
        if any_data and mi == 0:
            ax.legend(fontsize=10, loc="lower left")
        ax.grid(True, alpha=0.3, which="both")
    fig.suptitle(f"|attribution score| distribution — {method}, head_mlp, n=1000",
                 fontsize=22, y=0.96)
    out = FIGS / f"score_dist_{method}_head_mlp.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")
    return out


def plot_neuron_from_npz(npz_path: Path) -> Path | None:
    if not npz_path.exists():
        print(f"[skip neuron] no NPZ at {npz_path}")
        return None
    data = np.load(npz_path, allow_pickle=True)
    counts_per_task = {str(t): data[f"counts_{t}"] for t in data["tasks"]}
    bins = data["bins"]
    bin_centers = np.sqrt(bins[:-1] * bins[1:])

    fig, ax = plt.subplots(figsize=(10, 6))
    for task, counts in counts_per_task.items():
        total = counts.sum()
        if total == 0:
            continue
        density = counts / (total * np.diff(bins))
        mask = density > 0
        ax.plot(bin_centers[mask], density[mask],
                label=task, color=TASK_COLORS.get(task, "black"), linewidth=1.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$|\mathrm{score}|$", fontsize=15)
    ax.set_ylabel("density", fontsize=15)
    ax.set_title("|attribution score| distribution — RelP, neuron, Qwen3-4B, n=200",
                 fontsize=17)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=11, loc="lower left")
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    out = FIGS / "score_dist_relp_neuron_qwen3-4b.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")
    return out


def main():
    plot_method_panels("relp")
    plot_method_panels("eap")
    plot_neuron_from_npz(OUT / "neuron_score_histograms.npz")


if __name__ == "__main__":
    main()
