"""Compare RelP component scores to EAP and EAP-IG per (model, task).

Produces a 1x2 figure:
  1. Per-example Spearman correlation between RelP and EAP scores.
  2. Per-example Spearman correlation between RelP and EAP-IG scores.

Each cell averages the per-example correlation over the examples shared by the
two caches.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 150,
})

REPO = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO / "cache"
OUT_DIR = REPO / "results" / "method_correlations" / "figs"

MODELS = [
    ("google_gemma-2-2b", "Gemma-2-2B"),
    ("google_gemma-2-2b-it", "Gemma-2-2B-IT"),
    ("meta-llama_Llama-3.2-3B", "Llama-3.2-3B"),
    ("meta-llama_Llama-3.2-3B-Instruct", "Llama-3.2-3B-IT"),
    ("qwen3-4b", "Qwen3-4B"),
    ("qwen3-8b", "Qwen3-8B"),
]
TASKS = ["addition", "arc_challenge", "arc_easy", "boolean", "ioi", "mcqa"]
COMPARISONS = [
    ("eap", "RelP vs EAP"),
    ("eap_ig__ig5", "RelP vs EAP-IG"),
]


def cache_path(method_token: str, model: str, task: str) -> Path:
    digits = "d3" if task == "addition" else "dna"
    return CACHE_DIR / f"{model}__none__{task}__{method_token}__n1000__{digits}__s42.jsonl"


def load_components(path: Path) -> list[dict[tuple, float]]:
    out = []
    with path.open() as f:
        for line in f:
            comps = json.loads(line)["components"]
            out.append({(c["layer"], c["kind"], c["index"]): c["score"] for c in comps})
    return out


def mean_spearman(path_a: Path, path_b: Path) -> tuple[float, int]:
    a_rows = load_components(path_a)
    b_rows = load_components(path_b)
    n = min(len(a_rows), len(b_rows))
    spearmans = []
    for a, b in zip(a_rows[:n], b_rows[:n]):
        keys = sorted(a.keys() | b.keys())
        sa = np.array([a.get(k, 0.0) for k in keys])
        sb = np.array([b.get(k, 0.0) for k in keys])
        if sa.std() > 0 and sb.std() > 0:
            rho, _ = spearmanr(sa, sb)
            spearmans.append(rho)
    return (float(np.mean(spearmans)) if spearmans else float("nan"), n)


def plot_pair(matrices: dict[str, np.ndarray], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.9))
    fig.subplots_adjust(left=0.10, right=0.95, bottom=0.18, top=0.96, wspace=0.28)
    row_labels = [m[1] for m in MODELS]

    for ax, (method, panel_label) in zip(axes, COMPARISONS):
        matrix = matrices[method]
        im = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_xticks(np.arange(len(TASKS)))
        ax.set_xticklabels(TASKS, rotation=30, ha="right")
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels(row_labels)
        ax.set_title(panel_label, fontsize=12, pad=4)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                v = matrix[i, j]
                if np.isnan(v):
                    continue
                txt_color = "white" if v < 0.55 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", color=txt_color, fontsize=9)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.04, pad=0.02)
    cbar.set_label("Spearman rho")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    matrices = {method: np.full((len(MODELS), len(TASKS)), np.nan) for method, _ in COMPARISONS}
    rows = []

    for i, (model_key, model_label) in enumerate(MODELS):
        for j, task in enumerate(TASKS):
            relp_path = cache_path("relp", model_key, task)
            if not relp_path.exists():
                continue
            for method, _ in COMPARISONS:
                other_path = cache_path(method, model_key, task)
                if not other_path.exists():
                    continue
                rho, n = mean_spearman(relp_path, other_path)
                matrices[method][i, j] = rho
                rows.append({
                    "model": model_label,
                    "task": task,
                    "comparison": f"relp_vs_{method}",
                    "spearman_mean": rho,
                    "n_examples": n,
                })

    OUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    (OUT_DIR.parent / "summary.json").write_text(json.dumps(rows, indent=2))
    plot_pair(matrices, OUT_DIR / "spearman_relp_vs_others.png")


if __name__ == "__main__":
    main()
