"""
Spearman-based circuit-stability metrics from cached per-example RelP
attribution scores at head_mlp granularity. Reads jsonls from cache/ and
writes heatmaps to results/spearman_stability/figs/.

Two metrics, both restricted to the union of top-K%-by-|score| components per
comparison:
  (a) Mean pairwise Spearman rho across examples (sampled pairs).
  (b) Mean Spearman rho between an aggregate (mean-|score|) reference ranking
      and each per-example ranking.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import rankdata

CMAP = LinearSegmentedColormap.from_list(
    "sharp_rdbu",
    [(0.0, "#053061"), (0.40, "#4393c3"), (0.50, "#f7f7f7"),
     (0.60, "#d6604d"), (1.0, "#67001f")],
)

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "cache"
OUT = REPO / "results" / "spearman_stability"
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
METHODS = ["relp", "eap"]
K_PCTS = [1, 5, 10, 20, 30]
N_PAIRS = 2000
SEED = 42


def cache_path(model_slug: str, task: str, method: str) -> Path:
    digit_slug = "d3" if task == "addition" else "dna"
    return CACHE / f"{model_slug}__none__{task}__{method}__n1000__{digit_slug}__s42.jsonl"


def load_score_matrix(model_slug: str, task: str, method: str):
    path = cache_path(model_slug, task, method)
    if not path.exists():
        return None
    examples = []
    comp_idx: dict[tuple, int] = {}
    with path.open() as f:
        for line in f:
            ex = json.loads(line)
            examples.append(ex["components"])
            for c in ex["components"]:
                key = (c["kind"], c["layer"], c["index"])
                if key not in comp_idx:
                    comp_idx[key] = len(comp_idx)
    N, D = len(examples), len(comp_idx)
    M = np.zeros((N, D), dtype=np.float32)
    for i, comps in enumerate(examples):
        for c in comps:
            M[i, comp_idx[(c["kind"], c["layer"], c["index"])]] = abs(float(c["score"]))
    return M


def topk_idx(row: np.ndarray, K_abs: int) -> np.ndarray:
    if K_abs >= row.shape[0]:
        return np.arange(row.shape[0])
    return np.argpartition(-row, K_abs)[:K_abs]


def spearman_on_union(x: np.ndarray, y: np.ndarray) -> float:
    if x.shape[0] < 2 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    rx, ry = rankdata(x), rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def pairwise_spearman(scores: np.ndarray, K_abs: int, n_pairs: int, rng: random.Random) -> float:
    N = scores.shape[0]
    top_per_ex = [topk_idx(scores[i], K_abs) for i in range(N)]
    vals = []
    for _ in range(n_pairs):
        i, j = rng.sample(range(N), 2)
        union = np.unique(np.concatenate([top_per_ex[i], top_per_ex[j]]))
        rho = spearman_on_union(scores[i, union], scores[j, union])
        if not np.isnan(rho):
            vals.append(rho)
    return float(np.mean(vals)) if vals else float("nan")


def reference_spearman(scores: np.ndarray, K_abs: int) -> float:
    aggregate = scores.mean(axis=0)
    agg_top = topk_idx(aggregate, K_abs)
    vals = []
    for i in range(scores.shape[0]):
        ex_top = topk_idx(scores[i], K_abs)
        union = np.unique(np.concatenate([agg_top, ex_top]))
        rho = spearman_on_union(scores[i, union], aggregate[union])
        if not np.isnan(rho):
            vals.append(rho)
    return float(np.mean(vals)) if vals else float("nan")


def compute_all(method: str) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(SEED)
    pairwise = np.full((len(MODELS), len(TASKS), len(K_PCTS)), np.nan, dtype=np.float32)
    reference = np.full_like(pairwise, np.nan)
    for mi, (slug, label) in enumerate(MODELS):
        for ti, task in enumerate(TASKS):
            M = load_score_matrix(slug, task, method)
            if M is None:
                print(f"[skip] {method} / {label} / {task}")
                continue
            N, D = M.shape
            for ki, K_pct in enumerate(K_PCTS):
                K_abs = max(1, int(round(D * K_pct / 100)))
                pairwise[mi, ti, ki] = pairwise_spearman(M, K_abs, N_PAIRS, rng)
                reference[mi, ti, ki] = reference_spearman(M, K_abs)
            print(f"[{method}/{label}/{task}] N={N} D={D}  "
                  f"pair@1%={pairwise[mi,ti,0]:.3f} pair@30%={pairwise[mi,ti,-1]:.3f}  "
                  f"ref@1%={reference[mi,ti,0]:.3f} ref@30%={reference[mi,ti,-1]:.3f}")
    return pairwise, reference


def plot_heatmap(data: np.ndarray, fname: str, title: str):
    fig, axes = plt.subplots(2, 3, figsize=(22, 11))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.86, bottom=0.18, wspace=0.45, hspace=0.22)
    for mi, (_, label) in enumerate(MODELS):
        ax = axes.flat[mi]
        im = ax.imshow(data[mi], aspect="auto", vmin=-1.0, vmax=1.0, cmap=CMAP)
        ax.set_xticks(range(len(K_PCTS)))
        ax.set_yticks(range(len(TASKS)))
        ax.set_yticklabels(TASKS, fontsize=18)
        ax.set_title(label, fontsize=22)
        for ti in range(len(TASKS)):
            for ki in range(len(K_PCTS)):
                v = data[mi, ti, ki]
                if np.isnan(v):
                    continue
                ax.text(ki, ti, f"{v:.2f}", ha="center", va="center",
                        color=("white" if abs(v) > 0.55 else "black"), fontsize=16)
        # Top row: hide x-tick labels (shared with bottom row). Bottom row only.
        if mi // 3 == 1:
            ax.set_xticklabels([f"{k}%" for k in K_PCTS], fontsize=18)
            ax.set_xlabel("top-$K$", fontsize=20)
        else:
            ax.set_xticklabels([])
    fig.suptitle(title, fontsize=24, y=0.96)
    cbar_ax = fig.add_axes([0.30, 0.07, 0.40, 0.022])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(r"Spearman $\rho$", fontsize=20)
    cbar.ax.tick_params(labelsize=17)
    out_path = FIGS / fname
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"wrote {out_path}")
    return out_path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    saved = {}
    for method in METHODS:
        print(f"\n=== method = {method} ===")
        pairwise, reference = compute_all(method)
        saved[f"{method}_pairwise"] = pairwise
        saved[f"{method}_reference"] = reference
        plot_heatmap(
            pairwise, f"spearman_{method}_pairwise_heatmap.png",
            f"Mean pairwise Spearman $\\rho$ across examples ({method}, head_mlp, n=1000)",
        )
        plot_heatmap(
            reference, f"spearman_{method}_reference_heatmap.png",
            f"Spearman $\\rho$ vs aggregate-ranking reference ({method}, head_mlp, n=1000)",
        )
    np.savez(
        OUT / "spearman_metrics.npz",
        models=[label for _, label in MODELS],
        tasks=TASKS,
        K_pcts=K_PCTS,
        **saved,
    )


if __name__ == "__main__":
    main()
