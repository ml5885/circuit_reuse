"""Compare EAP and EAP-IG attribution scores per (model, task).

Two heatmaps:
  1. Per-example Spearman correlation between EAP and EAP-IG component scores,
     averaged across examples.
  2. Per-example Jaccard overlap of the top-K% components selected by each
     method, averaged across examples, one heatmap per K.

Outputs to results/eap_vs_eap_ig/figs/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 150,
})

REPO = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO / "cache"
OUT_DIR = REPO / "results" / "eap_vs_eap_ig" / "figs"

MODELS = [
    ("google_gemma-2-2b", "Gemma-2-2B"),
    ("google_gemma-2-2b-it", "Gemma-2-2B-IT"),
    ("meta-llama_Llama-3.2-3B", "Llama-3.2-3B"),
    ("meta-llama_Llama-3.2-3B-Instruct", "Llama-3.2-3B-IT"),
    ("qwen3-4b", "Qwen3-4B"),
    ("qwen3-8b", "Qwen3-8B"),
]
TASKS = ["addition", "arc_challenge", "arc_easy", "boolean", "ioi", "mcqa"]
KS = [1, 5, 10, 30]


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


def compare_one(eap_path: Path, eapig_path: Path, ks: list[int]):
    eap = load_components(eap_path)
    eapig = load_components(eapig_path)
    n = min(len(eap), len(eapig))
    spearmans = []
    jaccards = {k: [] for k in ks}
    for a, b in zip(eap[:n], eapig[:n]):
        keys = sorted(a.keys() | b.keys())
        sa = np.array([a.get(k, 0.0) for k in keys])
        sb = np.array([b.get(k, 0.0) for k in keys])
        if sa.std() > 0 and sb.std() > 0:
            rho, _ = spearmanr(sa, sb)
            spearmans.append(rho)
        n_comp = len(keys)
        for k in ks:
            top_n = max(1, n_comp * k // 100)
            top_a = set(sorted(a, key=lambda x: -a[x])[:top_n])
            top_b = set(sorted(b, key=lambda x: -b[x])[:top_n])
            inter = len(top_a & top_b)
            union = len(top_a | top_b)
            jaccards[k].append(inter / union if union else 0.0)
    return {
        "n_examples": n,
        "spearman_mean": float(np.mean(spearmans)) if spearmans else float("nan"),
        "spearman_std": float(np.std(spearmans)) if spearmans else float("nan"),
        "jaccard_mean": {k: float(np.mean(v)) if v else float("nan") for k, v in jaccards.items()},
        "jaccard_std": {k: float(np.std(v)) if v else float("nan") for k, v in jaccards.items()},
    }


def heatmap(matrix, row_labels, col_labels, title, cbar_label, vmin, vmax, cmap, out_path):
    fig, ax = plt.subplots(figsize=(1.0 + 0.95 * len(col_labels), 0.7 + 0.55 * len(row_labels)))
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if np.isnan(v):
                continue
            txt_color = "white" if (vmax - v) / (vmax - vmin + 1e-9) > 0.55 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color=txt_color, fontsize=10)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ks", type=int, nargs="+", default=KS)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = p.parse_args()

    rows = []
    sp_matrix = np.full((len(MODELS), len(TASKS)), np.nan)
    jac_matrices = {k: np.full((len(MODELS), len(TASKS)), np.nan) for k in args.ks}

    for i, (key, label) in enumerate(MODELS):
        for j, task in enumerate(TASKS):
            eap_p = cache_path("eap", key, task)
            eapig_p = cache_path("eap_ig__ig5", key, task)
            if not (eap_p.exists() and eapig_p.exists()):
                print(f"[skip] {label} / {task}: missing cache")
                continue
            stats = compare_one(eap_p, eapig_p, args.ks)
            sp_matrix[i, j] = stats["spearman_mean"]
            for k in args.ks:
                jac_matrices[k][i, j] = stats["jaccard_mean"][k]
            rows.append({"model": label, "task": task, **stats})
            print(f"[ok] {label:18} {task:14} n={stats['n_examples']:4} "
                  f"rho={stats['spearman_mean']:.3f} "
                  f"jac@10={stats['jaccard_mean'].get(10, float('nan')):.3f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir.parent / "summary.json").write_text(json.dumps(rows, indent=2))

    row_labels = [m[1] for m in MODELS]
    heatmap(sp_matrix, row_labels, TASKS,
            "Spearman $\\rho$ of EAP vs EAP-IG per-component scores\n(per-example, then averaged)",
            "Spearman $\\rho$", vmin=0.0, vmax=1.0, cmap="viridis",
            out_path=args.out_dir / "spearman_eap_vs_eap_ig.png")

    for k in args.ks:
        heatmap(jac_matrices[k], row_labels, TASKS,
                f"Top-{k}% Jaccard between EAP and EAP-IG circuits\n(per-example, then averaged)",
                "Jaccard", vmin=0.0, vmax=1.0, cmap="magma",
                out_path=args.out_dir / f"jaccard_topk{k}_eap_vs_eap_ig.png")

    print(f"\nWrote heatmaps to {args.out_dir}")


if __name__ == "__main__":
    main()
