"""Compute cross-task circuit overlap restricted to attention heads (no MLPs).

Answers reviewer ye6M Q1: how much of the cross-task overlap remains if MLP
layers are excluded and circuits are computed only over attention heads?

We work directly from the per-input attribution caches in `cache/`. For each
example we sort components by score, take the top-K% within either (a) the
full pool (heads + MLPs, baseline matching the paper) or (b) the heads-only
pool. We then aggregate per-input circuits at threshold P to get a per-task
shared set, and compute pairwise overlap (Jaccard and Overlap Coefficient).
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 11,
})

REPO = Path("/Users/michaelli/Desktop/Research/circuit_reuse")
CACHE_DIR = REPO / "cache"
OUT_DIR = REPO / "results2" / "attention_only_overlap"

TASKS = ["addition", "arc_challenge", "arc_easy", "boolean", "ioi", "mcqa"]
TASK_DISPLAY = {
    "addition": "Addition",
    "arc_challenge": "ARC (C)",
    "arc_easy": "ARC (E)",
    "boolean": "Boolean",
    "ioi": "IOI",
    "mcqa": "MCQA",
}

MODELS = [
    "google_gemma-2-2b",
    "google_gemma-2-2b-it",
    "meta-llama_Llama-3.2-3B",
    "meta-llama_Llama-3.2-3B-Instruct",
    "qwen3-4b",
    "qwen3-8b",
]
MODEL_DISPLAY = {
    "google_gemma-2-2b": "Gemma 2 2B",
    "google_gemma-2-2b-it": "Gemma 2 2B IT",
    "meta-llama_Llama-3.2-3B": "Llama-3.2-3B",
    "meta-llama_Llama-3.2-3B-Instruct": "Llama-3.2-3B Inst",
    "qwen3-4b": "Qwen3-4B",
    "qwen3-8b": "Qwen3-8B",
}


def load_per_input_scores(cache_path: Path):
    """Returns list of dicts: each is {component_id_str: score} for one example."""
    examples = []
    with open(cache_path) as f:
        for line in f:
            d = json.loads(line)
            scores = {}
            for c in d["components"]:
                key = f"{c['kind']}[layer={c['layer']}, index={c['index']}]"
                scores[key] = c["score"]
            examples.append(scores)
    return examples


def is_head(comp_id: str) -> bool:
    return comp_id.startswith("head[")


def shared_set(per_input_scores, K_pct: int, P_pct: int, heads_only: bool):
    """Compute the shared component set for one task at top-K% and threshold P%."""
    if heads_only:
        filtered = [{c: s for c, s in sc.items() if is_head(c)} for sc in per_input_scores]
    else:
        filtered = per_input_scores

    total = len(filtered[0])
    take = max(1, int(total * K_pct / 100))
    counts = Counter()
    for sc in filtered:
        ranked = sorted(sc.items(), key=lambda x: abs(x[1]), reverse=True)
        for c, _ in ranked[:take]:
            counts[c] += 1
    n_ex = len(filtered)
    need = int(np.ceil(P_pct / 100.0 * n_ex))
    return {c for c, cnt in counts.items() if cnt >= need}, take


def pairwise_overlap(circuits: dict):
    """Return DataFrames of Jaccard and Overlap Coefficient between task pairs."""
    tasks = list(circuits.keys())
    n = len(tasks)
    jac = np.full((n, n), np.nan)
    oc = np.full((n, n), np.nan)
    for i, t1 in enumerate(tasks):
        for j, t2 in enumerate(tasks):
            a, b = circuits[t1], circuits[t2]
            if not a or not b:
                continue
            inter = len(a & b)
            union = len(a | b)
            jac[i, j] = inter / union if union else 0.0
            oc[i, j] = inter / min(len(a), len(b))
    labels = [TASK_DISPLAY.get(t, t) for t in tasks]
    return (
        pd.DataFrame(jac, index=labels, columns=labels),
        pd.DataFrame(oc, index=labels, columns=labels),
    )


def offdiag_mean(df: pd.DataFrame) -> float:
    arr = df.values.astype(float)
    mask = ~np.eye(arr.shape[0], dtype=bool)
    vals = arr[mask]
    vals = vals[~np.isnan(vals)]
    return float(np.mean(vals)) if len(vals) else float("nan")


def offdiag_unique(df: pd.DataFrame):
    """Return the upper-triangle off-diagonal values (one per unordered task pair)."""
    arr = df.values.astype(float)
    n = arr.shape[0]
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            v = arr[i, j]
            if not np.isnan(v):
                out.append((df.index[i], df.columns[j], v))
    return out


def collect_for_model(model: str, method: str, K: int, P: int):
    """Build per-task shared sets at both granularities for one model."""
    all_circuits = {}
    heads_circuits = {}
    sizes_all = {}
    sizes_heads = {}
    for task in TASKS:
        d_suffix = "d3" if task == "addition" else "dna"
        fname = f"{model}__none__{task}__{method}__n1000__{d_suffix}__s42.jsonl"
        path = CACHE_DIR / fname
        if not path.exists():
            print(f"  [skip] missing {path.name}", file=sys.stderr)
            continue
        examples = load_per_input_scores(path)
        s_all, take_all = shared_set(examples, K, P, heads_only=False)
        s_heads, take_h = shared_set(examples, K, P, heads_only=True)
        all_circuits[task] = s_all
        heads_circuits[task] = s_heads
        sizes_all[task] = (len(s_all), take_all)
        sizes_heads[task] = (len(s_heads), take_h)
    return all_circuits, heads_circuits, sizes_all, sizes_heads


def plot_heatmap(df: pd.DataFrame, title: str, out_path: Path, vmax: float = 1.0):
    n = df.shape[0]
    fig, ax = plt.subplots(figsize=(0.85 * n + 2.0, 0.7 * n + 1.5))
    arr = df.values.astype(float)
    im = ax.imshow(arr, cmap="YlOrRd", vmin=0, vmax=vmax)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(df.columns, rotation=45, ha="right")
    ax.set_yticklabels(df.index)
    for i in range(n):
        for j in range(n):
            v = arr[i, j]
            if np.isnan(v):
                continue
            color = "white" if v > 0.55 * vmax else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", color=color, fontsize=10)
    ax.set_title(title, fontsize=12)
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_paired_heatmap(jac_all, jac_heads, oc_all, oc_heads, model_disp, K, P, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    for ax, df, title, vmax in [
        (axes[0, 0], jac_all,   f"Jaccard (head+MLP), K={K}%, P={P}%",     1.0),
        (axes[0, 1], jac_heads, f"Jaccard (heads only), K={K}%, P={P}%",   1.0),
        (axes[1, 0], oc_all,    f"Overlap Coeff (head+MLP), K={K}%, P={P}%", 1.0),
        (axes[1, 1], oc_heads,  f"Overlap Coeff (heads only), K={K}%, P={P}%", 1.0),
    ]:
        arr = df.values.astype(float)
        n = arr.shape[0]
        im = ax.imshow(arr, cmap="YlOrRd", vmin=0, vmax=vmax)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(df.columns, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(df.index, fontsize=9)
        for i in range(n):
            for j in range(n):
                v = arr[i, j]
                if np.isnan(v):
                    continue
                color = "white" if v > 0.55 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", color=color, fontsize=8)
        ax.set_title(title, fontsize=11)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f"{model_disp}: cross-task overlap with vs. without MLP layers", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="eap")
    parser.add_argument("--K", type=int, nargs="+", default=[1, 5, 10, 20, 30])
    parser.add_argument("--P", type=int, default=85)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    pair_rows = []
    for model in MODELS:
        model_disp = MODEL_DISPLAY[model]
        print(f"[model] {model_disp}")
        for K in args.K:
            print(f"  K={K}%, P={args.P}%")
            all_c, heads_c, sizes_all, sizes_heads = collect_for_model(model, args.method, K, args.P)
            if not all_c:
                continue
            jac_all, oc_all = pairwise_overlap(all_c)
            jac_heads, oc_heads = pairwise_overlap(heads_c)

            jac_all_off = offdiag_mean(jac_all)
            jac_heads_off = offdiag_mean(jac_heads)
            oc_all_off = offdiag_mean(oc_all)
            oc_heads_off = offdiag_mean(oc_heads)

            mean_size_all = float(np.mean([s for s, _ in sizes_all.values()]))
            mean_size_heads = float(np.mean([s for s, _ in sizes_heads.values()]))

            summary_rows.append({
                "model": model_disp,
                "K": K,
                "P": args.P,
                "method": args.method,
                "jaccard_offdiag_all": jac_all_off,
                "jaccard_offdiag_heads": jac_heads_off,
                "overlap_coef_offdiag_all": oc_all_off,
                "overlap_coef_offdiag_heads": oc_heads_off,
                "shared_size_mean_all": mean_size_all,
                "shared_size_mean_heads": mean_size_heads,
            })

            for (t1, t2, v) in offdiag_unique(jac_all):
                pair_rows.append({"model": model_disp, "K": K, "P": args.P,
                                  "method": args.method, "granularity": "head+MLP",
                                  "t1": t1, "t2": t2, "jaccard": v})
            for (t1, t2, v) in offdiag_unique(jac_heads):
                pair_rows.append({"model": model_disp, "K": K, "P": args.P,
                                  "method": args.method, "granularity": "heads only",
                                  "t1": t1, "t2": t2, "jaccard": v})


    summary = pd.DataFrame(summary_rows)
    csv_path = OUT_DIR / f"summary_P{args.P}_{args.method}.csv"
    summary.to_csv(csv_path, index=False, float_format="%.3f")
    print(f"\nSaved summary to {csv_path}")

    pairs = pd.DataFrame(pair_rows)
    pairs_path = OUT_DIR / f"pairs_P{args.P}_{args.method}.csv"
    pairs.to_csv(pairs_path, index=False, float_format="%.4f")
    print(f"Saved per-pair values to {pairs_path}")

    pivot = summary.pivot_table(
        index="model",
        columns="K",
        values=["overlap_coef_offdiag_all", "overlap_coef_offdiag_heads"],
        aggfunc="first",
    )
    print("\nMean off-diagonal Overlap Coefficient (head+MLP vs heads-only):")
    print(pivot.round(2).to_string())

    pivot_jac = summary.pivot_table(
        index="model",
        columns="K",
        values=["jaccard_offdiag_all", "jaccard_offdiag_heads"],
        aggfunc="first",
    )
    print("\nMean off-diagonal Jaccard (head+MLP vs heads-only):")
    print(pivot_jac.round(2).to_string())


if __name__ == "__main__":
    main()
