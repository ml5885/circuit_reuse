"""Donor-donor circuit overlap, corrected for the size of the component pool.

Raw Jaccard is not comparable across granularities: a neuron basis offers three
orders of magnitude more components than a head/MLP basis, so two unrelated
circuits collide far less often by chance alone. This script reports observed
overlap alongside the overlap expected from two independent uniform draws of the
same sizes, and their ratio.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import tempfile
from pathlib import Path

_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "circuit_reuse_mplconfig"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd

from analysis.granularity_parity import CONFIG, EXCLUDED_MODELS, EXCLUDED_TASKS, infer_granularity

HF_IDS = {"google/gemma-2-2b": "google/gemma-2-2b",
          "google/gemma-2-2b-it": "google/gemma-2-2b-it",
          "meta-llama/Llama-3.2-3B": "meta-llama/Llama-3.2-3B",
          "meta-llama/Llama-3.2-3B-Instruct": "meta-llama/Llama-3.2-3B-Instruct",
          "qwen3-4b": "Qwen/Qwen3-4B",
          "qwen3-8b": "Qwen/Qwen3-8B"}


def component_pools(models: list[str], cache: Path) -> dict[str, dict[str, int]]:
    """Number of candidate components per model at each granularity."""
    known = json.loads(cache.read_text()) if cache.exists() else {}
    missing = [m for m in models if m not in known]
    if missing:
        from transformers import AutoConfig
        for model in missing:
            cfg = AutoConfig.from_pretrained(HF_IDS.get(model, model))
            layers, heads = cfg.num_hidden_layers, cfg.num_attention_heads
            known[model] = {"head_mlp": layers * (heads + 1),
                            "neuron": layers * cfg.intermediate_size}
        cache.write_text(json.dumps(known, indent=2))
    return known


def read_circuits(root: Path, K: int, p: int) -> pd.DataFrame:
    rows = []
    for path in sorted(root.rglob("metrics.json")):
        data = json.loads(path.read_text())
        model, task = data.get("model_name"), data.get("task")
        if any(x in str(model) for x in EXCLUDED_MODELS) or task in EXCLUDED_TASKS:
            continue
        cell = data.get("by_k", {}).get(str(K), {}).get("thresholds", {}).get(str(p))
        if not cell:
            continue
        rows.append({"model": model, "task": task, "method": data.get("method", "eap"),
                     "granularity": infer_granularity(data, path),
                     "components": frozenset(cell.get("shared_components") or [])})
    return pd.DataFrame(rows)


def pairwise_overlap(circuits: pd.DataFrame, pools: dict) -> pd.DataFrame:
    rows = []
    for (model, method, gran), group in circuits.groupby(["model"] + CONFIG):
        pool = pools[model][gran]
        sets = {r.task: r.components for r in group.itertuples()}
        for a, b in itertools.combinations(sorted(sets), 2):
            A, B = sets[a], sets[b]
            if not (A or B):
                continue
            # Expected intersection of two independent uniform draws is
            # |A||B|/N, and the union follows by inclusion-exclusion.
            expected_int = len(A) * len(B) / pool
            expected_union = len(A) + len(B) - expected_int
            rows.append({"model": model, "method": method, "granularity": gran,
                         "task_a": a, "task_b": b, "pool": pool,
                         "size_a": len(A), "size_b": len(B),
                         "observed": len(A & B) / len(A | B),
                         "chance": expected_int / expected_union if expected_union else np.nan})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["ratio_to_chance"] = out.observed / out.chance
    return out


def make_figure(pairs: pd.DataFrame, out: Path):
    import matplotlib.pyplot as plt
    from analysis.granularity_parity import (CONFIG_ORDER, _color,
                                             _fill_kw, _label, _legend, _panel,
                                             _ramp, _task_label)
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                         "axes.titlesize": 11, "axes.labelsize": 10,
                         "xtick.labelsize": 9, "ytick.labelsize": 9,
                         "legend.fontsize": 9, "figure.titlesize": 12})

    by_config = pairs.groupby(CONFIG).agg(
        observed=("observed", "mean"), chance=("chance", "mean")).reset_index()
    by_config["ratio"] = by_config.observed / by_config.chance
    order = [k for k in CONFIG_ORDER if k in set(map(tuple, by_config[CONFIG].values))]
    by_config = by_config.set_index(CONFIG).reindex(order).reset_index()
    names = [_label(k) for k in order]
    colors = [_color(k) for k in order]
    x = np.arange(len(order))

    def style(ax):
        ax.grid(alpha=.25, linewidth=.6, axis="y")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    # Observed against chance. Bars on a log axis have no meaningful baseline,
    # so draw each pair as a stem from chance up to the observed value.
    ax = axes[0]
    for xi, (obs, ch, c) in enumerate(zip(by_config.observed, by_config.chance, colors)):
        ax.plot([xi, xi], [ch, obs], color=c, lw=2.5, solid_capstyle="round", zorder=2)
        ax.scatter([xi], [ch], marker="_", s=260, color="0.35", zorder=3, linewidth=2)
        ax.scatter([xi], [obs], s=70, color=c, zorder=4, edgecolor="white", linewidth=.8)
        ax.annotate(f"{obs / ch:.0f}x", (xi, (obs * ch) ** .5), textcoords="offset points",
                    xytext=(9, -3), fontsize=9, color=c, fontweight="bold")
    ax.set(ylabel="Cross-task Jaccard overlap", yscale="log")
    ax.scatter([], [], marker="_", s=260, color="0.35", linewidth=2, label="chance baseline")
    ax.scatter([], [], s=70, color="0.35", label="observed")
    _legend(ax, loc="lower left")
    style(ax)

    ax = axes[1]
    for xi, (key, r) in enumerate(zip(order, by_config.ratio)):
        ax.bar([xi], [r], .55, **_fill_kw(key))
    ax.axhline(1, ls="--", c="k", lw=1)
    ax.text(len(order) - .45, 1, " chance", va="center", fontsize=8, color="0.35")
    for xi, r in enumerate(by_config.ratio):
        ax.annotate(f"{r:.0f}x", (xi, r), ha="center", va="bottom",
                    textcoords="offset points", xytext=(0, 3), fontsize=9)
    ax.set(ylabel="Observed / chance")
    style(ax)

    for ax in axes:
        ax.set_xticks(x, names, rotation=12, ha="right")
    fig.tight_layout()
    fig.savefig(out / "overlap_vs_chance.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Task-pair overlap matrices. A basis that discriminates should be dark
    # except where tasks are genuinely related.
    tasks = sorted(set(pairs.task_a) | set(pairs.task_b))
    labels = [_task_label(t) for t in tasks]
    fig, axes = plt.subplots(1, len(order), figsize=(3.7 * len(order), 4.0), squeeze=False)
    for i, (ax, key) in enumerate(zip(axes[0], order)):
        sub = pairs[(pairs.method == key[0]) & (pairs.granularity == key[1])]
        grid = np.full((len(tasks), len(tasks)), np.nan)
        for (a, b), v in sub.groupby(["task_a", "task_b"])["observed"].mean().items():
            grid[tasks.index(a)][tasks.index(b)] = v
            grid[tasks.index(b)][tasks.index(a)] = v
        # The shared 0-1 scale is what makes the granularities comparable, but it
        # also flattens the neuron panels, so every cell carries its value. Blank
        # cells stay white and so remain distinct from a genuine zero.
        im = ax.imshow(grid, cmap=_ramp(), vmin=0, vmax=1)
        for r in range(len(tasks)):
            for c in range(len(tasks)):
                if np.isnan(grid[r][c]):
                    continue
                ax.text(c, r, f"{grid[r][c]:.2f}".lstrip("0"), ha="center",
                        va="center", fontsize=6.5,
                        color="white" if grid[r][c] > .55 else "0.15")
        ax.set_xticks(range(len(tasks)), labels, rotation=45, ha="right", fontsize=7.5)
        # Only the leftmost panel carries y labels; repeating them crowds the row.
        if i == 0:
            ax.set_yticks(range(len(tasks)), labels, fontsize=7.5)
        else:
            ax.set_yticks(range(len(tasks)), [""] * len(tasks))
        _panel(ax, _label(key))
    fig.colorbar(im, ax=axes[0].tolist(), label="Jaccard overlap", fraction=.022)
    fig.savefig(out / "overlap_task_pairs.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root",
                    help="raw extraction root; omit with --replot")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--p", type=int, default=50)
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--replot", action="store_true",
                    help="regenerate figures from overlap_pairs.csv already in "
                         "--output-dir, without re-reading raw circuits")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.replot:
        # overlap_pairs.csv holds one row per model and task pair, which is all
        # the figures need once the component sets themselves are gone.
        pairs = pd.read_csv(out / "overlap_pairs.csv")
        make_figure(pairs, out)
        print(f"replotted from overlap_pairs.csv ({len(pairs)} pairs)")
        return

    if not args.results_root:
        ap.error("--results-root is required unless --replot is given")
    circuits = read_circuits(Path(args.results_root), args.K, args.p)
    pools = component_pools(sorted(circuits.model.unique()), out / "component_pools.json")
    pairs = pairwise_overlap(circuits, pools)
    pairs.to_csv(out / "overlap_pairs.csv", index=False)

    # Report the ratio of mean overlap to mean chance. Averaging per-pair ratios
    # instead lets pairs with a near-zero chance baseline dominate the mean.
    sizes = circuits.assign(size=circuits.components.map(len)).groupby(CONFIG)["size"].mean()
    summary = pairs.groupby(CONFIG).agg(
        pool=("pool", "mean"),
        observed_jaccard=("observed", "mean"), chance_jaccard=("chance", "mean"),
        median_pair_ratio=("ratio_to_chance", "median"), n_pairs=("observed", "size")).reset_index()
    summary["mean_circuit_size"] = summary.set_index(CONFIG).index.map(sizes)
    summary["ratio_to_chance"] = summary.observed_jaccard / summary.chance_jaccard
    summary.to_csv(out / "overlap_summary.csv", index=False)
    per_model = pairs.groupby(CONFIG + ["model"]).agg(
        observed_jaccard=("observed", "mean"), chance_jaccard=("chance", "mean")).reset_index()
    per_model["ratio_to_chance"] = per_model.observed_jaccard / per_model.chance_jaccard
    per_model.to_csv(out / "overlap_by_model.csv", index=False)

    if args.plots:
        make_figure(pairs, out)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
