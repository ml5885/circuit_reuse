"""Visualize layer-level distribution of top-K% circuit components.

Produces several plot variants showing where in the network (early/mid/late layers)
the most important components live, broken down by heads vs MLPs.

Plots:
  1. Stacked area: fraction of top-K% that comes from each layer third (early/mid/late)
  2. Layer line: per-K distribution over layers (cumulative, so y-axis is readable)
  3. Heatmap: normalized component density by layer and K% (compact)
  4. Head vs MLP inclusion rate: % of all heads / % of all MLPs in the circuit
  5. MLP vs Head fraction: stacked bars showing MLP/head split across K%
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from circuit_reuse.dataset import get_model_display_name, get_task_display_name

matplotlib.use("Agg")
plt.rcParams.update({"font.family": "serif", "font.size": 14})

SKIP_TASKS = ["mmlu"]

# Color palette
COLORS = {
    "early": "#6a4c93",   # muted purple
    "middle": "#1982c4",  # calm blue
    "late": "#8ac926",    # fresh green
    "head": "#f7a011",    # light orange (Asteroid City)
    "mlp": "#7EC8B8",     # muted teal-blue (Asteroid City)
}


def parse_attrib_filename(path: Path) -> dict:
    stem = path.stem
    parts = stem.split("__")
    return {
        "model_name": parts[0].replace("_", "/", 1) if "_" in parts[0] else parts[0],
        "task": parts[2],
    }


def load_attributions(path: Path) -> list[list[dict]]:
    examples = []
    with path.open() as f:
        for line in f:
            examples.append(json.loads(line)["components"])
    return examples


def compute_layer_fractions(
    examples: list[list[dict]], k_percents: list[float], n_layers: int,
) -> dict[float, dict[str, np.ndarray]]:
    """For each K%, compute fraction of top-K components at each layer, split by kind."""
    n_total = len(examples[0])
    n_examples = len(examples)
    result = {}

    for k_pct in k_percents:
        top_n = max(1, math.ceil(n_total * k_pct / 100))
        head_counts = np.zeros(n_layers)
        mlp_counts = np.zeros(n_layers)
        for ex in examples:
            for c in ex[:top_n]:
                if c["kind"] == "head":
                    head_counts[c["layer"]] += 1
                else:
                    mlp_counts[c["layer"]] += 1

        total = head_counts.sum() + mlp_counts.sum()
        if total > 0:
            head_counts /= total
            mlp_counts /= total

        result[k_pct] = {"head": head_counts, "mlp": mlp_counts}

    return result


def compute_layer_counts_unnorm(
    examples: list[list[dict]], k_percents: list[float], n_layers: int,
) -> dict[float, dict[str, np.ndarray]]:
    """Mean count of heads/MLPs per layer in top-K% circuit."""
    n_total = len(examples[0])
    n_examples = len(examples)
    result = {}

    for k_pct in k_percents:
        top_n = max(1, math.ceil(n_total * k_pct / 100))
        head_counts = np.zeros(n_layers)
        mlp_counts = np.zeros(n_layers)
        for ex in examples:
            for c in ex[:top_n]:
                if c["kind"] == "head":
                    head_counts[c["layer"]] += 1
                else:
                    mlp_counts[c["layer"]] += 1

        result[k_pct] = {"head": head_counts / n_examples, "mlp": mlp_counts / n_examples}

    return result


def get_component_totals(examples: list[list[dict]], n_layers: int) -> dict[str, int]:
    """Count total heads and MLPs in the model from the first example's full component list."""
    n_heads = sum(1 for c in examples[0] if c["kind"] == "head")
    n_mlps = sum(1 for c in examples[0] if c["kind"] == "mlp")
    return {"head": n_heads, "mlp": n_mlps}


# ---------------------------------------------------------------------------
# Plot 1: Stacked area — early/mid/late fraction vs K%
# ---------------------------------------------------------------------------
def plot_stacked_area(
    all_data: dict[str, dict[str, dict[float, dict[str, np.ndarray]]]],
    k_percents: list[float],
    n_layers_map: dict[str, int],
    output_dir: Path,
):
    out_dir = output_dir / "stacked_area"
    out_dir.mkdir(parents=True, exist_ok=True)

    for model, task_data in sorted(all_data.items()):
        tasks = sorted(task_data.keys())
        n_layers = n_layers_map[model]
        third = n_layers // 3
        bins = [
            ("Early", range(0, third)),
            ("Middle", range(third, 2 * third)),
            ("Late", range(2 * third, n_layers)),
        ]
        colors = [COLORS["early"], COLORS["middle"], COLORS["late"]]

        ncols = min(len(tasks), 3)
        nrows = math.ceil(len(tasks) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)

        for idx, task in enumerate(tasks):
            ax = axes[idx // ncols][idx % ncols]
            fracs = task_data[task]

            stacks = {name: [] for name, _ in bins}
            for k in k_percents:
                combined = fracs[k]["head"] + fracs[k]["mlp"]
                for name, layer_range in bins:
                    stacks[name].append(combined[list(layer_range)].sum())

            bottom = np.zeros(len(k_percents))
            for (name, _), color in zip(bins, colors):
                vals = np.array(stacks[name])
                ax.fill_between(k_percents, bottom, bottom + vals, label=name,
                                alpha=0.85, color=color, edgecolor="white", linewidth=0.5)
                bottom += vals

            ax.set_title(get_task_display_name(task), fontsize=16)
            ax.set_xlim(k_percents[0], k_percents[-1])
            ax.set_ylim(0, 1)
            if idx // ncols == nrows - 1:
                ax.set_xlabel("Top-K%", fontsize=14)
            if idx % ncols == 0:
                ax.set_ylabel("Fraction of Circuit", fontsize=14)
            ax.tick_params(labelsize=12)

        for k in range(len(tasks), nrows * ncols):
            fig.delaxes(axes[k // ncols][k % ncols])

        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=13,
                   bbox_to_anchor=(0.5, -0.06))
        model_disp = get_model_display_name(model)
        fig.tight_layout(rect=[0, 0.02, 1, 1.0])

        safe = model.replace("/", "_")
        out = out_dir / f"stacked_area_{safe}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[SAVED] {out}")


# ---------------------------------------------------------------------------
# Plot 2: Layer density heatmap — compact, one row of tasks per model
# ---------------------------------------------------------------------------
def plot_layer_heatmap(
    all_data: dict[str, dict[str, dict[float, dict[str, np.ndarray]]]],
    k_percents: list[float],
    n_layers_map: dict[str, int],
    output_dir: Path,
):
    """Per model: compact heatmap. Each task gets one column showing combined
    head+MLP density. Layers on y-axis, K% on x-axis."""
    out_dir = output_dir / "layer_heatmap"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmap = LinearSegmentedColormap.from_list("density", ["#f7f7f7", "#6a4c93"], N=256)

    for model, task_data in sorted(all_data.items()):
        tasks = sorted(task_data.keys())
        n_layers = n_layers_map[model]
        model_disp = get_model_display_name(model)

        n_tasks = len(tasks)
        fig_width = max(3.5 * n_tasks + 1.5, 10)
        fig_height = max(n_layers * 0.18 + 2, 5)
        fig, axes = plt.subplots(1, n_tasks, figsize=(fig_width, fig_height), squeeze=False)

        all_vmax = 0
        matrices = []
        for task in tasks:
            fracs = task_data[task]
            mat = np.zeros((n_layers, len(k_percents)))
            for ki, k in enumerate(k_percents):
                mat[:, ki] = fracs[k]["head"] + fracs[k]["mlp"]
            matrices.append(mat)
            all_vmax = max(all_vmax, mat.max())

        for idx, (task, mat) in enumerate(zip(tasks, matrices)):
            ax = axes[0][idx]
            ax.imshow(mat[::-1], aspect="auto", cmap=cmap, vmin=0, vmax=all_vmax,
                      interpolation="nearest")
            ax.set_title(get_task_display_name(task), fontsize=14)
            ax.set_xticks(range(len(k_percents)))
            ax.set_xticklabels([f"{k:g}%" for k in k_percents], fontsize=10, rotation=45, ha="right")

            step = max(1, n_layers // 8)
            tick_pos = list(range(0, n_layers, step))
            tick_labels = [str(n_layers - 1 - t) for t in tick_pos]
            ax.set_yticks(tick_pos)
            ax.set_yticklabels(tick_labels, fontsize=9)

            if idx == 0:
                ax.set_ylabel("Layer", fontsize=13)
            else:
                ax.tick_params(axis="y", labelleft=False)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, all_vmax))
        cbar = fig.colorbar(sm, ax=axes[0].tolist(), shrink=0.7, pad=0.02)
        cbar.set_label("Fraction of Circuit", fontsize=12)
        cbar.ax.tick_params(labelsize=9)

        fig.tight_layout(rect=[0, 0, 0.92, 1.0])

        safe = model.replace("/", "_")
        out = out_dir / f"layer_heatmap_{safe}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[SAVED] {out}")


# ---------------------------------------------------------------------------
# Plot 3: Layer line — cumulative fraction so y-axis is 0-1
# ---------------------------------------------------------------------------
def plot_layer_lines(
    all_data_counts: dict[str, dict[str, dict[float, dict[str, np.ndarray]]]],
    k_percents: list[float],
    n_layers_map: dict[str, int],
    output_dir: Path,
):
    """Per model: line plots showing cumulative fraction of circuit by layer."""
    out_dir = output_dir / "layer_line"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmap = plt.get_cmap("magma")

    for model, task_data in sorted(all_data_counts.items()):
        tasks = sorted(task_data.keys())
        n_layers = n_layers_map[model]
        model_disp = get_model_display_name(model)
        layers = np.arange(n_layers)

        k_colors = {k: cmap(0.2 + 0.7 * i / max(1, len(k_percents) - 1))
                    for i, k in enumerate(k_percents)}

        ncols = min(len(tasks), 3)
        nrows = math.ceil(len(tasks) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)

        for idx, task in enumerate(tasks):
            ax = axes[idx // ncols][idx % ncols]
            counts = task_data[task]

            for k in k_percents:
                combined = counts[k]["head"] + counts[k]["mlp"]
                total = combined.sum()
                if total > 0:
                    cumulative = np.cumsum(combined) / total
                else:
                    cumulative = np.zeros(n_layers)

                ax.plot(layers, cumulative, marker=".", markersize=3,
                        color=k_colors[k], label=f"{k:g}%", linewidth=1.8, alpha=0.85)

            ax.set_title(get_task_display_name(task), fontsize=28, pad=8)
            if idx // ncols == nrows - 1:
                ax.set_xlabel("Layer", fontsize=24)
            else:
                ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(labelsize=20)
            ax.set_xlim(0, n_layers - 1)
            ax.set_ylim(0, 1.05)
            ax.axhline(0.5, color="gray", linewidth=0.8, linestyle=":", alpha=0.5)

        for k in range(len(tasks), nrows * ncols):
            fig.delaxes(axes[k // ncols][k % ncols])

        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(k_percents), fontsize=22,
                   title="Top-K%", title_fontsize=24, bbox_to_anchor=(0.5, -0.12))
        fig.supylabel("Cumulative Fraction", fontsize=26, x=0.06)
        fig.tight_layout(rect=[0.06, 0.03, 1, 1.0])

        safe = model.replace("/", "_")
        out = out_dir / f"layer_line_{safe}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[SAVED] {out}")


# ---------------------------------------------------------------------------
# Plot 4: Head vs MLP inclusion rate — % of all heads / MLPs in circuit
# ---------------------------------------------------------------------------
def plot_head_mlp_inclusion(
    all_data_counts: dict[str, dict[str, dict[float, dict[str, np.ndarray]]]],
    k_percents: list[float],
    component_totals: dict[str, dict[str, int]],
    output_dir: Path,
):
    """Per model: grouped bars showing what % of all heads and % of all MLPs
    are in the top-K% circuit (averaged over examples)."""
    out_dir = output_dir / "head_mlp_inclusion"
    out_dir.mkdir(parents=True, exist_ok=True)

    for model, task_data in sorted(all_data_counts.items()):
        tasks = sorted(task_data.keys())
        model_disp = get_model_display_name(model)
        totals = component_totals[model]

        ncols = min(len(tasks), 3)
        nrows = math.ceil(len(tasks) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.0 * nrows), squeeze=False)

        for idx, task in enumerate(tasks):
            ax = axes[idx // ncols][idx % ncols]
            counts = task_data[task]

            head_pcts = []
            mlp_pcts = []
            for k in k_percents:
                h = counts[k]["head"].sum()
                m = counts[k]["mlp"].sum()
                head_pcts.append(100 * h / totals["head"] if totals["head"] > 0 else 0)
                mlp_pcts.append(100 * m / totals["mlp"] if totals["mlp"] > 0 else 0)

            x = np.arange(len(k_percents))
            w = 0.35
            ax.bar(x - w / 2, head_pcts, w, color=COLORS["head"], label="Attention Heads",
                   edgecolor="white", linewidth=0.5, alpha=0.85)
            ax.bar(x + w / 2, mlp_pcts, w, color=COLORS["mlp"], label="MLPs",
                   edgecolor="white", linewidth=0.5, alpha=0.85)

            ax.set_xticks(x)
            ax.set_xticklabels([f"{k:g}%" for k in k_percents], fontsize=14)
            ax.set_title(get_task_display_name(task), fontsize=19, pad=4)
            if idx // ncols == nrows - 1:
                ax.set_xlabel("Top-K%", fontsize=15)
            ax.tick_params(labelsize=14)

        for k in range(len(tasks), nrows * ncols):
            fig.delaxes(axes[k // ncols][k % ncols])

        fig.supylabel("% Included in Circuit", fontsize=20, x=0.06, y=0.52)

        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=17,
                   bbox_to_anchor=(0.52, -0.06))
        fig.tight_layout(rect=[0.04, 0.02, 1, 1.0])

        safe = model.replace("/", "_")
        out = out_dir / f"head_mlp_inclusion_{safe}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[SAVED] {out}")


# ---------------------------------------------------------------------------
# Plot 5: MLP vs Head fraction — stacked bars across K%
# ---------------------------------------------------------------------------
def plot_mlp_head_fraction(
    all_data_counts: dict[str, dict[str, dict[float, dict[str, np.ndarray]]]],
    k_percents: list[float],
    output_dir: Path,
):
    """Per model: stacked bars showing fraction of circuit that is MLP vs heads,
    sweeping across top-K%."""
    out_dir = output_dir / "mlp_head_fraction"
    out_dir.mkdir(parents=True, exist_ok=True)

    for model, task_data in sorted(all_data_counts.items()):
        tasks = sorted(task_data.keys())
        model_disp = get_model_display_name(model)

        ncols = min(len(tasks), 3)
        nrows = math.ceil(len(tasks) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)

        for idx, task in enumerate(tasks):
            ax = axes[idx // ncols][idx % ncols]
            counts = task_data[task]

            mlp_fracs = []
            head_fracs = []
            for k in k_percents:
                h = counts[k]["head"].sum()
                m = counts[k]["mlp"].sum()
                total = h + m
                if total > 0:
                    mlp_fracs.append(m / total)
                    head_fracs.append(h / total)
                else:
                    mlp_fracs.append(0)
                    head_fracs.append(0)

            x = np.arange(len(k_percents))
            ax.bar(x, mlp_fracs, color=COLORS["mlp"], label="MLPs",
                   edgecolor="white", linewidth=0.5, alpha=0.85)
            ax.bar(x, head_fracs, bottom=mlp_fracs, color=COLORS["head"], label="Attention Heads",
                   edgecolor="white", linewidth=0.5, alpha=0.85)

            ax.set_xticks(x)
            ax.set_xticklabels([f"{k:g}%" for k in k_percents], fontsize=12)
            ax.set_ylim(0, 1.05)
            ax.set_title(get_task_display_name(task), fontsize=15)
            if idx // ncols == nrows - 1:
                ax.set_xlabel("Top-K%", fontsize=13)
            if idx % ncols == 0:
                ax.set_ylabel("Share of Circuit Components", fontsize=13)
            ax.tick_params(labelsize=12)

        for k in range(len(tasks), nrows * ncols):
            fig.delaxes(axes[k // ncols][k % ncols])

        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=13,
                   bbox_to_anchor=(0.5, -0.04))
        fig.suptitle(model_disp, fontsize=17, y=1.02)
        fig.tight_layout(rect=[0, 0.02, 1, 1.0])

        safe = model.replace("/", "_")
        out = out_dir / f"mlp_head_fraction_{safe}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[SAVED] {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=str, default="cache")
    p.add_argument("--output-dir", type=str, default="results2/layer_distribution")
    p.add_argument("--k-percents", type=str, default="1,5,10,20,30")
    p.add_argument("--exclude-tasks", type=str, default="mmlu")
    args = p.parse_args()

    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    k_percents = [float(x) for x in args.k_percents.split(",")]
    exclude = set(args.exclude_tasks.split(",")) if args.exclude_tasks else set()

    all_frac: dict[str, dict[str, dict]] = defaultdict(dict)
    all_counts: dict[str, dict[str, dict]] = defaultdict(dict)
    n_layers_map: dict[str, int] = {}
    component_totals: dict[str, dict[str, int]] = {}

    for path in sorted(cache_dir.glob("*.jsonl")):
        info = parse_attrib_filename(path)
        model, task = info["model_name"], info["task"]
        if task in exclude:
            continue

        print(f"[LOAD] {model} / {task}")
        examples = load_attributions(path)
        if not examples:
            continue

        n_layers = max(c["layer"] for c in examples[0]) + 1
        n_layers_map[model] = n_layers

        if model not in component_totals:
            component_totals[model] = get_component_totals(examples, n_layers)

        all_frac[model][task] = compute_layer_fractions(examples, k_percents, n_layers)
        all_counts[model][task] = compute_layer_counts_unnorm(examples, k_percents, n_layers)

    print(f"\nLoaded {sum(len(v) for v in all_frac.values())} model/task combos")
    for model, totals in sorted(component_totals.items()):
        print(f"  {get_model_display_name(model)}: {totals['head']} heads, {totals['mlp']} MLPs")

    plot_stacked_area(all_frac, k_percents, n_layers_map, output_dir)
    plot_layer_heatmap(all_frac, k_percents, n_layers_map, output_dir)
    plot_layer_lines(all_counts, k_percents, n_layers_map, output_dir)
    plot_head_mlp_inclusion(all_counts, k_percents, component_totals, output_dir)
    plot_mlp_head_fraction(all_counts, k_percents, output_dir)

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
