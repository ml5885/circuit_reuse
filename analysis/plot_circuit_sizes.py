"""Plot shared circuit sizes as a function of top-K, per model and task."""

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

from circuit_reuse.dataset import get_task_display_name, get_model_display_name


def load_circuit_sizes(results_dir: Path):
    """Load circuit sizes from all metrics.json files.

    Returns dict: model -> task -> K -> circuit_size
    """
    data = {}
    for run_dir in results_dir.iterdir():
        if not run_dir.is_dir():
            continue
        metrics_file = run_dir / "metrics.json"
        if not metrics_file.exists():
            continue
        with open(metrics_file) as f:
            metrics = json.load(f)
        model = metrics["model_name"]
        task = metrics["task"]
        by_k = metrics.get("by_k", {})
        for k_str, k_data in by_k.items():
            K = int(k_str)
            for t_str, t_data in k_data.get("thresholds", {}).items():
                threshold = int(t_str)
                if threshold != 100:
                    continue
                size = t_data.get("shared_circuit_size", len(t_data.get("shared_components", [])))
                data.setdefault(model, {}).setdefault(task, {})[K] = size
    return data


def plot_sizes_per_model(data: dict, output_dir: Path):
    """One subplot per model: line plot of circuit size vs K, one line per task."""
    models = sorted(data.keys(), key=lambda m: get_model_display_name(m))
    n = len(models)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), squeeze=False)

    for idx, model in enumerate(models):
        ax = axes[idx // ncols][idx % ncols]
        tasks = sorted(data[model].keys())
        for task in tasks:
            ks = sorted(data[model][task].keys())
            sizes = [data[model][task][k] for k in ks]
            ax.plot(ks, sizes, "o-", label=get_task_display_name(task), markersize=5)
        ax.set_xlabel("Top-K (%)")
        ax.set_ylabel("Shared circuit size")
        ax.set_title(get_model_display_name(model), fontsize=11)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("Shared circuit size (reuse@100) vs Top-K", fontsize=13, y=1.01)
    fig.tight_layout()
    out = output_dir / "circuit_sizes_vs_k.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_sizes_per_task(data: dict, output_dir: Path):
    """One subplot per task: line plot of circuit size vs K, one line per model."""
    all_tasks = sorted({t for m in data for t in data[m]})
    n = len(all_tasks)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), squeeze=False)

    models = sorted(data.keys(), key=lambda m: get_model_display_name(m))
    for idx, task in enumerate(all_tasks):
        ax = axes[idx // ncols][idx % ncols]
        for model in models:
            if task not in data[model]:
                continue
            ks = sorted(data[model][task].keys())
            sizes = [data[model][task][k] for k in ks]
            ax.plot(ks, sizes, "o-", label=get_model_display_name(model), markersize=5)
        ax.set_xlabel("Top-K (%)")
        ax.set_ylabel("Shared circuit size")
        ax.set_title(get_task_display_name(task), fontsize=11)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("Shared circuit size (reuse@100) vs Top-K", fontsize=13, y=1.01)
    fig.tight_layout()
    out = output_dir / "circuit_sizes_vs_k_by_task.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_sizes_bar_per_k(data: dict, output_dir: Path):
    """For each K value, a grouped bar chart: models on x-axis, bars per task."""
    all_ks = sorted({k for m in data for t in data[m] for k in data[m][t]})
    all_tasks = sorted({t for m in data for t in data[m]})
    models = sorted(data.keys(), key=lambda m: get_model_display_name(m))
    model_labels = [get_model_display_name(m) for m in models]

    n = len(all_ks)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), squeeze=False)

    x = np.arange(len(models))
    width = 0.8 / len(all_tasks)

    for idx, K in enumerate(all_ks):
        ax = axes[idx // ncols][idx % ncols]
        for ti, task in enumerate(all_tasks):
            sizes = [data[m].get(task, {}).get(K, 0) for m in models]
            ax.bar(x + ti * width - 0.4 + width / 2, sizes, width,
                   label=get_task_display_name(task))
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Shared circuit size")
        ax.set_title(f"K = {K}%", fontsize=11)
        ax.legend(fontsize=6, loc="upper left")
        ax.grid(True, alpha=0.3, axis="y")

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("Shared circuit size (reuse@100) by model and task", fontsize=13, y=1.01)
    fig.tight_layout()
    out = output_dir / "circuit_sizes_bars.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="Directory containing run subdirectories with metrics.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_circuit_sizes(args.results_dir)
    if not data:
        print(f"No metrics.json files found in {args.results_dir}")
        return

    print(f"Found {len(data)} models")
    for model in sorted(data):
        tasks = data[model]
        print(f"  {get_model_display_name(model)}: {', '.join(sorted(tasks))}")

    plot_sizes_per_model(data, args.output_dir)
    plot_sizes_per_task(data, args.output_dir)
    plot_sizes_bar_per_k(data, args.output_dir)


if __name__ == "__main__":
    main()
