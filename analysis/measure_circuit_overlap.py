"""Measure pairwise overlap between task circuits and produce heatmaps."""

import argparse
import json
import itertools
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use("Agg")

TASK_DISPLAY = {
    "addition": "Addition",
    "arc_challenge": "ARC (Challenge)",
    "arc_easy": "ARC (Easy)",
    "boolean": "Boolean",
    "ioi": "IOI",
    "mcqa": "Colored Objects MCQA",
    "mmlu": "MMLU",
}

MODEL_DISPLAY = {
    "google/gemma-2-2b-it": "Google/Gemma 2 2B it",
    "google/gemma-2-2b": "Google/Gemma 2 2B",
    "meta-llama/Llama-3.2-3B-Instruct": "Llama-3.2-3B Instruct",
    "meta-llama/Llama-3.2-3B": "Llama-3.2-3B",
    "qwen3-4b": "Qwen3-4B",
    "qwen3-8b": "Qwen3-8B",
}


def load_circuits(results_dir: Path, tasks: list[str], K: int, threshold: int):
    """Load shared_components for each task from metrics.json files.

    Returns dict mapping task -> set of component strings.
    """
    circuits = {}
    for run_dir in results_dir.iterdir():
        if not run_dir.is_dir():
            continue
        metrics_file = run_dir / "metrics.json"
        if not metrics_file.exists():
            continue
        with open(metrics_file) as f:
            metrics = json.load(f)
        task = metrics["task"]
        if task not in tasks:
            continue
        k_data = metrics.get("by_k", {}).get(str(K))
        if k_data is None:
            continue
        t_data = k_data.get("thresholds", {}).get(str(threshold))
        if t_data is None:
            continue
        circuits[task] = set(t_data["shared_components"])
    return circuits


def compute_overlap_metrics(circuits: dict[str, set], tasks: list[str]):
    """Compute pairwise overlap metrics between all task circuits.

    Returns a dict of metric_name -> DataFrame (tasks x tasks).
    """
    n = len(tasks)
    jaccard = np.zeros((n, n))
    overlap_coeff = np.zeros((n, n))
    intersection_frac = np.zeros((n, n))  # |A ∩ B| / |A| (row-normalized)
    intersection_size = np.zeros((n, n))

    for i, t1 in enumerate(tasks):
        for j, t2 in enumerate(tasks):
            if t1 not in circuits or t2 not in circuits:
                jaccard[i, j] = overlap_coeff[i, j] = np.nan
                intersection_frac[i, j] = intersection_size[i, j] = np.nan
                continue
            a, b = circuits[t1], circuits[t2]
            inter = len(a & b)
            union = len(a | b)
            jaccard[i, j] = inter / union if union else 0
            overlap_coeff[i, j] = inter / min(len(a), len(b)) if min(len(a), len(b)) else 0
            intersection_frac[i, j] = inter / len(a) if len(a) else 0
            intersection_size[i, j] = inter

    labels = [TASK_DISPLAY.get(t, t) for t in tasks]
    return {
        "jaccard": pd.DataFrame(jaccard, index=labels, columns=labels),
        "overlap_coefficient": pd.DataFrame(overlap_coeff, index=labels, columns=labels),
        "containment": pd.DataFrame(intersection_frac, index=labels, columns=labels),
        "intersection_size": pd.DataFrame(intersection_size, index=labels, columns=labels),
    }


def plot_heatmap(df: pd.DataFrame, title: str, out_path: Path, fmt: str = ".2f",
                 vmin: float = 0, vmax: float = 1, cmap: str = "YlOrRd"):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(df.values, vmin=vmin, vmax=vmax, cmap=cmap, aspect="equal")
    ax.set_xticks(range(len(df.columns)))
    ax.set_yticks(range(len(df.index)))
    ax.set_xticklabels(df.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(df.index, fontsize=9)
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            v = df.values[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:{fmt}}", ha="center", va="center", fontsize=8,
                    color="white" if v > (vmax - vmin) * 0.65 + vmin else "black")
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel("Task B", fontsize=10)
    ax.set_ylabel("Task A", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_multimodel_heatmap(all_dfs: dict[str, pd.DataFrame], metric_name: str,
                            out_path: Path, fmt: str = ".2f",
                            vmin: float = 0, vmax: float = 1, cmap: str = "YlOrRd"):
    """Plot a grid of heatmaps, one per model."""
    models = list(all_dfs.keys())
    n = len(models)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows), squeeze=False)

    for idx, model in enumerate(models):
        ax = axes[idx // ncols][idx % ncols]
        df = all_dfs[model]
        im = ax.imshow(df.values, vmin=vmin, vmax=vmax, cmap=cmap, aspect="equal")
        ax.set_xticks(range(len(df.columns)))
        ax.set_yticks(range(len(df.index)))
        ax.set_xticklabels(df.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(df.index, fontsize=8)
        for i in range(len(df.index)):
            for j in range(len(df.columns)):
                v = df.values[i, j]
                if np.isnan(v):
                    continue
                ax.text(j, i, f"{v:{fmt}}", ha="center", va="center", fontsize=7,
                        color="white" if v > (vmax - vmin) * 0.65 + vmin else "black")
        ax.set_title(MODEL_DISPLAY.get(model, model), fontsize=11)

    # hide unused axes
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(f"Circuit overlap — {metric_name}", fontsize=14, y=1.01)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label=metric_name)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_circuit_sizes(all_sizes: dict[str, dict[str, int]], out_path: Path):
    """Bar chart of circuit sizes per model and task."""
    df = pd.DataFrame(all_sizes).T
    df.columns = [TASK_DISPLAY.get(c, c) for c in df.columns]
    ax = df.plot.bar(figsize=(10, 5), rot=30)
    ax.set_ylabel("Number of components")
    ax.set_title("Shared circuit sizes")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def discover_models(results_dir: Path) -> list[str]:
    """Find all unique model names from metrics.json files."""
    models = set()
    for run_dir in results_dir.iterdir():
        metrics_file = run_dir / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                models.add(json.load(f)["model_name"])
    return sorted(models)


def get_model_results_dir(results_dir: Path, model_name: str) -> Path:
    """Return a filtered view by yielding only dirs matching this model."""
    # We just use results_dir directly; load_circuits filters by task.
    # But we need to filter by model. We'll create a helper.
    return results_dir


def load_circuits_for_model(results_dir: Path, model_name: str,
                            tasks: list[str], K: int, threshold: int):
    circuits = {}
    for run_dir in results_dir.iterdir():
        if not run_dir.is_dir():
            continue
        metrics_file = run_dir / "metrics.json"
        if not metrics_file.exists():
            continue
        with open(metrics_file) as f:
            metrics = json.load(f)
        if metrics["model_name"] != model_name:
            continue
        task = metrics["task"]
        if task not in tasks:
            continue
        k_data = metrics.get("by_k", {}).get(str(K))
        if k_data is None:
            continue
        t_data = k_data.get("thresholds", {}).get(str(threshold))
        if t_data is None:
            continue
        circuits[task] = set(t_data["shared_components"])
    return circuits


def main():
    parser = argparse.ArgumentParser(description="Measure circuit overlap across tasks")
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="Directory containing run subdirectories with metrics.json")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory to save CSVs and plots")
    parser.add_argument("--tasks", type=str,
                        default="addition,arc_challenge,arc_easy,boolean,ioi,mcqa",
                        help="Comma-separated task names")
    parser.add_argument("--K", type=int, nargs="+", default=[50, 100],
                        help="Top-K percentages to analyze")
    parser.add_argument("--threshold", type=int, default=100,
                        help="Reuse threshold percentage")
    args = parser.parse_args()

    tasks = args.tasks.split(",")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    models = discover_models(args.results_dir)
    print(f"Found models: {models}")
    print(f"Tasks: {tasks}")
    print(f"K values: {args.K}, threshold: {args.threshold}")

    for K in args.K:
        print(f"\n{'='*60}")
        print(f"K = {K}")
        print(f"{'='*60}")

        all_metrics: dict[str, dict[str, pd.DataFrame]] = {}  # model -> metric -> df
        all_sizes: dict[str, dict[str, int]] = {}

        for model in models:
            circuits = load_circuits_for_model(args.results_dir, model, tasks, K, args.threshold)
            available = [t for t in tasks if t in circuits]
            if len(available) < 2:
                print(f"  {model}: only {len(available)} tasks found, skipping")
                continue

            print(f"  {model}: {len(available)} tasks, sizes = "
                  + ", ".join(f"{t}={len(circuits[t])}" for t in available))

            metrics = compute_overlap_metrics(circuits, tasks)
            all_metrics[model] = metrics

            model_label = MODEL_DISPLAY.get(model, model)
            all_sizes[model_label] = {t: len(circuits[t]) for t in tasks if t in circuits}

            # Per-model CSVs
            model_slug = model.replace("/", "_")
            for metric_name, df in metrics.items():
                csv_path = args.output_dir / f"{metric_name}_{model_slug}_K{K}_t{args.threshold}.csv"
                df.to_csv(csv_path)

                plot_path = args.output_dir / f"{metric_name}_{model_slug}_K{K}_t{args.threshold}.png"
                vmax = 1.0 if metric_name != "intersection_size" else df.values[~np.isnan(df.values)].max()
                fmt = ".2f" if metric_name != "intersection_size" else ".0f"
                plot_heatmap(df, f"{metric_name} — {MODEL_DISPLAY.get(model, model)} (K={K})",
                             plot_path, fmt=fmt, vmax=vmax)

        # Multi-model grid plots
        if all_metrics:
            for metric_name in ["jaccard", "overlap_coefficient", "containment"]:
                model_dfs = {m: all_metrics[m][metric_name] for m in all_metrics}
                out = args.output_dir / f"multimodel_{metric_name}_K{K}_t{args.threshold}.png"
                plot_multimodel_heatmap(model_dfs, metric_name, out)

            # Circuit sizes bar chart
            if all_sizes:
                plot_circuit_sizes(all_sizes, args.output_dir / f"circuit_sizes_K{K}_t{args.threshold}.png")

        # Summary CSV: one row per (model, task_a, task_b) with all metrics
        rows = []
        for model in all_metrics:
            for metric_name, df in all_metrics[model].items():
                for i, t1 in enumerate(df.index):
                    for j, t2 in enumerate(df.columns):
                        rows.append({
                            "model": model,
                            "task_a": t1,
                            "task_b": t2,
                            "metric": metric_name,
                            "value": df.values[i, j],
                            "K": K,
                            "threshold": args.threshold,
                        })
        if rows:
            summary = pd.DataFrame(rows)
            summary.to_csv(args.output_dir / f"circuit_overlap_summary_K{K}_t{args.threshold}.csv",
                           index=False)
            print(f"\n  Summary saved to circuit_overlap_summary_K{K}_t{args.threshold}.csv")

    print("\nDone!")


if __name__ == "__main__":
    main()
