import argparse
import json
from pathlib import Path
import math

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from circuit_reuse.dataset import get_task_display_name, get_model_display_name


def discover_metrics(results_dir):
    if not results_dir.exists():
        return []
    return sorted(results_dir.rglob("metrics.json"))


def load_metrics_json(path):
    with path.open("r") as f:
        return json.load(f)


def _expand_v2(r):
    rows = []
    base = {
        "model_name": r["model_name"],
        "hf_revision": r["hf_revision"],
        "task": r["task"],
        "method": r["method"],
    }

    by_k = r["by_k"]
    for k_str, block in by_k.items():
        K = int(k_str)

        thresholds = block["thresholds"]
        for p_str, tblock in thresholds.items():
            P = int(p_str)

            tr = tblock["train"]
            va = tblock["val"]

            def compute_lift_val(ablation, control, baseline):
                if baseline == 0:
                    return float("nan")
                return (control - ablation) / baseline

            lift_train = compute_lift_val(
                tr["ablation_accuracy"],
                tr["control_accuracy"],
                r["baseline_train_accuracy"],
            )
            lift_val = compute_lift_val(
                va["ablation_accuracy"],
                va["control_accuracy"],
                r["baseline_val_accuracy"],
            )

            row = dict(base)
            row.update(
                {
                    "top_k": K,
                    "reuse_threshold": P,
                    "reuse_percent": tblock["reuse_percent"],
                    "lift_train": lift_train,
                    "lift_val": lift_val,
                }
            )
            rows.append(row)
    return rows


def aggregate(paths):
    expanded = []
    for p in paths:
        r = load_metrics_json(p)
        if "by_k" in r:
            expanded.extend(_expand_v2(r))
    return pd.DataFrame(expanded) if expanded else pd.DataFrame()


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _save_task_grid(
    df,
    tasks,
    y_col,
    y_label,
    title_prefix,
    out_path,
    threshold,
    nrows=2,
    ncols=3,
):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 10), squeeze=False)
    axes = axes.flatten()

    legend_handles = None
    legend_labels = None

    for ax, task in zip(axes, tasks):
        task_df = df[df["task_display"] == task]

        sns.lineplot(
            data=task_df,
            x="top_k",
            y=y_col,
            hue="model_display",
            marker="o",
            ax=ax,
            legend=True,
        )

        ax.set_title(task)
        ax.set_xlabel("Top-K Components")
        ax.set_ylabel(y_label)

        if y_col == "lift_val":
            ax.axhline(0, color="gray", linestyle="--", linewidth=1)

        handles, labels = ax.get_legend_handles_labels()
        if handles and labels:
            legend_handles = handles
            legend_labels = labels

        if ax.get_legend() is not None:
            ax.get_legend().remove()

    for ax in axes[len(tasks) :]:
        ax.set_visible(False)

    fig.suptitle("")

    if legend_handles and legend_labels:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=min(len(legend_labels), 4),
            bbox_to_anchor=(0.5, 0.98),
            frameon=True,
        )

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved {out_path}")


def plot_k_sweep(df, out_dir, threshold=100):
    df = df[df["reuse_threshold"] == threshold].copy()
    if df.empty:
        print(f"[WARN] No data for threshold {threshold}")
        return

    df["task_display"] = df["task"].apply(get_task_display_name)
    df["model_display"] = df["model_name"].apply(get_model_display_name)

    # Styling
    plt.rcParams["font.family"] = "serif"
    sns.set_theme(style="whitegrid")

    tasks = sorted(df["task_display"].unique())
    tasks_per_figure = 6
    num_pages = math.ceil(len(tasks) / tasks_per_figure)

    for page_idx, task_chunk in enumerate(chunked(tasks, tasks_per_figure), start=1):
        page_suffix = f"_page{page_idx}" if num_pages > 1 else ""

        lift_out_path = out_dir / f"k_sweep_necessity_grid_reuse{threshold}{page_suffix}.png"
        _save_task_grid(
            df=df,
            tasks=task_chunk,
            y_col="lift_val",
            y_label="Necessity (Val)",
            title_prefix="Necessity vs Top-K Components",
            out_path=lift_out_path,
            threshold=threshold,
            nrows=2,
            ncols=3,
        )

        reuse_out_path = out_dir / f"k_sweep_reuse_grid_reuse{threshold}{page_suffix}.png"
        _save_task_grid(
            df=df,
            tasks=task_chunk,
            y_col="reuse_percent",
            y_label="Reuse %",
            title_prefix="Reuse % vs Top-K Components",
            out_path=reuse_out_path,
            threshold=threshold,
            nrows=2,
            ncols=3,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="plots_k_sweep")
    parser.add_argument("--threshold", type=int, default=100)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = discover_metrics(results_dir)
    df = aggregate(paths)

    if df.empty:
        print("No data found.")
        return

    plot_k_sweep(df, out_dir, args.threshold)


if __name__ == "__main__":
    main()