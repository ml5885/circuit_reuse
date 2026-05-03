"""
Generate a report relating circuit reuse (IoU/Jaccard-style overlap) to
Spearman rank stability for the cached cross-task runs.

Outputs:
  - results/reuse_vs_spearman/combined_exact_values.csv
  - results/reuse_vs_spearman/slice_summary.csv
  - results/reuse_vs_spearman/reuse_30_neighborhood.csv
  - results/reuse_vs_spearman/figs/*.png
  - docs/reuse_vs_spearman_report.md
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
SPEARMAN_NPZ = REPO / "results" / "spearman_stability" / "spearman_metrics.npz"
OUT = REPO / "results" / "reuse_vs_spearman"
FIGS = OUT / "figs"
REPORT = REPO / "docs" / "reuse_vs_spearman_report.md"

METHOD_TO_ROOT = {
    "eap": REPO / "results" / "cross_task",
    "relp": REPO / "results" / "cross_task_relp",
}

METHOD_LABELS = {
    "eap": "EAP",
    "relp": "RelP",
}

MODEL_ORDER = [
    ("google_gemma-2-2b", "Gemma-2-2B"),
    ("google_gemma-2-2b-it", "Gemma-2-2B-IT"),
    ("meta-llama_Llama-3.2-3B", "Llama-3.2-3B"),
    ("meta-llama_Llama-3.2-3B-Instruct", "Llama-3.2-3B-Instruct"),
    ("qwen3-4b", "Qwen3-4B"),
    ("qwen3-8b", "Qwen3-8B"),
]

TASK_ORDER = [
    ("addition", "Addition"),
    ("arc_challenge", "ARC (Challenge)"),
    ("arc_easy", "ARC (Easy)"),
    ("boolean", "Boolean"),
    ("ioi", "IOI"),
    ("mcqa", "CopyColors MCQA"),
]

K_ORDER = [1, 5, 10, 20, 30]
REUSE_THRESHOLDS = [95, 96, 97, 98, 99, 100]

MODEL_LABELS = dict(MODEL_ORDER)
TASK_LABELS = dict(TASK_ORDER)
MODEL_INDEX = {slug: i for i, (slug, _) in enumerate(MODEL_ORDER)}
TASK_INDEX = {task: i for i, (task, _) in enumerate(TASK_ORDER)}
MODEL_NAME_TO_SLUG = {
    "google/gemma-2-2b": "google_gemma-2-2b",
    "google/gemma-2-2b-it": "google_gemma-2-2b-it",
    "meta-llama/Llama-3.2-3B": "meta-llama_Llama-3.2-3B",
    "meta-llama/Llama-3.2-3B-Instruct": "meta-llama_Llama-3.2-3B-Instruct",
    "qwen3-4b": "qwen3-4b",
    "qwen3-8b": "qwen3-8b",
}


def _load_spearman_lookup() -> dict[tuple[str, str, str, int], dict[str, float]]:
    data = np.load(SPEARMAN_NPZ, allow_pickle=True)
    npz_models = list(data["models"])
    npz_tasks = list(data["tasks"])
    npz_ks = [int(v) for v in data["K_pcts"]]

    label_to_model_slug = {label: slug for slug, label in MODEL_ORDER}
    lookup: dict[tuple[str, str, str, int], dict[str, float]] = {}
    for method in METHOD_TO_ROOT:
        pairwise = data[f"{method}_pairwise"]
        reference = data[f"{method}_reference"]
        for mi, model_label in enumerate(npz_models):
            model_slug = label_to_model_slug[model_label]
            for ti, task in enumerate(npz_tasks):
                if task not in TASK_INDEX:
                    continue
                for ki, k_pct in enumerate(npz_ks):
                    lookup[(method, model_slug, task, int(k_pct))] = {
                        "spearman_pairwise": float(pairwise[mi, ti, ki]),
                        "spearman_reference": float(reference[mi, ti, ki]),
                    }
    return lookup


def _iter_reuse_rows(spearman_lookup: dict[tuple[str, str, str, int], dict[str, float]]):
    for method, root in METHOD_TO_ROOT.items():
        for path in sorted(root.glob("**/metrics.json")):
            with path.open() as f:
                metrics = json.load(f)
            model_name = metrics["model_name"]
            model = MODEL_NAME_TO_SLUG.get(model_name, model_name)
            task = metrics["task"]
            if model not in MODEL_LABELS or task not in TASK_LABELS:
                continue
            if task == "mmlu":
                continue
            for k_str, k_block in metrics["by_k"].items():
                k_pct = int(k_str)
                key = (method, model, task, k_pct)
                if key not in spearman_lookup:
                    continue
                spearman = spearman_lookup[key]
                for thr_str, thr_block in k_block["thresholds"].items():
                    thr = int(thr_str)
                    if thr not in REUSE_THRESHOLDS:
                        continue
                    yield {
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "model": model,
                        "model_label": MODEL_LABELS[model],
                        "task": task,
                        "task_label": TASK_LABELS[task],
                        "K_pct": k_pct,
                        "reuse_threshold": thr,
                        "reuse_percent": float(thr_block["reuse_percent"]),
                        "shared_circuit_size": int(thr_block["shared_circuit_size"]),
                        "spearman_pairwise": spearman["spearman_pairwise"],
                        "spearman_reference": spearman["spearman_reference"],
                    }


def _build_dataframes() -> tuple[pd.DataFrame, pd.DataFrame]:
    spearman_lookup = _load_spearman_lookup()
    combined = pd.DataFrame(list(_iter_reuse_rows(spearman_lookup)))
    combined["method"] = pd.Categorical(combined["method"], categories=list(METHOD_TO_ROOT), ordered=True)
    combined["model_label"] = pd.Categorical(
        combined["model_label"],
        categories=[label for _, label in MODEL_ORDER],
        ordered=True,
    )
    combined["task_label"] = pd.Categorical(
        combined["task_label"],
        categories=[label for _, label in TASK_ORDER],
        ordered=True,
    )
    combined = combined.sort_values(
        ["method", "model_label", "task_label", "K_pct", "reuse_threshold"]
    ).reset_index(drop=True)

    summary = (
        combined.pivot_table(
            index=[
                "method",
                "method_label",
                "model",
                "model_label",
                "task",
                "task_label",
                "K_pct",
                "spearman_pairwise",
                "spearman_reference",
            ],
            columns="reuse_threshold",
            values="reuse_percent",
        )
        .reset_index()
        .rename(columns={thr: f"reuse_at_{thr}" for thr in REUSE_THRESHOLDS})
    )
    summary = summary.sort_values(
        ["method", "model_label", "task_label", "K_pct"]
    ).reset_index(drop=True)
    return combined, summary


def _plot_scatter(df: pd.DataFrame, spearman_col: str, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True, sharey=True)
    colors = {1: "#1f77b4", 5: "#ff7f0e", 10: "#2ca02c", 20: "#d62728", 30: "#9467bd"}
    markers = {95: "o", 96: "s", 97: "^", 98: "D", 99: "P", 100: "X"}

    for ax, method in zip(axes, ["eap", "relp"]):
        sub = df[df["method"] == method]
        for thr in REUSE_THRESHOLDS:
            thr_sub = sub[sub["reuse_threshold"] == thr]
            for k_pct in K_ORDER:
                pts = thr_sub[thr_sub["K_pct"] == k_pct]
                if pts.empty:
                    continue
                ax.scatter(
                    pts["reuse_percent"],
                    pts[spearman_col],
                    s=40,
                    alpha=0.75,
                    color=colors[k_pct],
                    marker=markers[thr],
                    edgecolors="white",
                    linewidths=0.4,
                )
        ax.axvline(30.0, color="0.6", linestyle="--", linewidth=1)
        ax.set_title(METHOD_LABELS[method])
        ax.set_xlabel("Reuse (%)")
        ax.grid(alpha=0.2)

    axes[0].set_ylabel("Spearman rho")
    fig.suptitle(
        "Circuit reuse vs Spearman stability"
        f" ({'pairwise' if spearman_col.endswith('pairwise') else 'reference'})",
        y=0.98,
    )
    legend_lines = [
        plt.Line2D([0], [0], marker="o", color="w", label=f"K={k}%", markerfacecolor=colors[k], markersize=8)
        for k in K_ORDER
    ]
    legend_marks = [
        plt.Line2D([0], [0], marker=markers[thr], color="0.2", linestyle="", label=f"reuse@{thr}", markersize=8)
        for thr in REUSE_THRESHOLDS
    ]
    fig.legend(
        handles=legend_lines + legend_marks,
        loc="lower center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_reuse30_box(df: pd.DataFrame, out_path: Path):
    near = df[(df["reuse_percent"] >= 25.0) & (df["reuse_percent"] <= 35.0)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    cols = [
        ("spearman_pairwise", "Pairwise Spearman"),
        ("spearman_reference", "Reference Spearman"),
    ]
    for ax, (col, title) in zip(axes, cols):
        groups = []
        labels = []
        for method in ["eap", "relp"]:
            vals = near.loc[near["method"] == method, col].to_numpy()
            if vals.size:
                groups.append(vals)
                labels.append(METHOD_LABELS[method])
        ax.boxplot(groups, tick_labels=labels, showmeans=True)
        ax.set_title(title)
        ax.grid(alpha=0.2, axis="y")
    axes[0].set_ylabel("Spearman rho")
    fig.suptitle("Spearman values for points with reuse in [25%, 35%]", y=0.98)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _summarize_reuse30(df: pd.DataFrame) -> pd.DataFrame:
    near = df[(df["reuse_percent"] >= 25.0) & (df["reuse_percent"] <= 35.0)].copy()
    rows = []
    for method in ["eap", "relp"]:
        sub = near[near["method"] == method]
        if sub.empty:
            continue
        for col in ["spearman_pairwise", "spearman_reference"]:
            rows.append(
                {
                    "method": METHOD_LABELS[method],
                    "metric": "pairwise" if col.endswith("pairwise") else "reference",
                    "count": int(sub.shape[0]),
                    "reuse_min": float(sub["reuse_percent"].min()),
                    "reuse_max": float(sub["reuse_percent"].max()),
                    "spearman_min": float(sub[col].min()),
                    "spearman_mean": float(sub[col].mean()),
                    "spearman_median": float(sub[col].median()),
                    "spearman_max": float(sub[col].max()),
                }
            )
    return pd.DataFrame(rows)


def _nearest_to_target(df: pd.DataFrame, target: float = 30.0) -> pd.DataFrame:
    nearest = df.copy()
    nearest["reuse_distance"] = (nearest["reuse_percent"] - target).abs()
    nearest = nearest.sort_values(
        ["method", "model_label", "task_label", "K_pct", "reuse_distance", "reuse_threshold"]
    )
    nearest = nearest.groupby(["method", "model_label", "task_label", "K_pct"], as_index=False).first()
    return nearest.sort_values(["method", "model_label", "task_label", "K_pct"]).reset_index(drop=True)


def _format_float(v: float) -> str:
    return f"{v:.3f}"


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    table = df[columns].copy()
    headers = list(table.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in table.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def _write_report(
    combined: pd.DataFrame,
    summary: pd.DataFrame,
    reuse30_summary: pd.DataFrame,
    reuse30_nearest: pd.DataFrame,
):
    pair_fig = OUT / "figs" / "reuse_vs_spearman_pairwise.png"
    ref_fig = OUT / "figs" / "reuse_vs_spearman_reference.png"
    near_fig = OUT / "figs" / "reuse30_spearman_boxplot.png"

    overall = []
    for method in ["eap", "relp"]:
        sub = combined[combined["method"] == method]
        overall.append(
            {
                "Method": METHOD_LABELS[method],
                "Reuse min": _format_float(float(sub["reuse_percent"].min())),
                "Reuse mean": _format_float(float(sub["reuse_percent"].mean())),
                "Reuse max": _format_float(float(sub["reuse_percent"].max())),
                "Pairwise rho mean": _format_float(float(sub["spearman_pairwise"].mean())),
                "Reference rho mean": _format_float(float(sub["spearman_reference"].mean())),
            }
        )
    overall_md = _markdown_table(
        pd.DataFrame(overall),
        ["Method", "Reuse min", "Reuse mean", "Reuse max", "Pairwise rho mean", "Reference rho mean"],
    )

    reuse30_df = reuse30_summary.copy()
    for col in ["reuse_min", "reuse_max", "spearman_min", "spearman_mean", "spearman_median", "spearman_max"]:
        reuse30_df[col] = reuse30_df[col].map(_format_float)
    reuse30_df = reuse30_df.rename(
        columns={
            "method": "Method",
            "metric": "Metric",
            "count": "Count",
            "reuse_min": "Reuse min",
            "reuse_max": "Reuse max",
            "spearman_min": "Spearman min",
            "spearman_mean": "Spearman mean",
            "spearman_median": "Spearman median",
            "spearman_max": "Spearman max",
        }
    )
    reuse30_md = _markdown_table(
        reuse30_df,
        [
            "Method",
            "Metric",
            "Count",
            "Reuse min",
            "Reuse max",
            "Spearman min",
            "Spearman mean",
            "Spearman median",
            "Spearman max",
        ],
    )

    nearest_cols = [
        "method_label",
        "model_label",
        "task_label",
        "K_pct",
        "reuse_threshold",
        "reuse_percent",
        "spearman_pairwise",
        "spearman_reference",
    ]
    nearest_display = reuse30_nearest.copy()
    for col in ["reuse_percent", "reuse_distance", "spearman_pairwise", "spearman_reference"]:
        nearest_display[col] = nearest_display[col].map(_format_float)
    nearest_display = nearest_display.rename(
        columns={
            "method_label": "Method",
            "model_label": "Model",
            "task_label": "Task",
            "K_pct": "K (%)",
            "reuse_threshold": "Reuse @p",
            "reuse_percent": "Reuse (%)",
            "reuse_distance": "Reuse Dist",
            "spearman_pairwise": "Pairwise rho",
            "spearman_reference": "Reference rho",
        }
    )

    lines = [
        "# Reuse vs Spearman Report",
        "",
        "This report aligns the two existing circuit-similarity metrics on the same cached slices:",
        "",
        "- `reuse_percent`: shared-circuit size normalized by the per-example top-`K%` circuit size, using the reuse threshold `p` from the existing overlap analysis.",
        "- `Spearman rho`: rank correlation on the union of the two top-`K%` scored component sets, using the existing stability analysis.",
        "",
        "Important caveat: these are related but not equivalent metrics. For a fixed `(method, model, task, K)`, the Spearman value is constant while `reuse_percent` changes with the reuse threshold `p`. So there is no single exact mapping from reuse to Spearman; the best we can show is the empirical relationship across slices and thresholds.",
        "",
        "## Figures",
        "",
        f"![Reuse vs Spearman pairwise](../results/reuse_vs_spearman/figs/{pair_fig.name})",
        "",
        f"![Reuse vs Spearman reference](../results/reuse_vs_spearman/figs/{ref_fig.name})",
        "",
        f"![Reuse 30 neighborhood](../results/reuse_vs_spearman/figs/{near_fig.name})",
        "",
        "## Overall Summary",
        "",
        overall_md,
        "",
        "## What Does 30% Reuse Mean?",
        "",
        "There is no unique translation, so the table below summarizes the observed Spearman values for points whose reuse lies in `[25%, 35%]`.",
        "",
        reuse30_md,
        "",
        "The full exact values are still available in CSV form if you want them later.",
        "",
        f"Full CSVs: [`combined_exact_values.csv`](../results/reuse_vs_spearman/combined_exact_values.csv), [`slice_summary.csv`](../results/reuse_vs_spearman/slice_summary.csv), [`reuse_30_neighborhood.csv`](../results/reuse_vs_spearman/reuse_30_neighborhood.csv)",
        "",
    ]

    REPORT.write_text("\n".join(lines))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    combined, summary = _build_dataframes()
    reuse30_summary = _summarize_reuse30(combined)
    reuse30_nearest = _nearest_to_target(combined, target=30.0)

    combined.to_csv(OUT / "combined_exact_values.csv", index=False)
    summary.to_csv(OUT / "slice_summary.csv", index=False)
    reuse30_nearest.to_csv(OUT / "reuse_30_neighborhood.csv", index=False)

    _plot_scatter(combined, "spearman_pairwise", FIGS / "reuse_vs_spearman_pairwise.png")
    _plot_scatter(combined, "spearman_reference", FIGS / "reuse_vs_spearman_reference.png")
    _plot_reuse30_box(combined, FIGS / "reuse30_spearman_boxplot.png")
    _write_report(combined, summary, reuse30_summary, reuse30_nearest)

    print(f"wrote {OUT}")
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
