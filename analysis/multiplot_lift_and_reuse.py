from __future__ import annotations
import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.patches import Patch

from circuit_reuse.dataset import get_task_display_name, get_model_display_name

METHOD_DISPLAY = {"eap": "EAP", "gradient": "Gradient"}

SKIP_TASKS = ["mmlu"]

def _extract_step_from_revision(rev: str) -> Optional[int]:
    m = re.search(r"(?:^|[-_])step(\d+)(?:$|[-_])", str(rev) if rev is not None else "")
    return int(m.group(1)) if m else None


def _model_display_with_revision(row: pd.Series) -> str:
    base = get_model_display_name(row.get("model_name"))
    rev = row.get("hf_revision", None)
    if pd.isna(rev) or rev is None or str(rev) == "":
        return base
    step = _extract_step_from_revision(str(rev))
    return f"step{step}" if step is not None else base


def safe_filename(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name))
    s = re.sub(r"_+", "_", s)
    return s.strip("_.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--percent", action="store_true", default=True)
    return p.parse_args()


def discover_metrics(results_dir: Path) -> List[Path]:
    if not results_dir.exists():
        return []
    return sorted(results_dir.rglob("metrics.json"))


def load_metrics_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load {path}: {e}")
        return {}


def _expand_v2(r: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    base = {
        "version": r.get("version", 1),
        "model_name": r.get("model_name"),
        "hf_revision": r.get("hf_revision"),
        "task": r.get("task"),
        "method": r.get("method"),
        "num_examples": r.get("num_examples"),
        "baseline_train_accuracy": r.get("baseline_train_accuracy"),
        "baseline_val_accuracy": r.get("baseline_val_accuracy"),
    }

    by_k = (r.get("by_k") or {})
    for k_str, block in by_k.items():
        try:
            K = int(k_str)
        except Exception:
            continue

        thresholds = (block.get("thresholds") or {})
        for p_str, tblock in thresholds.items():
            try:
                P = int(p_str)
            except Exception:
                continue

            tr = (tblock.get("train") or {})
            va = (tblock.get("val") or {})
            row = dict(base)
            row.update({
                "top_k": K,
                "reuse_threshold": P,
                "reuse_percent": tblock.get("reuse_percent"),
                "shared_circuit_size": tblock.get("shared_circuit_size"),
                "ablation_train_accuracy": tr.get("ablation_accuracy"),
                "control_train_accuracy": tr.get("control_accuracy"),
                "ablation_val_accuracy": va.get("ablation_accuracy"),
                "control_val_accuracy": va.get("control_accuracy"),
                "knockout_diff_train": tr.get("knockout_diff"),
                "knockout_diff_val": va.get("knockout_diff"),
            })
            rows.append(row)

    return rows


def aggregate(paths: List[Path]) -> pd.DataFrame:
    expanded: List[Dict[str, Any]] = []
    for p in paths:
        r = load_metrics_json(p)
        if not r:
            continue

        if int(r.get("version", 1)) >= 2 and "by_k" in r:
            expanded.extend(_expand_v2(r))
        else:
            expanded.append(r)

    return pd.DataFrame(expanded) if expanded else pd.DataFrame()


def to_display(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    if "task" in df.columns:
        df["task_display"] = df["task"].apply(get_task_display_name)
    
    df["hf_revision_step"] = df["hf_revision"].apply(_extract_step_from_revision)
    df["model_display"] = df.apply(_model_display_with_revision, axis=1)
    df["method_display"] = df.get("method", pd.Series(dtype=str)).map(METHOD_DISPLAY).fillna(
        df.get("method", pd.Series(dtype=str)).str.title()
    )
    return df


def compute_lift(df: pd.DataFrame, split: str) -> pd.Series:
    b = df[f"baseline_{split}_accuracy"].astype(float)
    a = df[f"ablation_{split}_accuracy"].astype(float)
    c = df[f"control_{split}_accuracy"].astype(float)

    with np.errstate(divide="ignore", invalid="ignore"):
        lift = (c - a) / b

    return pd.to_numeric(lift).replace([np.inf, -np.inf], np.nan)


def compute_reuse(df: pd.DataFrame, percent: bool, top_k_col="top_k") -> pd.Series:
    if "reuse_percent" in df.columns and df["reuse_percent"].notna().any():
        val = df["reuse_percent"].astype(float)
        return val if percent else (val / 100.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        frac = (df["shared_circuit_size"].astype(float) / df[top_k_col].astype(float))

    return (frac * 100.0) if percent else frac


def _subplot_grid(n: int) -> Tuple[int, int]:
    rows = max(1, (n + 2) // 3)
    cols = min(3, n)
    return rows, cols


def _multiplot_for_k(df_k: pd.DataFrame, out_dir: Path, *, split: str, percent: bool,
                     plot_reuse: bool = True):
    df_k = df_k[~df_k["task"].isin(SKIP_TASKS)]
    k_val = int(df_k["top_k"].iloc[0])

    ps_sorted = sorted(df_k["reuse_threshold"].dropna().unique().tolist())
    norm = Normalize(vmin=min(ps_sorted), vmax=max(ps_sorted))
    cmap_lift = plt.get_cmap("viridis")
    cmap_reuse = plt.get_cmap("magma")
    colors_lift = {p: cmap_lift(norm(p)) for p in ps_sorted}
    colors_reuse = {p: cmap_reuse(norm(p)) for p in ps_sorted}
    
    # Custom sorting for models
    def model_sort_key(model_name):
        row = df_k[df_k['model_display'] == model_name].iloc[0]
        step = row['hf_revision_step']
        return (step if not pd.isna(step) else float('inf'), model_name)

    tasks = sorted(df_k["task_display"].dropna().unique().tolist())
    models = sorted(df_k["model_display"].dropna().unique().tolist(), key=model_sort_key)
    methods = sorted(df_k["method_display"].dropna().unique().tolist())
    
    FONT_SIZES = {
        "title": 36,
        "label": 32,
        "tick": 24,
        "legend_title": 26,
    }

    TITLE_PARAMS = {
        "title_y": 1,
    }

    shared_plot_params = {
        "figsize": (4.0 * len(tasks), 1.8 * len(models)),
        "constrained_layout": False,
        "squeeze": False,
    }
    bbox_to_anchor = (0.5, -0.12)

    if len(tasks) == 1:
        shared_plot_params["figsize"] = (14, 6)
        bbox_to_anchor = (0.5, -0.20)

    shared_ticklabel_params = {"rotation": 30, "ha": "right", "fontsize": 22}
    shared_grid_params = {"axis": "y", "linestyle": "-", "alpha": 0.8}

    def _plot_bars(ax, metric_map, ylabel, ylim, show_ylabel, show_xlabel, color_map):
        xloc = np.arange(len(models))
        nP = len(ps_sorted)
        width = min(0.8 / max(1, nP), 0.25)
        offs = (np.arange(nP) - (nP - 1) / 2.0) * (width + 0.015)

        for j, model in enumerate(models):
            for i, p in enumerate(ps_sorted):
                y = metric_map.get((model, p), np.nan)
                if not np.isnan(y):
                    ax.bar(
                        xloc[j] + offs[i],
                        y,
                        width=width,
                        color=color_map[p],
                        edgecolor="black",
                        linewidth=0.4
                    )

        ax.set_xticks(xloc)
        if show_xlabel:
            ax.set_xticklabels(models, **shared_ticklabel_params)
        else:
            ax.set_xticklabels([])

        ax.tick_params(axis="y", labelsize=22)

        if show_ylabel:
            ax.set_ylabel(ylabel, fontsize=FONT_SIZES["label"])
            ax.tick_params(axis="y", labelsize=22)
        else:
            ax.tick_params(axis="y", labelleft=False)

        ax.grid(**shared_grid_params)
        ax.set_ylim(*ylim)

    for method in methods:
        sub_m = df_k[df_k["method_display"] == method].copy()
        if sub_m.empty:
            continue

        sub_m["lift"] = compute_lift(sub_m, split=split)
        rows, cols = _subplot_grid(len(tasks))
        fig, axes = plt.subplots(rows, cols, **shared_plot_params)
        fig.subplots_adjust(wspace=0.1, hspace=0.15)

        for idx, task in enumerate(tasks):
            ax = axes[idx // cols][idx % cols]
            dd = sub_m[sub_m["task_display"] == task]

            metric_map: Dict[Tuple[str, int], float] = {}
            for _, row in dd.iterrows():
                metric_map[(row["model_display"], int(row["reuse_threshold"]))] = row["lift"]

            _plot_bars(
                ax,
                metric_map,
                ylabel="Necessity",
                ylim=(-1.0, 1.0),
                show_ylabel=(idx % cols == 0),
                show_xlabel=(idx // cols == rows - 1),
                color_map=colors_lift,
            )
            ax.axhline(0.0, color="gray", linewidth=1, linestyle="--", alpha=0.7)
            ax.set_title(task, fontsize=FONT_SIZES["title"], pad=20)

        for k in range(len(tasks), rows * cols):
            fig.delaxes(axes[k // cols][k % cols])


        handles = [Patch(facecolor=colors_lift[p], edgecolor="black", label=str(p)) for p in ps_sorted]
        fig.legend(
            handles=handles,
            title="reuse@P",
            loc="lower center",
            bbox_to_anchor=bbox_to_anchor,
            fontsize=FONT_SIZES["tick"],
            title_fontsize=FONT_SIZES["legend_title"],
            ncol=len(ps_sorted)
        )

        outp = out_dir / f"multiplot_necessity_k{k_val}_{safe_filename(method.lower())}_{split}.png"
        outp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outp, dpi=200, bbox_inches="tight")
        print(f"[INFO] Saved plot to {outp}")
        plt.close(fig)

        if plot_reuse:
            sub_m["reuse_metric"] = compute_reuse(sub_m, percent=percent)
            fig, axes = plt.subplots(rows, cols, **shared_plot_params)
            fig.subplots_adjust(wspace=0.1, hspace=0.15)

            for idx, task in enumerate(tasks):
                ax = axes[idx // cols][idx % cols]
                dd = sub_m[sub_m["task_display"] == task]
                metric_map: Dict[Tuple[str, int], float] = {}
                for _, row in dd.iterrows():
                    metric_map[(row["model_display"], int(row["reuse_threshold"]))] = row["reuse_metric"]

                _plot_bars(
                    ax,
                    metric_map,
                    ylabel="Reuse (%)" if percent else "Reuse",
                    ylim=(0, 100 if percent else 1),
                    show_ylabel=(idx % cols == 0),
                    show_xlabel=(idx // cols == rows - 1),
                    color_map=colors_reuse,
                )
                ax.set_title(task, fontsize=FONT_SIZES["title"], pad=4)

            for k in range(len(tasks), rows * cols):
                fig.delaxes(axes[k // cols][k % cols])

    
            handles = [Patch(facecolor=colors_reuse[p], edgecolor="black", label=str(p)) for p in ps_sorted]
            fig.legend(
                handles=handles,
                title="reuse@P",
                loc="lower center",
                bbox_to_anchor=bbox_to_anchor,
                fontsize=FONT_SIZES["tick"],
                title_fontsize=FONT_SIZES["legend_title"],
                ncol=len(ps_sorted)
            )

            outp = out_dir / f"multiplot_reuse_k{k_val}_{safe_filename(method.lower())}.png"
            outp.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(outp, dpi=200, bbox_inches="tight")
            print(f"[INFO] Saved plot to {outp}")
            plt.close(fig)


def _lineplot_checkpoints_vs_k(df: pd.DataFrame, out_dir: Path, *, split: str,
                               percent: bool, plot_reuse: bool = True):
    """Line plot: x = top_k, one line per checkpoint, one subplot per task.

    Uses median reuse threshold to keep it simple (one line per checkpoint).
    """
    df = df[~df["task"].isin(SKIP_TASKS)].copy()
    if df.empty:
        return

    tasks = sorted(df["task_display"].dropna().unique().tolist())
    methods = sorted(df["method_display"].dropna().unique().tolist())

    # Use median reuse threshold
    ps_sorted = sorted(df["reuse_threshold"].dropna().unique().tolist())
    median_p = ps_sorted[len(ps_sorted) // 2]

    def model_sort_key(model_name):
        row = df[df['model_display'] == model_name].iloc[0]
        step = row['hf_revision_step']
        return (step if not pd.isna(step) else float('inf'), model_name)

    models = sorted(df["model_display"].dropna().unique().tolist(), key=model_sort_key)
    ks = sorted(df["top_k"].dropna().unique().tolist())

    cmap = plt.get_cmap("viridis", len(models))
    colors = {m: cmap(i) for i, m in enumerate(models)}

    cmap = plt.get_cmap("viridis", len(models))
    colors = {m: cmap(i) for i, m in enumerate(models)}

    FONT_SIZES = {"title": 28, "label": 24, "tick": 20, "legend": 20}
    k_labels = [str(int(k)) for k in ks]

    for method in methods:
        sub = df[(df["method_display"] == method) & (df["reuse_threshold"] == median_p)].copy()
        if sub.empty:
            continue

        sub["lift"] = compute_lift(sub, split=split)
        sub["reuse_metric"] = compute_reuse(sub, percent=percent)

        for metric, ylabel, ylim, suffix in [
            ("lift", "Necessity", (-1.0, 1.0), f"necessity_{split}"),
            *([("reuse_metric", "Reuse (%)" if percent else "Reuse",
                (0, 100 if percent else 1), "reuse")] if plot_reuse else []),
        ]:
            rows_g, cols_g = _subplot_grid(len(tasks))
            fig, axes = plt.subplots(rows_g, cols_g,
                                     figsize=(5.0 * cols_g, 3.5 * rows_g),
                                     constrained_layout=False, squeeze=False)
            fig.subplots_adjust(wspace=0.12, hspace=0.40)

            for idx, task in enumerate(tasks):
                ax = axes[idx // cols_g][idx % cols_g]
                dd = sub[sub["task_display"] == task]

                for model in models:
                    dm = dd[dd["model_display"] == model].sort_values("top_k")
                    if dm.empty:
                        continue
                    x_pos = [ks.index(k) for k in dm["top_k"]]
                    ax.plot(x_pos, dm[metric].values, marker="o", markersize=5,
                            label=model, color=colors[model], linewidth=1.5)

                ax.set_title(task, fontsize=FONT_SIZES["title"], pad=8)
                if metric == "lift":
                    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
                ax.set_ylim(*ylim)
                ax.set_xticks(range(len(ks)))
                if idx // cols_g == rows_g - 1:
                    ax.set_xticklabels(k_labels, fontsize=FONT_SIZES["tick"])
                else:
                    ax.set_xticklabels([])
                ax.tick_params(axis="y", labelsize=FONT_SIZES["tick"])
                if idx % cols_g == 0:
                    ax.set_ylabel(ylabel, fontsize=FONT_SIZES["label"])
                else:
                    ax.tick_params(axis="y", labelleft=False)
                ax.grid(axis="y", linestyle="-", alpha=0.8)

            for k in range(len(tasks), rows_g * cols_g):
                fig.delaxes(axes[k // cols_g][k % cols_g])

            handles = [plt.Line2D([0], [0], color=colors[m], marker="o", markersize=5,
                                  linewidth=1.5, label=m) for m in models]
            fig.legend(handles=handles, loc="lower center",
                       bbox_to_anchor=(0.5, -0.14),
                       fontsize=FONT_SIZES["legend"],
                       ncol=min(len(models), 4))

            outp = out_dir / f"lineplot_{suffix}_p{int(median_p)}_{safe_filename(method.lower())}.png"
            outp.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(outp, dpi=200, bbox_inches="tight")
            print(f"[INFO] Saved line plot to {outp}")
            plt.close(fig)


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    paths = discover_metrics(results_dir)
    df = aggregate(paths)

    if df.empty:
        print("[INFO] No metrics.json found.")
        return

    df = to_display(df)
    sns.set_theme(style="ticks", context="notebook", palette="colorblind")
    plt.rcParams.update({"font.family": "serif", "font.size": 14})

    available_reuse_ps = df["reuse_threshold"].dropna().unique().tolist()
    if available_reuse_ps:
        df = df[df["reuse_threshold"].isin(available_reuse_ps)]

    out_dir = Path(args.output_dir) if args.output_dir else results_dir / f"plots_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reuse plots (split-independent) go in their own folder
    out_dir_reuse = out_dir / "reuse"
    out_dir_reuse.mkdir(parents=True, exist_ok=True)
    for k_val, df_k in df.groupby("top_k"):
        if df_k.empty:
            continue
        df_k_sorted = df_k.sort_values(["task_display", "hf_revision_step", "method_display", "reuse_threshold"])
        _multiplot_for_k(
            df_k_sorted, out_dir_reuse,
            split="train", percent=args.percent,
            plot_reuse=True,
        )

    # Lift plots (split-dependent) go in train/val folders
    for split in ["train", "val"]:
        out_dir_split = out_dir / split
        out_dir_split.mkdir(parents=True, exist_ok=True)

        for k_val, df_k in df.groupby("top_k"):
            if df_k.empty:
                continue

            df_k_sorted = df_k.sort_values(["task_display", "hf_revision_step", "method_display", "reuse_threshold"])
            _multiplot_for_k(
                df_k_sorted, out_dir_split,
                split=split, percent=args.percent,
                plot_reuse=False,
            )

    # Line plots: checkpoints as lines, top_k on x-axis
    out_dir_lines = out_dir / "lines"
    out_dir_lines.mkdir(parents=True, exist_ok=True)
    _lineplot_checkpoints_vs_k(df, out_dir_lines, split="train", percent=args.percent, plot_reuse=True)


if __name__ == "__main__":
    main()