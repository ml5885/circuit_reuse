"""Aggregate and visualize selective ablation results.

Loads all selective_*.json files, computes selectivity metrics for each
condition, and produces:
  1. Markdown summary and per-pair tables
  2. Side-by-side bar plots: target vs non-target drop (absolute and relative)
  3. Target-only bar plots including full circuit
  4. Selectivity bar plots
  5. Scatter plots
  6. Flat CSV
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from circuit_reuse.dataset import get_model_display_name, get_task_display_name

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "serif"

CONDITIONS = ["shared_core", "residual_a", "residual_b", "random_control"]
COND_DISPLAY = {
    "shared_core": "Shared Core",
    "residual_a": "Task-Specific",
    "residual_b": "Task-Complement",
    "random_control": "Random Control",
}
COND_COLORS = {
    "shared_core": "#1f77b4",
    "residual_a": "#d62728",
    "residual_b": "#ff7f0e",
    "random_control": "#7f7f7f",
}

MIN_RESIDUAL_SIZE = 2


def load_all_results(results_dirs: list[Path]) -> list[dict]:
    records = []
    for d in results_dirs:
        for p in sorted(d.glob("selective_*.json")):
            with p.open() as f:
                records.append(json.load(f))
    return records


def load_cross_task_results(cross_task_dir: Path) -> dict:
    results = {}
    for p in sorted(cross_task_dir.glob("cross_task_*.json")):
        with p.open() as f:
            d = json.load(f)
        results[(d["model_name"], d["K"])] = d
    return results


def load_all_cross_task(base_dirs: list[Path]) -> dict:
    all_ct = {}
    for d in base_dirs:
        if d.exists():
            all_ct.update(load_cross_task_results(d))
    return all_ct


_EXCLUDE_TASKS: set[str] = set()


def compute_drops(record: dict, condition: str) -> dict | None:
    """Compute absolute and relative drops for target and non-target tasks."""
    task_a = record["task_a"]
    if task_a in _EXCLUDE_TASKS:
        return None
    eval_tasks = [t for t in record["baseline_accuracy"] if t not in _EXCLUDE_TASKS]
    cond = record["conditions"].get(condition, {})
    size = cond.get("size", 0)
    if size == 0:
        return None

    target_drop = cond.get(task_a, {}).get("drop_pp", 0.0)
    target_rel = cond.get(task_a, {}).get("relative_drop_pct", 0.0)

    other_drops = [cond[t]["drop_pp"] for t in eval_tasks if t != task_a and t in cond]
    other_rels = [cond[t]["relative_drop_pct"] for t in eval_tasks if t != task_a and t in cond]
    mean_other = sum(other_drops) / len(other_drops) if other_drops else 0.0
    mean_other_rel = sum(other_rels) / len(other_rels) if other_rels else 0.0

    return {
        "target_drop": target_drop,
        "nontarget_drop": mean_other,
        "target_rel": target_rel,
        "nontarget_rel": mean_other_rel,
        "size": size,
    }


def ratio_of_means(drops: list[dict], key="target_drop", other_key="nontarget_drop") -> float:
    if not drops:
        return float("nan")
    tgt = np.mean([d[key] for d in drops])
    ntgt = np.mean([d[other_key] for d in drops])
    if abs(ntgt) < 1e-6:
        return float("inf") if tgt > 1e-6 else float("nan")
    return tgt / ntgt


def filter_by_residual_size(records: list[dict], min_size: int) -> list[dict]:
    return [r for r in records if r["circuit_sizes"].get("residual_a", 0) >= min_size]


def get_all_models(records: list[dict]) -> list[str]:
    return sorted({r["model_name"] for r in records})


# --- Markdown output ---

def md_compact_tables(records: list[dict], k_values: list[int], min_size: int,
                      all_models: list[str]) -> str:
    """One table per condition per model: rows = tasks, columns = K values.
    Each cell shows target drop / non-target drop (pp)."""
    conds = ["shared_core", "residual_a", "residual_b", "random_control"]

    # model -> task -> K -> condition -> list of drops
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    all_tasks = set()
    for r in records:
        if r["circuit_sizes"].get("residual_a", 0) < min_size:
            continue
        task_a = r["task_a"]
        if task_a in _EXCLUDE_TASKS:
            continue
        for cond in conds:
            d = compute_drops(r, cond)
            if d:
                data[r["model_name"]][task_a][r["K"]][cond].append(d)
                all_tasks.add(task_a)

    tasks = sorted(all_tasks)
    k_headers = " | ".join(f"K={k}%" for k in k_values)
    lines = [
        f"## Accuracy Drop by Condition (task-specific size >= {min_size})",
        "",
        "Each cell: target drop / non-target drop (pp).",
        "",
    ]

    for cond in conds:
        lines += [f"### {COND_DISPLAY[cond]}", ""]
        for model in all_models:
            model_disp = get_model_display_name(model)
            lines += [
                f"#### {model_disp}",
                "",
                f"| Task | {k_headers} |",
                "|------" + "|------:" * len(k_values) + "|",
            ]
            for task in tasks:
                task_disp = get_task_display_name(task)
                cells = [task_disp]
                for k in k_values:
                    drops = data[model][task][k][cond]
                    if drops:
                        tgt = np.mean([d["target_drop"] for d in drops])
                        ntgt = np.mean([d["nontarget_drop"] for d in drops])
                        cells.append(f"{tgt:.1f} / {ntgt:.1f}")
                    else:
                        cells.append("—")
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")

    return "\n".join(lines)


# --- Plot helpers ---

def _model_grid(all_models):
    n = len(all_models)
    ncols = min(n, 3)
    nrows = math.ceil(n / ncols)
    return n, nrows, ncols


def _make_drop_bars(records, k_filter, min_size, all_models, output_dir,
                    tgt_key, ntgt_key, ylabel, title_suffix, fname_suffix,
                    ylim=None):
    """One figure per model. Subplots are tasks. Side-by-side bars: target vs non-target drop per condition."""
    filtered = filter_by_residual_size([r for r in records if r["K"] == k_filter], min_size)

    # model -> task_a -> condition -> list of drops
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    all_tasks = set()
    for r in filtered:
        for cond in CONDITIONS:
            d = compute_drops(r, cond)
            if d:
                data[r["model_name"]][r["task_a"]][cond].append(d)
                all_tasks.add(r["task_a"])

    tasks = sorted(all_tasks)
    n_tasks = len(tasks)
    ncols = min(n_tasks, 3)
    nrows = math.ceil(n_tasks / ncols)

    for model in all_models:
        model_slug = model.replace("/", "_")
        model_disp = get_model_display_name(model)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), sharey=True)
        axes = np.atleast_2d(axes)

        for idx, task in enumerate(tasks):
            r, c = divmod(idx, ncols)
            ax = axes[r, c]
            target_means = []
            nontarget_means = []
            for cond in CONDITIONS:
                drops = data[model][task][cond]
                target_means.append(np.mean([d[tgt_key] for d in drops]) if drops else 0)
                nontarget_means.append(np.mean([d[ntgt_key] for d in drops]) if drops else 0)

            x = np.arange(len(CONDITIONS))
            w = 0.35
            ax.bar(x - w / 2, target_means, w, color="#2a9d8f")
            ax.bar(x + w / 2, nontarget_means, w, color="#e76f51")
            ax.set_xticks(x)
            ax.set_xticklabels([COND_DISPLAY[cond] for cond in CONDITIONS], fontsize=11, rotation=30, ha="right")
            ax.set_title(get_task_display_name(task), fontsize=15)
            if c == 0:
                ax.set_ylabel(ylabel)

            if not data[model][task]:
                ax.text(0.5, 0.5, "No valid pairs", transform=ax.transAxes,
                        ha="center", va="center", fontsize=15, color="gray")

        for idx in range(n_tasks, nrows * ncols):
            r, c = divmod(idx, ncols)
            axes[r, c].set_visible(False)

        if ylim is not None:
            for row in axes:
                for ax in row:
                    if ax.get_visible():
                        ax.set_ylim(ylim)

        handles = [
            plt.Rectangle((0, 0), 1, 1, fc="#2a9d8f", label="Target task"),
            plt.Rectangle((0, 0), 1, 1, fc="#e76f51", label="Non-target mean"),
        ]
        fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=11,
                   bbox_to_anchor=(0.5, 1.0), frameon=False)
        fig.suptitle(f"{model_disp}: {title_suffix} (K={k_filter}%)", fontsize=15, y=1.03)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        model_dir = output_dir / model_slug
        model_dir.mkdir(parents=True, exist_ok=True)
        out = model_dir / f"{fname_suffix}_k{k_filter}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[SAVED] {out}")


# --- Plots ---

def plot_drop_absolute(records, k_filter, min_size, all_models, output_dir):
    _make_drop_bars(records, k_filter, min_size, all_models, output_dir,
                    "target_drop", "nontarget_drop",
                    "Accuracy drop (pp)", "Accuracy Drop", "drop_absolute",
                    ylim=(-5, 100))


def plot_drop_relative(records, k_filter, min_size, all_models, output_dir):
    _make_drop_bars(records, k_filter, min_size, all_models, output_dir,
                    "target_rel", "nontarget_rel",
                    "Relative accuracy drop (%)", "Relative Accuracy Drop", "drop_relative",
                    ylim=(-5, 100))


def plot_condition_target_vs_nontarget(records, k_filter, min_size, all_models, condition, output_dir):
    """One multiplot per condition per K. Subplots = models, x = tasks,
    two bars per task: target drop vs non-target drop."""
    filtered = filter_by_residual_size([r for r in records if r["K"] == k_filter], min_size)

    data = defaultdict(lambda: defaultdict(list))
    all_tasks = set()
    for r in filtered:
        d = compute_drops(r, condition)
        if d:
            data[r["model_name"]][r["task_a"]].append(d)
            all_tasks.add(r["task_a"])

    tasks = sorted(all_tasks)
    n, nrows, ncols = _model_grid(all_models)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), sharey=True)
    axes = np.atleast_2d(axes)

    w = 0.3
    for idx, model in enumerate(all_models):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        x = np.arange(len(tasks))
        has_data = False

        tgt_means = []
        ntgt_means = []
        for task in tasks:
            drops = data[model][task]
            if drops:
                tgt_means.append(np.mean([d["target_drop"] for d in drops]))
                ntgt_means.append(np.mean([d["nontarget_drop"] for d in drops]))
                has_data = True
            else:
                tgt_means.append(0)
                ntgt_means.append(0)

        ax.bar(x - w / 2, tgt_means, w, color="#2a9d8f")
        ax.bar(x + w / 2, ntgt_means, w, color="#e76f51")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([get_task_display_name(t) for t in tasks], fontsize=11, rotation=30, ha="right")
        ax.set_title(get_model_display_name(model), fontsize=15)
        if c == 0:
            ax.set_ylabel("Accuracy drop (pp)")

        if not has_data:
            ax.text(0.5, 0.5, "No valid pairs", transform=ax.transAxes,
                    ha="center", va="center", fontsize=15, color="gray")

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    handles = [
        plt.Rectangle((0, 0), 1, 1, fc="#2a9d8f", label="Target task"),
        plt.Rectangle((0, 0), 1, 1, fc="#e76f51", label="Non-target mean"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=11,
               bbox_to_anchor=(0.5, 1.0), frameon=False)
    cond_name = COND_DISPLAY[condition]
    fig.suptitle(f"{cond_name}: Target vs Non-Target Drop (K={k_filter}%)", fontsize=15, y=1.03)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out = output_dir / f"{condition}_k{k_filter}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out}")


def plot_drop_diff(records, k_filter, min_size, all_models, output_dir):
    conds = ["shared_core", "residual_a", "random_control"]
    _make_diff_bars(records, k_filter, min_size, all_models, conds, COND_COLORS, COND_DISPLAY,
                    "Selectivity", "drop_diff", output_dir)



def _make_diff_bars(records, k_filter, min_size, all_models, conds, cond_colors,
                    cond_labels, title, fname, output_dir, cross_task=None):
    """Generic grouped bar plot: model subplots, x = tasks, y = target - nontarget drop."""
    filtered = filter_by_residual_size([r for r in records if r["K"] == k_filter], min_size)

    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    all_tasks = set()
    for r in filtered:
        for cond in conds:
            if cond == "full_circuit":
                continue
            d = compute_drops(r, cond)
            if d:
                data[r["model_name"]][r["task_a"]][cond].append(d["target_drop"] - d["nontarget_drop"])
                all_tasks.add(r["task_a"])

    # Full circuit diffs from cross-task results
    if cross_task and "full_circuit" in conds:
        for r in filtered:
            model, task_a = r["model_name"], r["task_a"]
            ct = cross_task.get((model, k_filter))
            if ct and task_a in ct.get("accuracy_drop_pp", {}):
                tgt = ct["accuracy_drop_pp"][task_a][task_a]
                others = [ct["accuracy_drop_pp"][task_a][t]
                          for t in ct["accuracy_drop_pp"][task_a] if t != task_a]
                ntgt = sum(others) / len(others) if others else 0
                data[model][task_a]["full_circuit"].append(tgt - ntgt)
                all_tasks.add(task_a)

    tasks = sorted(all_tasks)
    n_conds = len(conds)
    bar_width = 0.55 / n_conds

    # Warn about missing data
    for model in all_models:
        for task in tasks:
            missing = [c for c in conds if not data[model][task][c]]
            if missing and k_filter >= 10:
                print(f"  [MISSING] {fname} K={k_filter}% {model} / {task}: no data for {', '.join(missing)}")

    n, nrows, ncols = _model_grid(all_models)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), sharey=True)
    axes = np.atleast_2d(axes)

    for idx, model in enumerate(all_models):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        x = np.arange(len(tasks))
        has_data = False

        for i, cond in enumerate(conds):
            means = []
            for task in tasks:
                vals = data[model][task][cond]
                means.append(np.mean(vals) if vals else 0)
                if vals:
                    has_data = True
            offset = (i - n_conds / 2 + 0.5) * bar_width
            ax.bar(x + offset, means, bar_width * 0.9, color=cond_colors[cond])

        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([get_task_display_name(t) for t in tasks], fontsize=11, rotation=30, ha="right")
        ax.set_title(get_model_display_name(model), fontsize=15)
        if c == 0:
            ax.set_ylabel("Target − Non-Target Drop (pp)")

        if not has_data:
            ax.text(0.5, 0.5, "No valid pairs", transform=ax.transAxes,
                    ha="center", va="center", fontsize=15, color="gray")

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, fc=cond_colors[c], label=cond_labels[c]) for c in conds]
    fig.legend(handles=handles, loc="upper center", ncol=n_conds, fontsize=11,
               bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.suptitle(f"{title} (K={k_filter}%)", fontsize=15, y=1.03)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out = output_dir / f"{fname}_k{k_filter}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out}")


def plot_target_drop_diff(records, k_filter, min_size, all_models, cross_task, output_dir):
    all_conds = ["full_circuit"] + CONDITIONS
    colors = {"full_circuit": "#2ca02c", **COND_COLORS}
    labels = {"full_circuit": "Full Circuit", **COND_DISPLAY}
    _make_diff_bars(records, k_filter, min_size, all_models, all_conds, colors, labels,
                    "Selectivity", "all_drop", output_dir, cross_task)


def plot_drop_decomposition(records, k_filter, min_size, all_models, cross_task, output_dir):
    """Per-model subplot: for each task, side-by-side bars showing full circuit drop
    vs stacked shared-core + task-specific drops. Shows whether components' drops
    account for the full circuit drop."""
    filtered = filter_by_residual_size([r for r in records if r["K"] == k_filter], min_size)

    # model -> task -> {"full": float, "shared": float, "specific": float}
    data = defaultdict(lambda: defaultdict(lambda: {"full": [], "shared": [], "specific": []}))
    all_tasks = set()

    for r in filtered:
        model, task_a = r["model_name"], r["task_a"]
        if task_a in _EXCLUDE_TASKS:
            continue
        shared = compute_drops(r, "shared_core")
        specific = compute_drops(r, "residual_a")
        if shared and specific:
            data[model][task_a]["shared"].append(shared["target_drop"])
            data[model][task_a]["specific"].append(specific["target_drop"])
            all_tasks.add(task_a)

    # Full circuit from cross-task
    for r in filtered:
        model, task_a = r["model_name"], r["task_a"]
        ct = cross_task.get((model, k_filter))
        if not ct:
            continue
        drops = ct.get("accuracy_drop_pp", {})
        if task_a in drops and task_a in drops[task_a]:
            data[model][task_a]["full"].append(drops[task_a][task_a])

    tasks = sorted(all_tasks)
    bar_width = 0.35

    # Warn about missing data
    for model in all_models:
        for task in sorted(all_tasks):
            d = data[model][task]
            missing = [k for k in ["full", "shared", "specific"] if not d[k]]
            if missing and k_filter >= 10:
                print(f"  [MISSING] drop_decomposition K={k_filter}% {model} / {task}: no data for {', '.join(missing)}")

    n, nrows, ncols = _model_grid(all_models)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), sharey=True)
    axes = np.atleast_2d(axes)

    for idx, model in enumerate(all_models):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        x = np.arange(len(tasks))

        full_vals, shared_vals, specific_vals = [], [], []
        for task in tasks:
            d = data[model][task]
            full_vals.append(np.mean(d["full"]) if d["full"] else 0)
            shared_vals.append(np.mean(d["shared"]) if d["shared"] else 0)
            specific_vals.append(np.mean(d["specific"]) if d["specific"] else 0)

        ax.bar(x - bar_width / 2, full_vals, bar_width * 0.9, color="#2ca02c", label="Full Circuit")
        ax.bar(x + bar_width / 2, shared_vals, bar_width * 0.9, color=COND_COLORS["shared_core"],
               label="Shared Core")
        ax.bar(x + bar_width / 2, specific_vals, bar_width * 0.9, color=COND_COLORS["residual_a"],
               bottom=shared_vals, label="Task-Specific")

        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([get_task_display_name(t) for t in tasks], fontsize=11, rotation=30, ha="right")
        ax.set_title(get_model_display_name(model), fontsize=15)
        if c == 0:
            ax.set_ylabel("Accuracy Drop (pp)")

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    handles = [
        plt.Rectangle((0, 0), 1, 1, fc="#2ca02c", label="Full Circuit"),
        plt.Rectangle((0, 0), 1, 1, fc=COND_COLORS["shared_core"], label="Shared Core"),
        plt.Rectangle((0, 0), 1, 1, fc=COND_COLORS["residual_a"], label="Task-Specific"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=11,
               bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.suptitle(f"Drop Decomposition (K={k_filter}%)", fontsize=15, y=1.03)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out = output_dir / f"drop_decomposition_k{k_filter}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out}")


def plot_residual_diff(records, k_filter, min_size, all_models, output_dir):
    conds = ["residual_a", "residual_b", "random_control"]
    _make_diff_bars(records, k_filter, min_size, all_models, conds, COND_COLORS, COND_DISPLAY,
                    "Selectivity", "residual_diff", output_dir)


def _plot_accuracy_bars(records, k_filter, min_size, all_models, cross_task, output_dir,
                        relative=False):
    """Per-model subplot: grouped bars showing accuracy drop for full circuit,
    shared core, and task-specific. Each condition gets one color;
    target = solid fill, non-target mean = crosshatched."""
    filtered = filter_by_residual_size([r for r in records if r["K"] == k_filter], min_size)

    conds = ["full_circuit", "shared_core", "residual_a"]
    colors = {"full_circuit": "#2ca02c", "shared_core": "#1f77b4", "residual_a": "#d62728"}
    labels = {"full_circuit": "Full Circuit", "shared_core": "Shared Core", "residual_a": "Task-Specific"}

    tgt_key = "target_rel" if relative else "target_drop"
    ntgt_key = "nontarget_rel" if relative else "nontarget_drop"
    ct_key = "relative_drop_pct" if relative else "accuracy_drop_pp"

    # model -> task -> condition -> {"target": [floats], "nontarget": [floats]}
    data = defaultdict(lambda: defaultdict(dict))
    all_tasks = set()

    for r in filtered:
        model, task_a = r["model_name"], r["task_a"]
        if task_a in _EXCLUDE_TASKS:
            continue
        for cond in ["shared_core", "residual_a"]:
            d = compute_drops(r, cond)
            if d:
                data[model][task_a].setdefault(cond, {"target": [], "nontarget": []})
                data[model][task_a][cond]["target"].append(d[tgt_key])
                data[model][task_a][cond]["nontarget"].append(d[ntgt_key])
                all_tasks.add(task_a)

    # Full circuit from cross-task
    for r in filtered:
        model, task_a = r["model_name"], r["task_a"]
        if task_a in _EXCLUDE_TASKS:
            continue
        ct = cross_task.get((model, k_filter))
        if not ct:
            continue
        drops = ct.get(ct_key, {})
        if task_a not in drops or task_a not in drops[task_a]:
            continue
        tgt = drops[task_a][task_a]
        others = [drops[task_a][t] for t in drops[task_a] if t != task_a and t not in _EXCLUDE_TASKS]
        if not others:
            continue
        data[model][task_a].setdefault("full_circuit", {"target": [], "nontarget": []})
        data[model][task_a]["full_circuit"]["target"].append(tgt)
        data[model][task_a]["full_circuit"]["nontarget"].append(np.mean(others))
        all_tasks.add(task_a)

    tasks = sorted(all_tasks)
    n_conds = len(conds)
    bar_width = 0.55 / (n_conds * 2)

    n, nrows, ncols = _model_grid(all_models)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), sharey=True)
    axes = np.atleast_2d(axes)

    for idx, model in enumerate(all_models):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        x = np.arange(len(tasks))

        for i, cond in enumerate(conds):
            base_offset = (i - n_conds / 2 + 0.5) * (bar_width * 2.2)
            for j, (kind, hatch) in enumerate([("target", None), ("nontarget", "//")]):
                vals = []
                for task in tasks:
                    d = data[model][task].get(cond)
                    if d and d[kind]:
                        vals.append(np.mean(d[kind]))
                    else:
                        vals.append(0)
                offset = base_offset + (j - 0.5) * bar_width
                ax.bar(x + offset, vals, bar_width * 0.9,
                       color=colors[cond], hatch=hatch, edgecolor="white" if not hatch else colors[cond],
                       alpha=1.0 if not hatch else 0.3, linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels([get_task_display_name(t) for t in tasks], fontsize=11, rotation=30, ha="right")
        ax.set_title(get_model_display_name(model), fontsize=15)
        if c == 0:
            ax.set_ylabel("Relative Drop (%)" if relative else "Accuracy Drop (pp)")

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, fc=colors[c], label=labels[c]) for c in conds]
    handles.append(plt.Rectangle((0, 0), 1, 1, fc="gray", label="Target"))
    handles.append(plt.Rectangle((0, 0), 1, 1, fc="gray", alpha=0.3, hatch="//",
                                 edgecolor="gray", label="Non-Target"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=11,
               bbox_to_anchor=(0.5, 1.0), frameon=False)
    suffix = "Relative" if relative else "Absolute"
    fig.suptitle(f"{suffix} Accuracy Drop (K={k_filter}%)", fontsize=15, y=1.05)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    fname = "drop_relative" if relative else "drop_absolute"
    out = output_dir / f"{fname}_k{k_filter}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out}")


def plot_raw_accuracy(records, k_filter, min_size, all_models, cross_task, output_dir):
    _plot_accuracy_bars(records, k_filter, min_size, all_models, cross_task, output_dir, relative=False)


def plot_raw_accuracy_relative(records, k_filter, min_size, all_models, cross_task, output_dir):
    _plot_accuracy_bars(records, k_filter, min_size, all_models, cross_task, output_dir, relative=True)


def plot_selectivity_diff(records, k_filter, min_size, all_models, output_dir):
    """Per-model subplot: for each target task, grouped bars showing target_drop - nontarget_drop."""
    filtered = filter_by_residual_size([r for r in records if r["K"] == k_filter], min_size)

    # Organize: model -> condition -> task_a -> list of (target_drop - nontarget_drop)
    by_model = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    all_tasks = set()
    for r in filtered:
        for cond in CONDITIONS:
            d = compute_drops(r, cond)
            if d:
                diff = d["target_drop"] - d["nontarget_drop"]
                by_model[r["model_name"]][cond][r["task_a"]].append(diff)
                all_tasks.add(r["task_a"])

    tasks = sorted(all_tasks)
    n_conds = len(CONDITIONS)
    n, nrows, ncols = _model_grid(all_models)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), sharey=True)
    axes = np.atleast_2d(axes)

    bar_width = 0.8 / n_conds
    for idx, model in enumerate(all_models):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        x = np.arange(len(tasks))

        has_data = False
        for i, cond in enumerate(CONDITIONS):
            means = []
            for task in tasks:
                vals = by_model[model][cond][task]
                means.append(np.mean(vals) if vals else 0)
                if vals:
                    has_data = True
            offset = (i - n_conds / 2 + 0.5) * bar_width
            ax.bar(x + offset, means, bar_width * 0.9, color=COND_COLORS[cond])

        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([get_task_display_name(t) for t in tasks], fontsize=11, rotation=30, ha="right")
        ax.set_title(get_model_display_name(model), fontsize=15)
        if c == 0:
            ax.set_ylabel("Target − Non-Target Drop (pp)")

        if not has_data:
            ax.text(0.5, 0.5, "No valid pairs", transform=ax.transAxes,
                    ha="center", va="center", fontsize=15, color="gray")

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, fc=COND_COLORS[c], label=COND_DISPLAY[c]) for c in CONDITIONS]
    fig.legend(handles=handles, loc="upper center", ncol=n_conds, fontsize=11,
               bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.suptitle(f"Selectivity: Target − Non-Target Drop (K={k_filter}%)", fontsize=15, y=1.03)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out = output_dir / f"selectivity_diff_k{k_filter}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out}")


COMPARE_CONDS = ["shared_core", "residual_a"]
COMPARE_COLORS = {"shared_core": "#1f77b4", "residual_a": "#2a9d8f"}


def plot_shared_vs_specific(records, k_filter, min_size, all_models, output_dir):
    """One multiplot per K. Subplots = models. x = tasks, 2 bars per task:
    shared core vs task-specific, y = target - nontarget drop."""
    filtered = filter_by_residual_size([r for r in records if r["K"] == k_filter], min_size)

    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    all_tasks = set()
    for r in filtered:
        for cond in COMPARE_CONDS:
            d = compute_drops(r, cond)
            if d:
                data[r["model_name"]][r["task_a"]][cond].append(d["target_drop"] - d["nontarget_drop"])
                all_tasks.add(r["task_a"])

    tasks = sorted(all_tasks)
    n_conds = len(COMPARE_CONDS)
    bar_width = 0.3

    n, nrows, ncols = _model_grid(all_models)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), sharey=True)
    axes = np.atleast_2d(axes)

    for idx, model in enumerate(all_models):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        x = np.arange(len(tasks))
        has_data = False

        for i, cond in enumerate(COMPARE_CONDS):
            means = []
            for task in tasks:
                vals = data[model][task][cond]
                means.append(np.mean(vals) if vals else 0)
                if vals:
                    has_data = True
            offset = (i - n_conds / 2 + 0.5) * bar_width
            ax.bar(x + offset, means, bar_width * 0.85, color=COMPARE_COLORS[cond])

        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([get_task_display_name(t) for t in tasks], fontsize=11, rotation=30, ha="right")
        ax.set_title(get_model_display_name(model), fontsize=15)
        if c == 0:
            ax.set_ylabel("Target − Non-Target Drop (pp)")

        if not has_data:
            ax.text(0.5, 0.5, "No valid pairs", transform=ax.transAxes,
                    ha="center", va="center", fontsize=15, color="gray")

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, fc=COMPARE_COLORS[c], label=COND_DISPLAY[c]) for c in COMPARE_CONDS]
    fig.legend(handles=handles, loc="upper center", ncol=n_conds, fontsize=11,
               bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.suptitle(f"Shared Core vs Task-Specific Selectivity (K={k_filter}%)", fontsize=15, y=1.03)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out = output_dir / f"shared_vs_specific_k{k_filter}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out}")


TASK_COLORS = {
    "addition": "#1f77b4",
    "arc_challenge": "#ff7f0e",
    "arc_easy": "#2ca02c",
    "boolean": "#d62728",
    "ioi": "#9467bd",
    "mcqa": "#8c564b",
}


OVER_K_CONDITIONS = ["residual_a", "residual_b", "random_control"]
OVER_K_COLORS = {
    "residual_a": "#2a9d8f",
    "residual_b": "#e76f51",
    "random_control": "#7f7f7f",
}


def plot_target_drop_over_k(records, k_values, min_size, all_models, output_dir):
    """One figure per model. Subplots are tasks. Three lines per task:
    task-specific, task-complement, random control — all measuring drop on the target task."""
    # model -> task_a -> condition -> K -> list of target drops
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    all_tasks = set()

    for r in records:
        if r["circuit_sizes"].get("residual_a", 0) < min_size:
            continue
        for cond in OVER_K_CONDITIONS:
            d = compute_drops(r, cond)
            if d:
                data[r["model_name"]][r["task_a"]][cond][r["K"]].append(d["target_drop"])
                all_tasks.add(r["task_a"])

    tasks = sorted(all_tasks)
    n_tasks = len(tasks)
    ncols = min(n_tasks, 3)
    nrows = math.ceil(n_tasks / ncols)

    for model in all_models:
        model_slug = model.replace("/", "_")
        model_disp = get_model_display_name(model)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), sharey=True)
        axes = np.atleast_2d(axes)

        n_conds = len(OVER_K_CONDITIONS)
        bar_width = 0.8 / n_conds

        for idx, task in enumerate(tasks):
            r, c = divmod(idx, ncols)
            ax = axes[r, c]
            has_data = False
            x = np.arange(len(k_values))

            for i, cond in enumerate(OVER_K_CONDITIONS):
                means = []
                for k in k_values:
                    vals = data[model][task][cond][k]
                    if vals:
                        means.append(np.mean(vals))
                        has_data = True
                    else:
                        means.append(0)
                offset = (i - n_conds / 2 + 0.5) * bar_width
                ax.bar(x + offset, means, bar_width * 0.9, color=OVER_K_COLORS[cond])

            ax.axhline(0, color="black", linewidth=0.5)
            ax.set_title(get_task_display_name(task), fontsize=15)
            ax.set_xlabel("Top-K (%)")
            if c == 0:
                ax.set_ylabel("Target task drop (pp)")
            ax.set_xticks(x)
            ax.set_xticklabels([f"{k}%" for k in k_values], fontsize=11)

            if not has_data:
                ax.text(0.5, 0.5, "No valid pairs", transform=ax.transAxes,
                        ha="center", va="center", fontsize=15, color="gray")

        for idx in range(n_tasks, nrows * ncols):
            r, c = divmod(idx, ncols)
            axes[r, c].set_visible(False)

        handles = [plt.Line2D([0], [0], color=OVER_K_COLORS[c], marker="o", markersize=4,
                   linewidth=1.5, label=COND_DISPLAY[c]) for c in OVER_K_CONDITIONS]
        fig.legend(handles=handles, loc="upper center", ncol=len(OVER_K_CONDITIONS), fontsize=11,
                   bbox_to_anchor=(0.5, 1.0), frameon=False)
        fig.suptitle(f"{model_disp}: Target Task Drop over K", fontsize=15, y=1.04)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        out = output_dir / f"target_drop_over_k_{model_slug}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[SAVED] {out}")


def plot_selectivity_bars(records, k_filter, min_size, all_models, output_dir):
    """Multiplot: one subplot per task. x = models, grouped bars per condition showing selectivity."""
    filtered = filter_by_residual_size([r for r in records if r["K"] == k_filter], min_size)

    # model -> task_a -> condition -> list of drops
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    all_tasks = set()
    for r in filtered:
        for cond in CONDITIONS:
            d = compute_drops(r, cond)
            if d:
                data[r["model_name"]][r["task_a"]][cond].append(d)
                all_tasks.add(r["task_a"])

    tasks = sorted(all_tasks)
    n_tasks = len(tasks)
    ncols = min(n_tasks, 3)
    nrows = math.ceil(n_tasks / ncols)
    n_conds = len(CONDITIONS)
    width = 0.8 / n_conds

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), sharey=True)
    axes = np.atleast_2d(axes)

    for idx, task in enumerate(tasks):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        x = np.arange(len(all_models))
        has_data = False

        for i, cond in enumerate(CONDITIONS):
            vals = []
            for model in all_models:
                drops = data[model][task][cond]
                sel = ratio_of_means(drops) if drops else 0
                vals.append(sel if np.isfinite(sel) else 0)
                if drops:
                    has_data = True
            offset = (i - n_conds / 2 + 0.5) * width
            ax.bar(x + offset, vals, width * 0.9, color=COND_COLORS[cond])

        ax.axhline(y=1, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([get_model_display_name(m) for m in all_models],
                           fontsize=11, rotation=30, ha="right")
        ax.set_title(get_task_display_name(task), fontsize=15)
        if c == 0:
            ax.set_ylabel("Selectivity")

        if not has_data:
            ax.text(0.5, 0.5, "No valid pairs", transform=ax.transAxes,
                    ha="center", va="center", fontsize=15, color="gray")

    for idx in range(n_tasks, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, fc=COND_COLORS[c], label=COND_DISPLAY[c]) for c in CONDITIONS]
    fig.legend(handles=handles, loc="upper center", ncol=n_conds, fontsize=11,
               bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.suptitle(f"Selectivity by Task (K={k_filter}%)", fontsize=15, y=1.03)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out = output_dir / f"selectivity_bars_k{k_filter}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out}")


# --- Circuit decomposition sizes ---

DECOMP_COLORS = {
    "shared_core": "#264653",
    "residual_a": "#E76F51",
    "residual_b": "#2A9D8F",
}


def plot_circuit_decomposition(records, k_filter, all_models, output_dir):
    """Stacked bars: mean shared core / R_A / R_B sizes per task, one subplot per model."""
    filtered = [r for r in records if r["K"] == k_filter]
    if not filtered:
        return

    # Aggregate: for each (model, task_a), average the decomposition sizes across all task_b
    # model -> task -> {shared_core: [...], residual_a: [...], residual_b: [...]}
    agg = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    all_tasks = set()
    for r in filtered:
        task_a = r["task_a"]
        if task_a in _EXCLUDE_TASKS:
            continue
        cs = r["circuit_sizes"]
        for comp in ["shared_core", "residual_a", "residual_b"]:
            agg[r["model_name"]][task_a][comp].append(cs.get(comp, 0))
        all_tasks.add(task_a)

    tasks = sorted(all_tasks)
    models = [m for m in all_models if m in agg]
    if not models:
        return

    ncols = min(len(models), 3)
    nrows = math.ceil(len(models) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    components = ["shared_core", "residual_a", "residual_b"]
    comp_labels = ["Shared Core", "Task-Specific (R_A)", "Task-Complement (R_B)"]

    for idx, model in enumerate(models):
        ax = axes[idx // ncols][idx % ncols]
        x = np.arange(len(tasks))

        bottom = np.zeros(len(tasks))
        for comp, label in zip(components, comp_labels):
            means = []
            for task in tasks:
                vals = agg[model][task][comp]
                means.append(np.mean(vals) if vals else 0)
            means = np.array(means)
            ax.bar(x, means, bottom=bottom, color=DECOMP_COLORS[comp],
                   label=label, edgecolor="white", linewidth=0.5, alpha=0.85)
            bottom += means

        ax.set_xticks(x)
        ax.set_xticklabels([get_task_display_name(t) for t in tasks],
                           fontsize=10, rotation=30, ha="right")
        ax.set_title(get_model_display_name(model), fontsize=15)
        if idx % ncols == 0:
            ax.set_ylabel("Mean # Components", fontsize=13)
        ax.tick_params(labelsize=10)

    for k in range(len(models), nrows * ncols):
        fig.delaxes(axes[k // ncols][k % ncols])

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=12,
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"Circuit Decomposition Sizes (K={k_filter}%)", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])

    out = output_dir / f"circuit_decomposition_k{k_filter}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out}")


def plot_circuit_decomposition_over_k(records, k_values, all_models, output_dir):
    """For each model: subplots=tasks, x-axis=K%, stacked bars of decomposition sizes."""
    all_tasks = set()
    # model -> task -> K -> {comp: mean_size}
    agg = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for r in records:
        task_a = r["task_a"]
        if task_a in _EXCLUDE_TASKS:
            continue
        all_tasks.add(task_a)
        cs = r["circuit_sizes"]
        model = r["model_name"]
        k = r["K"]
        for comp in ["shared_core", "residual_a", "residual_b"]:
            agg[model][task_a].setdefault(k, defaultdict(list))[comp].append(cs.get(comp, 0))

    tasks = sorted(all_tasks)
    components = ["shared_core", "residual_a", "residual_b"]
    comp_labels = ["Shared Core", "Task-Specific (R_A)", "Task-Complement (R_B)"]

    for model in [m for m in all_models if m in agg]:
        model_disp = get_model_display_name(model)

        ncols = min(len(tasks), 3)
        nrows = math.ceil(len(tasks) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)

        for idx, task in enumerate(tasks):
            ax = axes[idx // ncols][idx % ncols]
            available_ks = sorted(k for k in k_values if k in agg[model][task])
            x = np.arange(len(available_ks))

            bottom = np.zeros(len(available_ks))
            for comp, label in zip(components, comp_labels):
                means = []
                for k in available_ks:
                    vals = agg[model][task][k][comp]
                    means.append(np.mean(vals) if vals else 0)
                means = np.array(means)
                ax.bar(x, means, bottom=bottom, color=DECOMP_COLORS[comp],
                       label=label, edgecolor="white", linewidth=0.5, alpha=0.85)
                bottom += means

            ax.set_xticks(x)
            ax.set_xticklabels([f"{k}%" for k in available_ks], fontsize=10)
            ax.set_title(get_task_display_name(task), fontsize=15)
            if idx // ncols == nrows - 1:
                ax.set_xlabel("Top-K%", fontsize=13)
            if idx % ncols == 0:
                ax.set_ylabel("Mean # Components", fontsize=13)
            ax.tick_params(labelsize=10)

        for k in range(len(tasks), nrows * ncols):
            fig.delaxes(axes[k // ncols][k % ncols])

        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=12,
                   bbox_to_anchor=(0.5, -0.06))
        fig.suptitle(f"{model_disp}: Circuit Decomposition over K",
                     fontsize=18, fontweight="bold")
        fig.tight_layout(rect=[0, 0.02, 1, 0.95])

        safe = model.replace("/", "_")
        out = output_dir / f"circuit_decomposition_over_k_{safe}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[SAVED] {out}")


# --- CSV ---

def save_csv(records: list[dict], output_dir: Path):
    out = output_dir / "selective_ablation_summary.csv"
    cols = ["model", "K", "task_a", "task_b", "condition", "condition_label", "size",
            "target_drop_pp", "nontarget_mean_drop_pp", "target_rel_pct", "nontarget_rel_pct"]
    lines = [",".join(cols)]

    for r in records:
        for cond in CONDITIONS:
            d = compute_drops(r, cond)
            size = r["conditions"].get(cond, {}).get("size", 0)
            lines.append(",".join([
                r["model_name"], str(r["K"]), r["task_a"], r["task_b"],
                cond, COND_DISPLAY[cond], str(size),
                f"{d['target_drop']:.2f}" if d else "0.00",
                f"{d['nontarget_drop']:.2f}" if d else "0.00",
                f"{d['target_rel']:.2f}" if d else "0.00",
                f"{d['nontarget_rel']:.2f}" if d else "0.00",
            ]))

    out.write_text("\n".join(lines) + "\n")
    print(f"[SAVED] {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dirs", type=str,
                    default="results/selective_ablation_k1,results/selective_ablation_k5,"
                            "results/selective_ablation_k10,results/selective_ablation_k20,"
                            "results/selective_ablation_k30")
    p.add_argument("--cross-task-dir", type=str, default="results2")
    p.add_argument("--output-dir", type=str, default="results2/selective_ablation")
    p.add_argument("--k-values", type=str, default="1,5,10,20,30")
    p.add_argument("--min-residual-size", type=int, default=2)
    p.add_argument("--exclude-tasks", type=str, default="arc_easy",
                    help="Comma-separated tasks to exclude from analysis.")
    args = p.parse_args()

    global _EXCLUDE_TASKS
    _EXCLUDE_TASKS = {t.strip() for t in args.exclude_tasks.split(",") if t.strip()} if args.exclude_tasks else set()
    results_dirs = [Path(d.strip()) for d in args.results_dirs.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    k_values = [int(x) for x in args.k_values.split(",")]
    min_size = args.min_residual_size

    records = [r for r in load_all_results(results_dirs)
                if r["task_a"] not in _EXCLUDE_TASKS and r["task_b"] not in _EXCLUDE_TASKS]
    print(f"Loaded {len(records)} selective ablation files (excluding {_EXCLUDE_TASKS or 'none'})")

    all_models = get_all_models(records)

    ct_base = Path(args.cross_task_dir)
    ct_dirs = [ct_base / f"cross_task_ablation_k{k}" for k in k_values]
    cross_task = load_all_cross_task(ct_dirs)
    print(f"Loaded {len(cross_task)} cross-task ablation files")

    save_csv(records, output_dir)

    # Create output subdirs
    dirs = {}
    per_cond_folders = {
        "shared_core": "shared_core",
        "residual_a": "task_specific",
        "residual_b": "task_complement",
    }
    for name in ["drop_diff", "all_drop", "residual_diff", "raw_accuracy", "raw_accuracy_relative", "drop_decomposition", "circuit_decomposition"] + list(per_cond_folders.values()):
        dirs[name] = output_dir / name
        dirs[name].mkdir(parents=True, exist_ok=True)

    md_parts = ["# Selective Ablation Results", ""]
    md_parts.append(md_compact_tables(records, k_values, min_size, all_models))

    for k in k_values:
        if not any(r["K"] == k for r in records):
            continue

        plot_drop_diff(records, k, min_size, all_models, dirs["drop_diff"])
        plot_target_drop_diff(records, k, min_size, all_models, cross_task, dirs["all_drop"])
        plot_residual_diff(records, k, min_size, all_models, dirs["residual_diff"])
        plot_drop_decomposition(records, k, min_size, all_models, cross_task, dirs["drop_decomposition"])
        plot_raw_accuracy(records, k, min_size, all_models, cross_task, dirs["raw_accuracy"])
        plot_raw_accuracy_relative(records, k, min_size, all_models, cross_task, dirs["raw_accuracy_relative"])
        plot_circuit_decomposition(records, k, all_models, dirs["circuit_decomposition"])
        for cond, folder in per_cond_folders.items():
            plot_condition_target_vs_nontarget(records, k, min_size, all_models, cond, dirs[folder])

    # Also generate per-model over-K decomposition plots
    plot_circuit_decomposition_over_k(records, k_values, all_models, dirs["circuit_decomposition"])

    md_path = output_dir / "selective_ablation_results.md"
    md_path.write_text("\n".join(md_parts))
    print(f"[SAVED] {md_path}")

    print("\n".join(md_parts))


if __name__ == "__main__":
    main()
