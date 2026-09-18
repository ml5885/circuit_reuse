"""Exploratory sweep plots for the paper2 results section.

Nothing here is a paper figure. The point is to see every metric at every value
of K and P without averaging over model, task, granularity, or method, so we can
decide which dimensions can be collapsed and which need their own figure.

Empty shared circuits are recorded with reuse and lift equal to zero, so they
are masked out of every metric and reported separately as coverage.

Run: python -m analysis.explore_sweep
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "circuit_reuse_mplconfig"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import itertools
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy.stats import spearmanr

from analysis.granularity_parity import (CONFIG_ORDER, EXCLUDED_MODELS,
                                         EXCLUDED_TASKS, GRAN_COLORS, OTHER_COLOR,
                                         _label, _legend, _model_label, _ramp,
                                         _task_label, infer_granularity)

REPO = Path(__file__).resolve().parent.parent
ANA = REPO / "results" / "granularity_parity_analysis"
RAW = REPO / "results" / "granularity_parity"
OUT = REPO / "paper2" / "explore"
OUT.mkdir(parents=True, exist_ok=True)

TASKS = ["addition", "boolean", "ioi", "mcqa", "arc_easy", "arc_challenge"]
MODELS = ["google/gemma-2-2b", "google/gemma-2-2b-it", "meta-llama/Llama-3.2-3B",
          "meta-llama/Llama-3.2-3B-Instruct", "qwen3-4b"]
KS = [1, 5, 10, 20, 30]
PS = [50, 75, 85, 90, 95, 96, 97, 98, 99, 100]
P_COARSE = [50, 85, 100]
K_REF, P_REF = 10, 50

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                     "axes.titlesize": 11, "axes.labelsize": 11,
                     "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
                     "figure.titlesize": 14,
                     "legend.fontsize": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": .25, "grid.linewidth": .5,
                     "axes.axisbelow": True})

P_CMAP = plt.get_cmap("viridis")
K_CMAP = plt.get_cmap("plasma")
# Diverging map for signed necessity, after the Life Aquatic palette: Zissou
# blue for positive, yellow through red for negative, paper canvas at zero.
NECESSITY_CMAP = LinearSegmentedColormap.from_list(
    "zissou", ["#D9401E", "#EBA22E", "#F5C061", "#F4F2ED", "#9FD1E6", "#3B9AB2", "#1B5E80"])

# Colour carries the granularity and line style carries the method, so a reader
# needs two facts rather than four. Light tints of each hue are unnecessary once
# the style does the second job.
GRAN_NAME = {"head_mlp": "Component-level", "neuron": "Neuron-level"}
HEATMAP_GRAN_NAME = {"head_mlp": "Attention Heads and MLPs", "neuron": "MLP Neurons"}
METHOD_NAME = {"eap_ig": "EAP-IG", "relp": "RelP"}
METHOD_STYLE = {"eap_ig": "-", "relp": (0, (3, 1.4))}
METHOD_MARK = {"eap_ig": "o", "relp": "^"}
METHOD_HATCH = {"eap_ig": "", "relp": "///"}


def cfg_color(key) -> str:
    return GRAN_COLORS[key[1]]


def cfg_line(key) -> dict:
    return {"color": GRAN_COLORS[key[1]], "ls": METHOD_STYLE[key[0]]}


def config_legend(fig, style_key="line", extra=()):
    """Two colours for the granularities, two styles for the methods."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Line2D([], [], color=c, lw=2.4, label=n)
               for c, n in ((GRAN_COLORS[g], GRAN_NAME[g]) for g in ("head_mlp", "neuron"))]
    for m in ("eap_ig", "relp"):
        if style_key == "bar":
            handles.append(Patch(facecolor="0.75", edgecolor="white",
                                 hatch=METHOD_HATCH[m], label=METHOD_NAME[m]))
            continue
        handles.append(Line2D([], [], color="0.35", lw=1.6, label=METHOD_NAME[m],
                              **({"ls": METHOD_STYLE[m]} if style_key == "line"
                                 else {"ls": "none", "marker": METHOD_MARK[m], "ms": 5})))
    handles += list(extra)
    _legend(fig, handles=handles, labels=[h.get_label() for h in handles],
            ncol=len(handles), loc="outside lower center")


def p_color(p: int) -> str:
    return P_CMAP((PS.index(p)) / (len(PS) - 1) * .9)


def save(fig, name: str):
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


def grid(nrow: int, ncol: int, w: float = 2.4, h: float = 1.95, **kw):
    fig, axes = plt.subplots(nrow, ncol, figsize=(w * ncol, h * nrow), squeeze=False,
                             layout="constrained", **kw)
    fig.get_layout_engine().set(h_pad=.07, w_pad=.05, hspace=.03, wspace=.03)
    return fig, axes


def hide_inner_y(axes):
    for row in axes:
        for ax in row[1:]:
            ax.tick_params(labelleft=False)


def row_label(ax, text: str):
    """Name a row along its right edge, leaving the y axis for the quantity."""
    ax.text(1.04, .5, text, transform=ax.transAxes, rotation=270, va="center",
            ha="left", fontsize=10.5)


def label_grid(axes, xlabel: str, ylabel: str):
    """One label for the quantity, row names on the right, model names on top."""
    axes[0][0].figure.supylabel(ylabel, fontsize=12)
    for c, model in enumerate(MODELS):
        axes[0][c].set_title(_model_label(model))
        axes[-1][c].set_xlabel(xlabel)
    for r, task in enumerate(TASKS):
        row_label(axes[r][-1], _task_label(task))


def k_axis(ax):
    ax.set_xscale("log")
    ax.set_xticks(KS, [str(k) for k in KS])
    ax.minorticks_off()


# --- data ---------------------------------------------------------------

extraction = pd.read_csv(ANA / "extraction_tidy.csv")
cross = pd.read_csv(ANA / "cross_task_tidy.csv")
extraction["nonempty"] = extraction.circuit_size > 0
ext = extraction.copy()
ext.loc[~ext.nonempty, ["reuse", "lift", "necessity_gap"]] = np.nan

# One row per (model, config, K, p, target): the cross-task file repeats the own
# and foreign summaries on every donor row of a cell.
spec = (cross[cross.donor == cross.target]
        .rename(columns={"target": "task"})
        [["model", "method", "granularity", "K", "p", "task", "donor_size",
          "own_drop_pp", "foreign_mean_drop_pp", "specificity_gap_pp",
          "baseline_accuracy"]])
spec.loc[spec.donor_size == 0, ["own_drop_pp", "specificity_gap_pp"]] = np.nan

cells = ext.merge(spec.drop(columns=["donor_size", "baseline_accuracy"]),
                  on=["model", "method", "granularity", "K", "p", "task"], how="outer")

METRICS = {
    "reuse": ("reuse", "Reuse@$P$ (%)", ext),
    "lift": ("lift", "Necessity (%)", ext),
    "size": ("circuit_size", "Shared circuit size\n(components)", extraction),
    "specgap": ("specificity_gap_pp", "Specificity gap\n(accuracy points)", spec),
    "own": ("own_drop_pp", "Own-circuit accuracy drop\n(points)", spec),
}

# --- 0. the headline comparison, one bar per cell ------------------------

def fig_bars(metric: str, ps=(50, 75, 95, 100)):
    """One bar per cell, so the comparison is made without averaging anything.

    A missing bar is a cell whose shared circuit is empty at that P, so the rows
    show both the metric and how much of the sample is left to read it from.
    """
    column, ylabel, source = METRICS[metric]
    if metric == "reuse":
        ylabel = "Reuse@$P$ (%), at the $P$ of each row"
    fig, axes = grid(len(ps), len(MODELS), w=2.6, h=1.9, sharex=True, sharey=True)
    x = np.arange(len(TASKS))
    width = .8 / len(CONFIG_ORDER)
    for r, p in enumerate(ps):
        for c, model in enumerate(MODELS):
            ax = axes[r][c]
            sub = source[(source.model == model) & (source.K == K_REF) & (source.p == p)]
            for i, key in enumerate(CONFIG_ORDER):
                d = (sub[(sub.method == key[0]) & (sub.granularity == key[1])]
                     .set_index("task")[column])
                ax.bar(x + (i - (len(CONFIG_ORDER) - 1) / 2) * width,
                       [d.get(t, np.nan) for t in TASKS], width, color=cfg_color(key),
                       hatch=METHOD_HATCH[key[0]], edgecolor="white", linewidth=.5)
            ax.axhline(0, c="0.35", lw=.7, zorder=0)
            # Only the bottom row carries the task names; ten rotated copies
            # would take more height than the bars.
            ax.set_xticks(x, [_task_label(t) for t in TASKS] if r == len(ps) - 1 else [],
                          rotation=55, ha="right", fontsize=9)
            if r == 0:
                ax.set_title(_model_label(model))
            if c == len(MODELS) - 1:
                row_label(ax, f"$P$={p}%")
    fig.supylabel(ylabel.replace(chr(10), " "), fontsize=12)
    config_legend(fig, style_key="bar")
    fig.suptitle(f"{ylabel.replace(chr(10), ' ')} per model and task, top-$K$={K_REF}%")
    save(fig, f"00_bars_{metric}")


# --- 1. coverage: where the shared set is empty --------------------------

def fig_coverage():
    fig, axes = grid(len(KS), len(MODELS), sharex=True, sharey=True)
    for r, K in enumerate(KS):
        for c, model in enumerate(MODELS):
            ax = axes[r][c]
            sub = extraction[(extraction.K == K) & (extraction.model == model)]
            for key in CONFIG_ORDER:
                s = sub[(sub.method == key[0]) & (sub.granularity == key[1])]
                frac = s.groupby("p").nonempty.mean()
                ax.plot(frac.index, frac.values * 100, lw=1.4, marker="o", ms=2.5,
                        **cfg_line(key))
            ax.axvline(P_REF, ls=":", c="grey", lw=.8, zorder=0)
            ax.set_ylim(-5, 105)
            if r == 0:
                ax.set_title(_model_label(model))
            if c == len(MODELS) - 1:
                row_label(ax, f"top-$K$={K}%")
            if r == len(KS) - 1:
                ax.set_xlabel("$P$ (%)")
    fig.supylabel("Tasks with a non-empty shared circuit (%)", fontsize=12)
    config_legend(fig)
    fig.suptitle("Percent of the six tasks whose shared circuit is not empty")
    save(fig, "01_coverage_nonempty")


def fig_vs_p_by_config(metric: str):
    """One panel per task and model, with the four configurations side by side."""
    column, ylabel, source = METRICS[metric]
    fig, axes = grid(len(TASKS), len(MODELS), sharex=True, sharey=True)
    for r, task in enumerate(TASKS):
        for c, model in enumerate(MODELS):
            ax = axes[r][c]
            sub = source[(source.task == task) & (source.model == model)
                         & (source.K == K_REF)]
            for key in CONFIG_ORDER:
                s = sub[(sub.method == key[0]) & (sub.granularity == key[1])].sort_values("p")
                ax.plot(s.p, s[column], lw=1.4, marker="o", ms=2.5, **cfg_line(key))
            if metric == "size":
                ax.set_yscale("symlog", linthresh=1)
            ax.axvline(P_REF, ls=":", c="grey", lw=.8, zorder=0)
    hide_inner_y(axes)
    label_grid(axes, "$P$ (%)", ylabel)
    config_legend(fig)
    fig.suptitle(f"{ylabel.replace(chr(10), ' ')} across $P$, top-$K$={K_REF}%")
    save(fig, f"02_vs_p_by_config_{metric}")


def fig_kp_heatmap(metric: str):
    """Both sweeps on one panel: top-K up the rows, P along the columns.

    The line figures fix one of the two and sweep the other. This shows the
    whole surface for every cell, so an interaction between top-K and P is
    visible rather than inferred from two slices.

    An empty shared circuit is a hole in the surface, not a low value, so it is
    drawn as hatched background rather than as a tint the colour ramp also uses.
    """
    column, ylabel, source = METRICS[metric]
    if metric == "lift":
        # Necessity is signed, so it diverges through the canvas at zero. The
        # range is clipped to [-50, 100], which cuts a few dozen of ~3900 cells.
        norm = TwoSlopeNorm(vmin=-50, vcenter=0, vmax=100)
        cmap = NECESSITY_CMAP.with_extremes(bad=(0, 0, 0, 0))
        cbar_kw = {"ticks": [-50, 0, 50, 100]}
    else:
        vmax = 100 if metric != "size" else np.nanpercentile(source[column], 99)
        norm = plt.Normalize(0, vmax)
        cmap = _ramp().with_extremes(bad=(0, 0, 0, 0))
        cbar_kw = {}
    for key in CONFIG_ORDER:
        df = source[(source.method == key[0]) & (source.granularity == key[1])]
        fig, axes = grid(len(TASKS), len(MODELS), sharex=True, sharey=True)
        for r, task in enumerate(TASKS):
            for c, model in enumerate(MODELS):
                ax = axes[r][c]
                ax.set_facecolor("0.93")
                ax.patch.set_hatch("//")
                ax.patch.set_edgecolor("0.78")
                surface = (df[(df.task == task) & (df.model == model)]
                           .pivot_table(index="K", columns="p", values=column)
                           .reindex(index=KS, columns=PS)
                           .clip(norm.vmin, norm.vmax))
                im = ax.imshow(surface.values, aspect="auto", origin="lower",
                               cmap=cmap, norm=norm, interpolation="nearest")
                ax.set_xticks([PS.index(p) for p in (50, 75, 90, 100)],
                              ["50", "75", "90", "100"])
                ax.set_yticks(range(len(KS)), [str(k) for k in KS])
                ax.grid(False)
        label_grid(axes, "$P$ (%)", "Top-$K$ (%)")
        cbar = fig.colorbar(im, ax=axes, location="bottom", fraction=.02, pad=.022,
                            shrink=.5, aspect=45, **cbar_kw)
        cbar.set_label(ylabel.replace(chr(10), " "), fontsize=11)
        fig.suptitle(f"{METHOD_NAME[key[0]]} \u2013 {HEATMAP_GRAN_NAME[key[1]]}")
        save(fig, f"09_kp_heatmap_{metric}__{key[0]}_{key[1]}")


# --- 2. the core sweep: every metric at every (K, P) ---------------------

def fig_sweep(metric: str):
    column, ylabel, source = METRICS[metric]
    for key in CONFIG_ORDER:
        df = source[(source.method == key[0]) & (source.granularity == key[1])]
        fig, axes = grid(len(TASKS), len(MODELS), sharex=True, sharey=True)
        for r, task in enumerate(TASKS):
            for c, model in enumerate(MODELS):
                ax = axes[r][c]
                sub = df[(df.task == task) & (df.model == model)]
                for p in PS:
                    s = sub[sub.p == p].sort_values("K")
                    ax.plot(s.K, s[column], lw=1.2, marker="o", ms=2.2,
                            color=p_color(p), label=f"$P$={p}%" if p in P_COARSE else None)
                k_axis(ax)
                if metric in ("specgap", "lift"):
                    ax.axhline(0, c="0.35", lw=.7, zorder=0)
                if metric == "size":
                    ax.set_yscale("symlog", linthresh=1)
        hide_inner_y(axes)
        label_grid(axes, "Top-$K$ (%)", ylabel)
        norm = plt.Normalize(0, len(PS) - 1)
        sm = plt.cm.ScalarMappable(cmap=P_CMAP.resampled(len(PS)), norm=norm)
        cbar = fig.colorbar(sm, ax=axes, location="bottom", fraction=.025, pad=.022,
                            shrink=.5, aspect=45, ticks=range(len(PS)))
        cbar.ax.set_xticklabels([str(p) for p in PS], fontsize=9)
        cbar.set_label("Consensus threshold $P$ (%)", fontsize=11)
        fig.suptitle(f"{_label(key)}")
        save(fig, f"03_sweep_{metric}__{key[0]}_{key[1]}")


def fig_sweep_vs_p(metric: str):
    """The same grid with the axes swapped, so a P trend can be read directly.

    Reading a trend in P off a colour ramp is harder than reading it off an
    axis, and P is the threshold that decides whether a circuit exists at all.
    """
    column, ylabel, source = METRICS[metric]
    for key in CONFIG_ORDER:
        df = source[(source.method == key[0]) & (source.granularity == key[1])]
        fig, axes = grid(len(TASKS), len(MODELS), sharex=True, sharey=True)
        for r, task in enumerate(TASKS):
            for c, model in enumerate(MODELS):
                ax = axes[r][c]
                sub = df[(df.task == task) & (df.model == model)]
                for i, K in enumerate(KS):
                    s = sub[sub.K == K].sort_values("p")
                    ax.plot(s.p, s[column], lw=1.2, marker="o", ms=2.2,
                            color=K_CMAP(i / (len(KS) - 1) * .9), label=f"top-$K$={K}%")
                ax.axvline(P_REF, ls=":", c="grey", lw=.8, zorder=0)
                if metric == "size":
                    ax.set_yscale("symlog", linthresh=1)
        hide_inner_y(axes)
        label_grid(axes, "$P$ (%)", ylabel)
        handles, labels = axes[0][0].get_legend_handles_labels()
        _legend(fig, handles=handles, labels=labels, ncol=len(KS), loc="outside lower center")
        fig.suptitle(f"{ylabel.replace(chr(10), ' ')} across $P$ — {_label(key)}")
        save(fig, f"08_vs_p_{metric}__{key[0]}_{key[1]}")


# --- 3. own against foreign, not only the gap ----------------------------

def fig_own_foreign():
    for key in CONFIG_ORDER:
        df = spec[(spec.method == key[0]) & (spec.granularity == key[1])]
        fig, axes = grid(len(TASKS), len(MODELS), sharex=True, sharey=True)
        for r, task in enumerate(TASKS):
            for c, model in enumerate(MODELS):
                ax = axes[r][c]
                sub = df[(df.task == task) & (df.model == model)]
                for p in P_COARSE:
                    s = sub[sub.p == p].sort_values("K")
                    ax.plot(s.K, s.own_drop_pp, lw=1.4, marker="o", ms=2.4,
                            color=cfg_color(key), alpha=.35 + .3 * P_COARSE.index(p),
                            label=f"own, $P$={p}%")
                    ax.plot(s.K, s.foreign_mean_drop_pp, lw=1.4, marker="^", ms=2.4,
                            ls=(0, (3, 1.4)), color=OTHER_COLOR,
                            alpha=.35 + .3 * P_COARSE.index(p),
                            label=f"other, $P$={p}%")
                k_axis(ax)
                ax.axhline(0, c="0.35", lw=.7, zorder=0)
        label_grid(axes, "Top-$K$ (%)", "Accuracy drop\n(points)")
        handles, labels = axes[0][0].get_legend_handles_labels()
        _legend(fig, handles=handles, labels=labels, ncol=3, loc="outside lower center")
        fig.suptitle(f"Accuracy drop from ablating a task's own circuit against "
                     f"other tasks' circuits — {_label(key)}")
        save(fig, f"04_own_foreign__{key[0]}_{key[1]}")


# --- 4. does the headline reversal survive every K and P? ----------------

def fig_plane_by_kp():
    fig, axes = grid(len(KS), len(P_COARSE), w=2.5, h=2.2, sharex=True, sharey=True)
    for r, K in enumerate(KS):
        for c, p in enumerate(P_COARSE):
            ax = axes[r][c]
            sub = cells[(cells.K == K) & (cells.p == p)]
            for key in CONFIG_ORDER:
                s = sub[(sub.method == key[0]) & (sub.granularity == key[1])]
                ax.scatter(s.reuse, s.specificity_gap_pp, s=15, alpha=.75,
                           color=cfg_color(key), marker=METHOD_MARK[key[0]],
                           edgecolor="none")
            ax.axhline(0, c="0.35", lw=.7, zorder=0)
            if r == 0:
                ax.set_title(f"$P$={p}%")
            if c == len(P_COARSE) - 1:
                row_label(ax, f"top-$K$={K}%")
            if r == len(KS) - 1:
                ax.set_xlabel("Reuse@$P$ (%)")
    fig.supylabel("Specificity gap (accuracy points)", fontsize=12)
    config_legend(fig, style_key="marker")
    fig.suptitle("Consistency against specificity, one point per model-task cell")
    save(fig, "05_plane_by_kp")


# --- 5. is averaging over K and P safe? ----------------------------------

def fig_avg_check():
    metrics = ["reuse", "lift", "specgap"]
    fig, axes = grid(1, len(metrics), w=3.1, h=3.1)
    for c, metric in enumerate(metrics):
        column, ylabel, source = METRICS[metric]
        ax = axes[0][c]
        ref = source[(source.K == K_REF) & (source.p == P_REF)]
        keys = ["model", "task", "method", "granularity"]
        avg = source.groupby(keys)[column].mean().rename("avg")
        merged = ref.set_index(keys)[[column]].join(avg).dropna()
        for key in CONFIG_ORDER:
            s = merged.xs(key, level=("method", "granularity"))
            ax.scatter(s[column], s.avg, s=18, alpha=.75, color=cfg_color(key),
                       marker=METHOD_MARK[key[0]], edgecolor="none")
        lo, hi = merged[[column, "avg"]].min().min(), merged[[column, "avg"]].max().max()
        ax.plot([lo, hi], [lo, hi], c="0.4", lw=.8, ls=":", zorder=0)
        rho = spearmanr(merged[column], merged.avg).statistic
        ax.set(xlabel=f"at $K$={K_REF}%, $P$={P_REF}%", ylabel="mean over all $K$, $P$",
               title=f"{ylabel.replace(chr(10), ' ')}\nSpearman $\\rho$={rho:.3f}")
    config_legend(fig, style_key="marker")
    fig.suptitle(f"$K$={K_REF}%, $P$={P_REF}% against the $K$, $P$ average")
    save(fig, "06_avg_check")


def variation_table() -> pd.DataFrame:
    """Spread attributable to K and P inside a cell, against spread across cells.

    If the within-cell sweep spread is small relative to the spread across tasks
    and models, collapsing K and P loses little. Everything is masked to
    non-empty circuits, so this measures variation in the metric rather than
    variation in whether the metric exists.
    """
    rows = []
    for metric in ["reuse", "lift", "specgap", "own"]:
        column, _, source = METRICS[metric]
        df = source.dropna(subset=[column])
        for key in CONFIG_ORDER:
            d = df[(df.method == key[0]) & (df.granularity == key[1])]
            across_k = d[d.p == P_REF].groupby(["model", "task"])[column].std()
            across_p = d[d.K == K_REF].groupby(["model", "task"])[column].std()
            ref = d[(d.K == K_REF) & (d.p == P_REF)]
            rows.append({
                "metric": metric, "method": key[0], "granularity": key[1],
                "sd_across_K": across_k.median(), "sd_across_P": across_p.median(),
                "sd_across_tasks": ref.groupby("model")[column].std().median(),
                "sd_across_models": ref.groupby("task")[column].std().median(),
                "range_over_K": d[d.p == P_REF].groupby(["model", "task"])[column]
                                 .agg(lambda s: s.max() - s.min()).median(),
                "range_over_P": d[d.K == K_REF].groupby(["model", "task"])[column]
                                 .agg(lambda s: s.max() - s.min()).median(),
            })
    return pd.DataFrame(rows).round(2)


# --- 6. cross-task overlap at every (K, P) -------------------------------

RELATED = ("arc_challenge", "arc_easy")


def overlap_sweep() -> pd.DataFrame:
    """Pairwise Jaccard between task circuits, swept over K and P.

    ``overlap_pairs.csv`` covers only K=10 and P=50, so this reads
    the shared-component lists straight out of the extraction JSON. Chance is
    the overlap of two independent uniform draws of the same sizes, which is
    what makes neuron and component values comparable.
    """
    cached = OUT / "overlap_sweep.csv"
    if cached.exists():
        return pd.read_csv(cached)

    pools = json.loads((ANA / "component_pools.json").read_text())
    circuits: dict[tuple, dict] = {}
    for path in sorted(RAW.rglob("metrics.json")):
        data = json.loads(path.read_text())
        model, task = data.get("model_name"), data.get("task")
        if any(x in str(model) for x in EXCLUDED_MODELS) or task in EXCLUDED_TASKS:
            continue
        key = (model, data.get("method", "eap"), infer_granularity(data, path))
        for K, entry in data.get("by_k", {}).items():
            for p, cell in entry.get("thresholds", {}).items():
                circuits.setdefault(key, {}).setdefault((int(K), int(p)), {})[task] = \
                    frozenset(cell.get("shared_components") or [])

    rows = []
    for (model, method, gran), by_kp in circuits.items():
        pool = pools[model][gran]
        for (K, p), sets in by_kp.items():
            for a, b in itertools.combinations(sorted(sets), 2):
                A, B = sets[a], sets[b]
                if not (A or B):
                    continue
                expected_int = len(A) * len(B) / pool
                expected_union = len(A) + len(B) - expected_int
                rows.append({"model": model, "method": method, "granularity": gran,
                             "K": K, "p": p, "task_a": a, "task_b": b,
                             "size_a": len(A), "size_b": len(B),
                             "observed": len(A & B) / len(A | B),
                             "chance": expected_int / expected_union if expected_union else np.nan})
    out = pd.DataFrame(rows)
    out["ratio_to_chance"] = out.observed / out.chance
    out["related"] = [tuple(sorted((r.task_a, r.task_b))) == RELATED for r in out.itertuples()]
    out.to_csv(cached, index=False)
    print(f"  {cached}")
    return out


def fig_overlap_sweep(pairs: pd.DataFrame):
    fig, axes = grid(len(KS), len(MODELS), sharex=True, sharey=True)
    for r, K in enumerate(KS):
        for c, model in enumerate(MODELS):
            ax = axes[r][c]
            sub = pairs[(pairs.K == K) & (pairs.model == model)]
            for key in CONFIG_ORDER:
                s = sub[(sub.method == key[0]) & (sub.granularity == key[1])]
                unrelated = s[~s.related].groupby("p")["observed"].mean()
                ax.plot(unrelated.index, unrelated.values, lw=1.4, **cfg_line(key))
                arc = s[s.related].groupby("p")["observed"].mean()
                ax.plot(arc.index, arc.values, lw=0, marker="*", ms=7,
                        color=cfg_color(key))
            ax.axvline(P_REF, ls=":", c="grey", lw=.8, zorder=0)
            ax.set_ylim(-.03, 1.03)
            if r == 0:
                ax.set_title(_model_label(model))
            if c == len(MODELS) - 1:
                row_label(ax, f"top-$K$={K}%")
            if r == len(KS) - 1:
                ax.set_xlabel("$P$ (%)")
    fig.supylabel("Circuit overlap (Jaccard)", fontsize=12)
    from matplotlib.lines import Line2D
    config_legend(fig, extra=[
        Line2D([], [], color="0.35", lw=1.6, label="Mean over unrelated pairs"),
        Line2D([], [], color="0.35", lw=0, marker="*", ms=8,
               label="ARC Easy against ARC Challenge")])
    fig.suptitle("Task-pair circuit overlap")
    save(fig, "07_overlap_sweep")


PAPER = (REPO / "paper2" /
         "_NeurIPS_2026_InterpScience_Workshop__How_Much_Do_Circuits_Tell_Us__"
         "Measuring_the_Consistency_and_Specificity_of_Language_Model_Circuits" /
         "figures" / "within_task")
PAPER_REUSE_SWEEP = PAPER / "reuse_sweep"
PAPER_NECESSITY_SWEEP = PAPER / "necessity_sweep"


def sync_paper():
    """Place the appendix sweep figures in dedicated within-task folders."""
    import shutil
    if not PAPER.exists():
        return
    PAPER_REUSE_SWEEP.mkdir(parents=True, exist_ok=True)
    PAPER_NECESSITY_SWEEP.mkdir(parents=True, exist_ok=True)

    for key in CONFIG_ORDER:
        reuse_src = OUT / f"09_kp_heatmap_reuse__{key[0]}_{key[1]}.png"
        reuse_dest = PAPER_REUSE_SWEEP / f"reuse_kp_{key[0]}_{key[1]}.png"
        shutil.copyfile(reuse_src, reuse_dest)
        print(f"  -> {reuse_dest.relative_to(REPO)}")

        necessity_src = OUT / f"09_kp_heatmap_lift__{key[0]}_{key[1]}.png"
        necessity_dest = PAPER_NECESSITY_SWEEP / f"sweep_necessity__{key[0]}_{key[1]}.png"
        shutil.copyfile(necessity_src, necessity_dest)
        print(f"  -> {necessity_dest.relative_to(REPO)}")


def task_statistics() -> pd.DataFrame:
    """Average reuse and lift per task for each model/method/granularity bucket."""
    rows = []
    for key in CONFIG_ORDER:
        d = ext[(ext.method == key[0]) & (ext.granularity == key[1])]
        for task in TASKS:
            sub = d[d.task == task].dropna(subset=["reuse", "lift", "necessity_gap"])
            if sub.empty:
                continue
            rows.append({
                "task": task,
                "method": key[0],
                "granularity": key[1],
                "avg_reuse": sub["reuse"].mean(),
                "avg_lift": sub["lift"].mean(),
                "avg_necessity_gap": sub["necessity_gap"].mean(),
                "avg_circuit_size": sub["circuit_size"].mean(),
                "n_cells": len(sub),
            })
    return pd.DataFrame(rows).sort_values(["task", "method", "granularity"]).reset_index(drop=True)


def main():
    print("writing to", OUT)
    for metric in ["reuse", "specgap"]:
        fig_bars(metric)
    fig_coverage()
    for metric in ["reuse", "size", "specgap"]:
        fig_vs_p_by_config(metric)
    fig_kp_heatmap("reuse")
    fig_kp_heatmap("lift")
    sync_paper()
    for metric in ["reuse", "lift", "specgap"]:
        fig_sweep(metric)
    for metric in ["reuse", "size"]:
        fig_sweep_vs_p(metric)
    fig_own_foreign()
    fig_plane_by_kp()
    fig_avg_check()
    fig_overlap_sweep(overlap_sweep())

    table = variation_table()
    task_table = task_statistics()
    table.to_csv(OUT / "variation_summary.csv", index=False)
    task_table.to_csv(OUT / "task_statistics.csv", index=False)
    print(f"  {OUT / 'variation_summary.csv'}")
    print(f"  {OUT / 'task_statistics.csv'}")
    print()
    print("Variation summary:")
    print(table.to_string(index=False))
    print()
    print("Per-task summary:")
    print(task_table.to_string(index=False))


if __name__ == "__main__":
    main()
