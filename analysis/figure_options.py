"""Alternative renderings of each main-text figure, plus per-model tables.

Every figure in the paper has more than one reasonable form, and the choice
between them is a judgement call about which claim the figure is meant to carry.
This module builds the candidates side by side under one style so they can be
compared, and writes the per-model tables that replace some of them outright.

Output goes to <analysis-dir>/options/ and the markdown tables to
<analysis-dir>/options/tables.md.
"""
from __future__ import annotations

import argparse
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

from analysis.granularity_parity import (CONFIG, CONFIG_ORDER, GRAN_COLORS,
                                         GRAN_SHORT, METHOD_LINESTYLE,
                                         METHOD_MARKER, METHOD_NAMES,
                                         MODEL_COLORS, SECONDARY, _color, _legend,
                                         _model_label, _ramp,
                                         _fill_kw, _label, _line_kw, _ordered,
                                         _task_label)

K, P = 10, 50
FOCUS = "meta-llama/Llama-3.2-3B"
TASK_ORDER = ["arc_easy", "arc_challenge", "mcqa", "ioi", "addition", "boolean"]


def short_model(m: str) -> str:
    return _model_label(m)


def setup():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                         "axes.titlesize": 10, "axes.labelsize": 10,
                         "xtick.labelsize": 9, "ytick.labelsize": 9,
                         "legend.fontsize": 8.5})
    return plt


def style(ax, axis="y"):
    ax.grid(alpha=.25, linewidth=.6, axis=axis)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def panel(ax, text, color=None):
    ax.set_title(text, loc="left", fontweight="bold", fontsize=9.5, pad=5)


class Data:
    def __init__(self, analysis: Path):
        at = lambda d: d[(d.K == K) & (d.p == P)]
        self.within = at(pd.read_csv(analysis / "within_task_summary.csv"))
        self.cross = at(pd.read_csv(analysis / "cross_task_summary.csv"))
        self.extraction = at(pd.read_csv(analysis / "extraction_tidy.csv"))
        self.cross_tidy = at(pd.read_csv(analysis / "cross_task_tidy.csv"))
        self.pairs = pd.read_csv(analysis / "overlap_pairs.csv")
        self.by_model = pd.read_csv(analysis / "overlap_by_model.csv")
        self.models = sorted(self.within.model.unique())
        self.color = dict(zip(self.models, MODEL_COLORS))

    def cell(self, frame, key, model=None, column=None):
        f = frame[(frame.method == key[0]) & (frame.granularity == key[1])]
        if model is not None:
            f = f[f.model == model]
        return f[column] if column else f


# ---------------------------------------------------------------- Figure 1

def fig1_paired_dots(d: Data, out: Path):
    """Model on the y-axis, one dot per granularity.

    The two granularities are unordered categories, so nothing connects them:
    a segment between them would imply an interpolation that does not exist.
    """
    plt = setup()
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2), sharey=True)
    metrics = [(d.within, "reuse", "Reuse@$P$ (%)"),
               (d.cross, "specificity_gap_pp", "Specificity gap (pp)")]
    y = np.arange(len(d.models))
    for ax, (frame, col, label) in zip(axes, metrics):
        for off, method in zip((-.17, .17), ("eap_ig", "relp")):
            for g in ("head_mlp", "neuron"):
                vals = [d.cell(frame, (method, g), m, col).mean() for m in d.models]
                ax.scatter(vals, y + off, s=52, color=_color((method, g)),
                           marker=METHOD_MARKER[method], zorder=3,
                           edgecolor="white", linewidth=.7)
        for yi in y:
            ax.axhline(yi + .5, color="0.9", lw=.7, zorder=0)
        ax.set_yticks(y, [short_model(m) for m in d.models])
        ax.set_xlabel(label)
        if col.startswith("spec"):
            ax.axvline(0, c="0.4", lw=.8, zorder=1)
        style(ax, axis="x")
    axes[0].invert_yaxis()
    handles = [plt.Line2D([], [], color=_color((me, g)), marker=METHOD_MARKER[me],
                          linestyle="none", markersize=7,
                          label=f"{METHOD_NAMES[me]}, {GRAN_SHORT[g]}")
               for g in ("head_mlp", "neuron") for me in ("eap_ig", "relp")]
    _legend(fig, handles=handles, ncol=4, loc="lower center",
               bbox_to_anchor=(.5, -.14))
    save(fig, out / "fig1_opt2_paired_dots.png")


def fig1_plane_arrows(d: Data, out: Path):
    """Each model as an arrow from its heads+MLP point to its neuron point."""
    plt = setup()
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for method in ("eap_ig", "relp"):
        for model in d.models:
            pts = []
            for g in ("head_mlp", "neuron"):
                pts.append((d.cell(d.within, (method, g), model, "reuse").mean(),
                            d.cell(d.cross, (method, g), model,
                                   "specificity_gap_pp").mean()))
            (x0, y0), (x1, y1) = pts
            # annotate() does not grow the data limits, so plot the segment too.
            ax.plot([x0, x1], [y0, y1], color=d.color[model], lw=1.3, alpha=.7,
                    linestyle="-" if method == "eap_ig" else "dashed", zorder=2)
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="-|>", color=d.color[model],
                                        lw=1.3, alpha=.85, shrinkA=0, shrinkB=0,
                                        linestyle="none"))
            ax.scatter([x0], [y0], s=44, color=d.color[model], zorder=3,
                       marker=METHOD_MARKER[method], edgecolor="white", linewidth=.7)
            ax.scatter([x1], [y1], s=64, color=d.color[model], zorder=3,
                       marker=METHOD_MARKER[method], edgecolor="black", linewidth=.9)
    ax.axhline(0, ls="--", c="0.6", lw=1, zorder=0)
    ax.set(xlabel="Reuse@$P$ (%)", ylabel="Specificity gap (pp)")
    style(ax, axis="both")
    handles = [plt.Line2D([], [], color=d.color[m], lw=2, label=short_model(m))
               for m in d.models]
    handles += [plt.Line2D([], [], color="0.4", marker=METHOD_MARKER[me],
                           linestyle="-" if me == "eap_ig" else "dashed",
                           label=METHOD_NAMES[me]) for me in ("eap_ig", "relp")]
    handles += [plt.Line2D([], [], color="0.4", marker="o", linestyle="none",
                           markeredgecolor="black", label="arrow head: neurons")]
    _legend(ax, handles=handles, loc="upper right", fontsize=8)
    save(fig, out / "fig1_opt3_plane_arrows.png")


def fig1_per_model_bars(d: Data, out: Path):
    """Model on the x-axis, one bar per granularity, EAP-IG only."""
    plt = setup()
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.3))
    metrics = [(d.within, "reuse", "Reuse@$P$ (%)"),
               (d.cross, "specificity_gap_pp", "Specificity gap (pp)")]
    x = np.arange(len(d.models))
    for ax, (frame, col, label) in zip(axes, metrics):
        for i, g in enumerate(("head_mlp", "neuron")):
            vals = [d.cell(frame, ("eap_ig", g), m, col).mean() for m in d.models]
            ax.bar(x + (i - .5) * .38, vals, .36, label=GRAN_SHORT[g],
                   **_fill_kw(("eap_ig", g)))
        ax.set_xticks(x, [short_model(m) for m in d.models], rotation=18, ha="right")
        ax.set_ylabel(label)
        ax.axhline(0, c="0.4", lw=.8)
        style(ax)
    _legend(axes[0], title="EAP-IG only", title_fontsize=8)
    save(fig, out / "fig1_opt4_per_model_bars.png")


# ---------------------------------------------------------------- Figure 2

def fig2_reuse_vs_necessity(d: Data, out: Path):
    """Reuse against necessity, one point per model, task, and configuration."""
    plt = setup()
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for key in CONFIG_ORDER:
        sub = d.cell(d.extraction, key)
        ax.scatter(sub.reuse, sub.necessity_gap, s=34, alpha=.75,
                   color=_color(key), marker=METHOD_MARKER[key[0]],
                   label=_label(key), edgecolor="white", linewidth=.5)
    ax.axhline(0, ls="--", c="0.6", lw=1, zorder=0)
    ax.set(xlabel="Reuse@$P$ (%)", ylabel="Necessity gap (pp)")
    style(ax, axis="both")
    _legend(ax, loc="upper center", ncol=2)
    save(fig, out / "fig2_opt2_reuse_vs_necessity.png")


def fig2_heatmap(d: Data, out: Path):
    """Model by task reuse, one panel per configuration."""
    plt = setup()
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 2.9), squeeze=False)
    for ax, key in zip(axes[0], CONFIG_ORDER):
        sub = d.cell(d.extraction, key)
        grid = (sub.pivot_table(index="model", columns="task", values="reuse")
                .reindex(index=d.models, columns=TASK_ORDER))
        im = ax.imshow(grid.values, cmap=_ramp(), vmin=0, vmax=100,
                       aspect="auto")
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                v = grid.values[r][c]
                if np.isfinite(v):
                    ax.text(c, r, f"{v:.0f}", ha="center", va="center", fontsize=7,
                            color="white" if v > 55 else "0.15")
        ax.set_xticks(range(len(TASK_ORDER)), [_task_label(t) for t in TASK_ORDER],
                      rotation=40, ha="right", fontsize=7.5)
        ax.set_yticks(range(len(d.models)),
                      [short_model(m) for m in d.models] if ax is axes[0][0]
                      else [""] * len(d.models), fontsize=7.5)
        panel(ax, _label(key), _color(key))
    fig.colorbar(im, ax=axes[0].tolist(), label="Reuse@$P$ (%)", fraction=.02)
    save(fig, out / "fig2_opt3_model_task_heatmap.png", tight=False)


def fig2_focus_model(d: Data, out: Path):
    """One model, reuse against K per task, both granularities."""
    plt = setup()
    full = pd.read_csv(out.parent / "extraction_tidy.csv")
    sub = full[(full.p == P) & (full.model == FOCUS)]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), sharey=True)
    for ax, method in zip(axes, ("eap_ig", "relp")):
        for g in ("head_mlp", "neuron"):
            for task in TASK_ORDER:
                s = sub[(sub.method == method) & (sub.granularity == g)
                        & (sub.task == task)].sort_values("K")
                ax.plot(s.K, s.reuse, color=GRAN_COLORS[g], alpha=.55, lw=1.2,
                        marker="o", markersize=3)
            m = (sub[(sub.method == method) & (sub.granularity == g)]
                 .groupby("K").reuse.mean())
            ax.plot(m.index, m.values, color=GRAN_COLORS[g], lw=2.6,
                    label=GRAN_SHORT[g], zorder=4)
        ax.set_xscale("log")
        ax.set_xticks([1, 5, 10, 30], ["1%", "5%", "10%", "30%"])
        ax.minorticks_off()
        ax.set_xlabel("Circuit size $K$ (%)")
        panel(ax, METHOD_NAMES[method])
        style(ax)
    axes[0].set_ylabel("Reuse@$P$ (%)")
    handles = [plt.Line2D([], [], color=GRAN_COLORS[g], lw=2.6, label=GRAN_SHORT[g])
               for g in ("head_mlp", "neuron")]
    handles += [plt.Line2D([], [], color="0.55", lw=1.2, marker="o", markersize=3,
                           label="one task")]
    _legend(axes[0], handles=handles, fontsize=8)
    save(fig, out / "fig2_opt4_focus_model.png")


# ---------------------------------------------------------------- Figure 3

def fig3_dumbbell(d: Data, out: Path):
    """One row per task, an arrow from the other-task drop to the own drop."""
    plt = setup()
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), sharey=True, sharex=True)
    y = np.arange(len(TASK_ORDER))
    for ax, g in zip(axes, ("head_mlp", "neuron")):
        for off, method in zip((-.16, .16), ("eap_ig", "relp")):
            sub = d.cross_tidy[(d.cross_tidy.granularity == g)
                               & (d.cross_tidy.method == method)
                               & (d.cross_tidy.donor == d.cross_tidy.target)]
            own = [sub[sub.target == t].own_drop_pp.mean() for t in TASK_ORDER]
            oth = [sub[sub.target == t].foreign_mean_drop_pp.mean() for t in TASK_ORDER]
            c = _color((method, g))
            for yi, (o, f) in enumerate(zip(own, oth)):
                ax.annotate("", xy=(o, yi + off), xytext=(f, yi + off),
                            arrowprops=dict(arrowstyle="-|>", color=c, lw=1.6))
                ax.scatter([f], [yi + off], s=26, color=SECONDARY, zorder=3)
        ax.set_yticks(y, [_task_label(t) for t in TASK_ORDER])
        ax.invert_yaxis()
        ax.set_xlabel("Accuracy drop (pp)")
        panel(ax, GRAN_SHORT[g], GRAN_COLORS[g])
        style(ax, axis="x")
    handles = [plt.Line2D([], [], color="0.35", lw=1.8, label=METHOD_NAMES[me])
               for me in ("eap_ig", "relp")]
    handles += [plt.Line2D([], [], color=SECONDARY, marker="o", linestyle="none",
                           label="other circuits")]
    _legend(axes[0], handles=handles, fontsize=8, loc="lower right")
    save(fig, out / "fig3_opt2_dumbbell.png")


def fig3_per_model_slopes(d: Data, out: Path):
    """The paired plot split by model instead of pooled, EAP-IG only."""
    plt = setup()
    fig, axes = plt.subplots(1, len(d.models), figsize=(2.05 * len(d.models), 3.2),
                             sharey=True)
    for ax, model in zip(axes, d.models):
        for g in ("head_mlp", "neuron"):
            sub = d.cross_tidy[(d.cross_tidy.model == model)
                               & (d.cross_tidy.granularity == g)
                               & (d.cross_tidy.method == "eap_ig")
                               & (d.cross_tidy.donor == d.cross_tidy.target)]
            for r in sub.itertuples():
                ax.plot([0, 1], [r.foreign_mean_drop_pp, r.own_drop_pp],
                        color=GRAN_COLORS[g], alpha=.35, lw=.9)
            ax.plot([0, 1], [sub.foreign_mean_drop_pp.mean(), sub.own_drop_pp.mean()],
                    color=GRAN_COLORS[g], lw=2.8, marker="o", markersize=5,
                    markeredgecolor="white", zorder=4, label=GRAN_SHORT[g])
        ax.set_xlim(-.25, 1.25)
        ax.set_xticks([0, 1], ["Other", "Own"])
        ax.axhline(0, c="0.4", lw=.8)
        panel(ax, short_model(model))
        style(ax)
    axes[0].set_ylabel("Accuracy drop (pp)")
    handles = [plt.Line2D([], [], color=GRAN_COLORS[g], lw=2.8, marker="o",
                          label=GRAN_SHORT[g]) for g in ("head_mlp", "neuron")]
    handles += [plt.Line2D([], [], color="0.6", lw=.9, label="one task")]
    _legend(axes[0], handles=handles, fontsize=7.5,
                   title="EAP-IG only", title_fontsize=7.5)
    save(fig, out / "fig3_opt3_per_model.png")


def fig3_scatter_diagonal(d: Data, out: Path):
    """Own against other for every model-task pair, against the y=x line.

    A point above the diagonal is a pair whose own circuit matters more than a
    foreign one. No line joins the two conditions, and the diagonal itself is the
    reference the claim is about.
    """
    plt = setup()
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.0), sharex=True, sharey=True)
    for ax, g in zip(axes, ("head_mlp", "neuron")):
        for method in ("eap_ig", "relp"):
            sub = d.cross_tidy[(d.cross_tidy.granularity == g)
                               & (d.cross_tidy.method == method)
                               & (d.cross_tidy.donor == d.cross_tidy.target)]
            ax.scatter(sub.foreign_mean_drop_pp, sub.own_drop_pp, s=30, alpha=.8,
                       color=_color((method, g)), marker=METHOD_MARKER[method],
                       edgecolor="white", linewidth=.5, zorder=3,
                       label=METHOD_NAMES[method])
        lim = [-8, 105]
        ax.plot(lim, lim, color="0.35", lw=1, ls="--", zorder=1)
        ax.set(xlim=lim, ylim=lim, xlabel="Other-circuit drop (pp)")
        ax.set_aspect("equal")
        panel(ax, GRAN_SHORT[g])
        style(ax, axis="both")
    axes[0].set_ylabel("Own-circuit drop (pp)")
    _legend(axes[0], loc="lower right", fontsize=8)
    save(fig, out / "fig3_opt4_scatter_diagonal.png")


def fig3_gap_distribution(d: Data, out: Path):
    """Distribution of the per-pair gap, one curve per granularity."""
    plt = setup()
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2), sharey=True)
    bins = np.linspace(-25, 80, 32)
    for ax, method in zip(axes, ("eap_ig", "relp")):
        for g in ("head_mlp", "neuron"):
            sub = d.cross_tidy[(d.cross_tidy.granularity == g)
                               & (d.cross_tidy.method == method)
                               & (d.cross_tidy.donor == d.cross_tidy.target)]
            vals = sub.specificity_gap_pp.dropna()
            ax.hist(vals, bins=bins, color=_color((method, g)), alpha=.6,
                    label=f"{GRAN_SHORT[g]} (median {vals.median():.1f})")
            ax.axvline(vals.median(), color=_color((method, g)), lw=1.8, ls="--")
        ax.axvline(0, color="0.35", lw=1)
        ax.set_xlabel("Own minus other drop (pp)")
        panel(ax, METHOD_NAMES[method])
        style(ax)
        _legend(ax, fontsize=8, loc="upper right")
    axes[0].set_ylabel("Model-task pairs")
    save(fig, out / "fig3_opt5_gap_distribution.png")


# ---------------------------------------------------------------- Figure 4

def _overlap_grid(pairs, key, model=None):
    sub = pairs[(pairs.method == key[0]) & (pairs.granularity == key[1])]
    if model is not None:
        sub = sub[sub.model == model]
    grid = np.full((len(TASK_ORDER), len(TASK_ORDER)), np.nan)
    for (a, b), v in sub.groupby(["task_a", "task_b"])["observed"].mean().items():
        if a in TASK_ORDER and b in TASK_ORDER:
            i, j = TASK_ORDER.index(a), TASK_ORDER.index(b)
            grid[i][j] = grid[j][i] = v
    return grid


def fig4_focus_model(d: Data, out: Path):
    """One model, one panel per granularity, EAP-IG only."""
    plt = setup()
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6))
    labels = [_task_label(t) for t in TASK_ORDER]
    for ax, g in zip(axes, ("head_mlp", "neuron")):
        grid = _overlap_grid(d.pairs, ("eap_ig", g), FOCUS)
        im = ax.imshow(grid, cmap=_ramp(), vmin=0, vmax=1)
        for r in range(len(TASK_ORDER)):
            for c in range(len(TASK_ORDER)):
                if np.isfinite(grid[r][c]):
                    ax.text(c, r, f"{grid[r][c]:.2f}".lstrip("0"), ha="center",
                            va="center", fontsize=7,
                            color="white" if grid[r][c] > .55 else "0.15")
        ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right", fontsize=7.5)
        ax.set_yticks(range(len(labels)), labels if ax is axes[0] else [""] * len(labels),
                      fontsize=7.5)
        panel(ax, GRAN_SHORT[g], GRAN_COLORS[g])
    fig.colorbar(im, ax=axes.tolist(), label="Jaccard overlap", fraction=.03)
    save(fig, out / "fig4_opt2_focus_model.png", tight=False)


def fig4_split_diagonal(d: Data, out: Path):
    """One matrix, heads+MLP below the diagonal and neurons above it."""
    plt = setup()
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    hm = _overlap_grid(d.pairs, ("eap_ig", "head_mlp"), FOCUS)
    nu = _overlap_grid(d.pairs, ("eap_ig", "neuron"), FOCUS)
    n = len(TASK_ORDER)
    merged = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i > j:
                merged[i][j] = hm[i][j]
            elif i < j:
                merged[i][j] = nu[i][j]
    im = ax.imshow(merged, cmap=_ramp(), vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            if np.isfinite(merged[i][j]):
                ax.text(j, i, f"{merged[i][j]:.2f}".lstrip("0"), ha="center",
                        va="center", fontsize=7.5,
                        color="white" if merged[i][j] > .55 else "0.15")
    labels = [_task_label(t) for t in TASK_ORDER]
    ax.set_xticks(range(n), labels, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(n), labels, fontsize=8)
    ax.plot([-.5, n - .5], [-.5, n - .5], color="white", lw=2.5)
    ax.text(.03, .10, GRAN_SHORT["head_mlp"], transform=ax.transAxes, fontsize=9,
            fontweight="bold")
    ax.text(.97, .95, GRAN_SHORT["neuron"], transform=ax.transAxes, fontsize=9,
            fontweight="bold", ha="right", va="top")
    fig.colorbar(im, ax=ax, label="Jaccard overlap", fraction=.045)
    save(fig, out / "fig4_opt3_split_diagonal.png")


def fig4_ratio_to_chance(d: Data, out: Path):
    """Overlap divided by the pool-matched chance baseline, on a log scale."""
    plt = setup()
    from matplotlib.colors import LogNorm
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.6))
    labels = [_task_label(t) for t in TASK_ORDER]
    for ax, g in zip(axes, ("head_mlp", "neuron")):
        sub = d.pairs[(d.pairs.method == "eap_ig") & (d.pairs.granularity == g)
                      & (d.pairs.model == FOCUS)]
        grid = np.full((len(TASK_ORDER), len(TASK_ORDER)), np.nan)
        for (a, b), v in sub.groupby(["task_a", "task_b"])["ratio_to_chance"].mean().items():
            if a in TASK_ORDER and b in TASK_ORDER:
                i, j = TASK_ORDER.index(a), TASK_ORDER.index(b)
                grid[i][j] = grid[j][i] = max(v, .1)
        im = ax.imshow(grid, cmap=_ramp(), norm=LogNorm(vmin=1, vmax=400))
        for r in range(len(TASK_ORDER)):
            for c in range(len(TASK_ORDER)):
                if np.isfinite(grid[r][c]):
                    ax.text(c, r, f"{grid[r][c]:.0f}", ha="center", va="center",
                            fontsize=7, color="white" if grid[r][c] > 40 else "0.15")
        ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right", fontsize=7.5)
        ax.set_yticks(range(len(labels)), labels if ax is axes[0] else [""] * len(labels),
                      fontsize=7.5)
        panel(ax, GRAN_SHORT[g], GRAN_COLORS[g])
    fig.colorbar(im, ax=axes.tolist(), label="Overlap / chance", fraction=.03)
    save(fig, out / "fig4_opt4_ratio_to_chance.png", tight=False)


# ------------------------------------------------------- K and P sensitivity

def sweep_reuse(d: Data, out: Path, analysis: Path):
    """Reuse, specificity, and lift against both thresholds.

    The operating point is one cell of a two-dimensional sweep, so the figure
    shows what happens away from it. Coverage is drawn under reuse@P because the
    two must be read together: reuse at strict P is computed over fewer and fewer
    non-empty circuits.
    """
    plt = setup()
    ext = pd.read_csv(analysis / "extraction_tidy.csv")
    cross = pd.read_csv(analysis / "cross_task_tidy.csv")
    cross = cross[cross.donor == cross.target]

    fig, axes = plt.subplots(2, 3, figsize=(11.4, 5.6),
                             gridspec_kw={"height_ratios": [2, 1]})
    for ax, (frame, col, xcol, fixed, label, xlabel) in zip(
            axes[0],
            [(ext, "reuse", "K", ("p", P), "Reuse@$P$ (%)", "Circuit size $K$ (%)"),
             (ext, "reuse", "p", ("K", K), "Reuse@$P$ (%)", "Consensus threshold $P$ (%)"),
             (cross, "specificity_gap_pp", "K", ("p", P), "Specificity gap (pp)",
              "Circuit size $K$ (%)")]):
        sub = frame[frame[fixed[0]] == fixed[1]]
        for key in CONFIG_ORDER:
            g = (sub[(sub.method == key[0]) & (sub.granularity == key[1])]
                 .groupby(xcol)[col].mean())
            ax.plot(g.index, g.values, markersize=4.5, label=_label(key),
                    **_line_kw(key))
        ax.set(xlabel=xlabel, ylabel=label)
        if xcol == "K":
            ax.set_xscale("log")
            ax.set_xticks([1, 5, 10, 30], ["1%", "5%", "10%", "30%"])
            ax.minorticks_off()
        ax.axvline(fixed[1] if xcol != fixed[0] else None, lw=0)
        ax.axvline(K if xcol == "K" else P, ls=":", c="0.45", lw=1.1, zorder=0)
        style(ax)

    # Coverage beneath, on the same x-axes, since a reuse value computed over two
    # surviving circuits is not comparable to one computed over thirty.
    for ax, (xcol, fixed, xlabel) in zip(
            axes[1], [("K", ("p", P), "Circuit size $K$ (%)"),
                      ("p", ("K", K), "Consensus threshold $P$ (%)"),
                      (None, None, None)]):
        if xcol is None:
            ax.axis("off")
            continue
        sub = ext[ext[fixed[0]] == fixed[1]]
        for key in CONFIG_ORDER:
            g = (sub[(sub.method == key[0]) & (sub.granularity == key[1])]
                 .groupby(xcol).circuit_size.agg(lambda v: (v > 0).mean()))
            ax.plot(g.index, g.values, markersize=4, **_line_kw(key))
        ax.axhline(.8, ls="--", c="0.35", lw=1)
        ax.axvline(K if xcol == "K" else P, ls=":", c="0.45", lw=1.1, zorder=0)
        ax.set(xlabel=xlabel, ylabel="Coverage", ylim=(-.05, 1.05))
        if xcol == "K":
            ax.set_xscale("log")
            ax.set_xticks([1, 5, 10, 30], ["1%", "5%", "10%", "30%"])
            ax.minorticks_off()
        style(ax)
    handles, labels = axes[0][0].get_legend_handles_labels()
    handles += [plt.Line2D([], [], ls="--", color="0.35", label="80% coverage"),
                plt.Line2D([], [], ls=":", color="0.45", label="operating point")]
    _legend(fig, handles=handles, ncol=6, loc="lower center", bbox_to_anchor=(.5, -.04))
    save(fig, out / "sweep_reuse_specificity.png")


def fig3_gap_only(d: Data, out: Path):
    """The gap itself, which is the quantity the section is about."""
    plt = setup()
    fig, ax = plt.subplots(figsize=(8.4, 3.4))
    x = np.arange(len(d.models))
    width = .8 / len(CONFIG_ORDER)
    for i, key in enumerate(CONFIG_ORDER):
        vals = [d.cell(d.cross, key, m, "specificity_gap_pp").mean() for m in d.models]
        pts = [d.cross_tidy[(d.cross_tidy.model == m)
                            & (d.cross_tidy.method == key[0])
                            & (d.cross_tidy.granularity == key[1])
                            & (d.cross_tidy.donor == d.cross_tidy.target)]
               .specificity_gap_pp.dropna() for m in d.models]
        pos = x + i * width - .4 + width / 2
        ax.bar(pos, vals, width * .92, label=_label(key), **_fill_kw(key))
        for xi, vv in zip(pos, pts):
            ax.scatter(np.full(len(vv), xi), vv, s=7, color="0.2", alpha=.55,
                       linewidth=0, zorder=4)
    ax.axhline(0, c="0.3", lw=.9)
    ax.set_xticks(x, [short_model(m) for m in d.models], rotation=18, ha="right")
    ax.set_ylabel("Own minus other drop (pp)")
    style(ax)
    handles, labels = ax.get_legend_handles_labels()
    _legend(fig, handles=handles, labels=labels, ncol=4, loc="lower center",
            bbox_to_anchor=(.5, -.13))
    save(fig, out / "fig3_opt6_gap_only.png")


# ---------------------------------------------------------------- tables

def tables(d: Data) -> dict[str, str]:
    """The two results that genuinely read better as a grid of numbers.

    Both are indexed by model and wanted at full precision, and neither has any
    shape a chart would reveal. The own-against-other drops are deliberately not
    here: their point is the paired comparison, which a table flattens.
    """
    keys = list(CONFIG_ORDER)
    out = {}

    rows = ["| Model | " + " | ".join(_label(k) for k in keys) + " |",
            "| --- | " + " | ".join("---:" for _ in keys) + " |"]
    for metric, frame, col in [("Reuse@P (%)", d.within, "reuse"),
                               ("Specificity gap (pp)", d.cross, "specificity_gap_pp")]:
        rows.append(f"| *{metric}* |" + " |" * len(keys))
        for m in d.models:
            vals = [f"{d.cell(frame, k, m, col).mean():.1f}" for k in keys]
            rows.append(f"| {short_model(m)} | " + " | ".join(vals) + " |")
    out["A"] = "\n".join(rows)

    rows = ["| Model | heads+MLP Jaccard | heads+MLP / chance | neurons Jaccard | neurons / chance |",
            "| --- | ---: | ---: | ---: | ---: |"]
    for m in d.models:
        cells = []
        for g in ("head_mlp", "neuron"):
            r = d.by_model[(d.by_model.method == "eap_ig")
                           & (d.by_model.granularity == g) & (d.by_model.model == m)]
            cells += ([f"{r.observed_jaccard.iloc[0]:.3f}",
                       f"{r.ratio_to_chance.iloc[0]:.0f}x"] if len(r) else ["", ""])
        rows.append(f"| {short_model(m)} | " + " | ".join(cells) + " |")
    out["B"] = "\n".join(rows)
    return out


# ---------------------------------------------------------------- driver

def save(fig, path: Path, tight=True):
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight")
    print(f"wrote {path.name}")
    import matplotlib.pyplot as plt
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", default="results/granularity_parity_analysis")
    args = ap.parse_args()
    analysis = Path(args.analysis_dir)
    out = analysis / "options"
    out.mkdir(parents=True, exist_ok=True)
    d = Data(analysis)

    fig1_paired_dots(d, out)
    fig1_plane_arrows(d, out)
    fig1_per_model_bars(d, out)
    fig2_reuse_vs_necessity(d, out)
    fig2_heatmap(d, out)
    fig2_focus_model(d, out)
    fig3_dumbbell(d, out)
    fig3_per_model_slopes(d, out)
    fig3_scatter_diagonal(d, out)
    fig3_gap_distribution(d, out)
    fig3_gap_only(d, out)
    sweep_reuse(d, out, analysis)
    fig4_focus_model(d, out)
    fig4_split_diagonal(d, out)
    fig4_ratio_to_chance(d, out)
    t = tables(d)
    for name, body in t.items():
        (out / f"table_{name}.md").write_text(body + "\n")
    print("wrote table_A.md, table_B.md")


if __name__ == "__main__":
    main()
