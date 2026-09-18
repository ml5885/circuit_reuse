"""Candidate figures for the paper2 revision.

Each main-text claim gets more than one candidate rendering, so the choice
between them can be made by looking rather than by imagining. Output goes to
paper2/figures_draft/, one PNG per candidate, named slot_variant_description.

Run: python -m analysis.paper2_figures
"""
from __future__ import annotations

import glob
import json
import os
import tempfile
from pathlib import Path

_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "circuit_reuse_mplconfig"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.granularity_parity import (CONFIG_ORDER, GRAN_COLORS, OTHER_COLOR,
                                         SECONDARY_LIGHT, _color, _fill_kw,
                                         _label, _legend, _model_label, _panel,
                                         _ramp, _task_label, bootstrap_mean)

REPO = Path(__file__).resolve().parent.parent
ANA = REPO / "results" / "granularity_parity_analysis"
OUT = REPO / "paper2" / "figures_draft"
OUT.mkdir(parents=True, exist_ok=True)

K, P = 10, 50
TASK_ORDER = ["addition", "boolean", "ioi", "mcqa", "arc_easy", "arc_challenge"]
MODEL_ORDER = ["google/gemma-2-2b", "google/gemma-2-2b-it",
               "meta-llama/Llama-3.2-3B", "meta-llama/Llama-3.2-3B-Instruct",
               "qwen3-4b"]
CONTROL = "0.62"

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                     "axes.titlesize": 11, "axes.labelsize": 10,
                     "xtick.labelsize": 9, "ytick.labelsize": 9,
                     "legend.fontsize": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": .25, "grid.linewidth": .6,
                     "axes.axisbelow": True})


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}.png")


def style(ax):
    ax.grid(axis="y", alpha=.25, linewidth=.6)
    ax.set_axisbelow(True)


# --- data ---------------------------------------------------------------

extraction = pd.read_csv(ANA / "extraction_tidy.csv")
cross_tidy = pd.read_csv(ANA / "cross_task_tidy.csv")
cross_sum = pd.read_csv(ANA / "cross_task_summary.csv")
overlap = pd.read_csv(ANA / "overlap_pairs.csv")
boot = pd.read_csv(ANA / "bootstrap_summary.csv")
selective = pd.read_csv(ANA / "selective_ablation_tidy.csv")
composition = pd.read_csv(ANA / "circuit_composition.csv")

ext = extraction[(extraction.K == K) & (extraction.p == P)]
crs = cross_sum[(cross_sum.K == K) & (cross_sum.p == P)]
crt = cross_tidy[(cross_tidy.K == K) & (cross_tidy.p == P)]


def bmean(method, gran, metric):
    r = boot[(boot.method == method) & (boot.granularity == gran) & (boot.metric == metric)]
    return r.iloc[0][["mean", "lo", "hi"]].astype(float).to_numpy()


def config_offsets(n_configs, width=.8):
    w = width / n_configs
    return [(i - (n_configs - 1) / 2) * w for i in range(n_configs)], w


# --- slot 1: the reversal (teaser) --------------------------------------

def fig1_a_bars_per_model():
    """Two panels, four configs per model. The form already in the analysis dir."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    offs, w = config_offsets(len(CONFIG_ORDER))
    x = np.arange(len(MODEL_ORDER))
    for ax, (col, ylab) in zip(axes, [("reuse", "Reuse@$P$ (%)"),
                                      ("specificity_gap_pp", "Specificity gap (pp)")]):
        src = ext if col == "reuse" else crs
        for off, key in zip(offs, CONFIG_ORDER):
            sub = src[(src.method == key[0]) & (src.granularity == key[1])]
            vals = [sub[sub.model == m][col].mean() for m in MODEL_ORDER]
            ax.bar(x + off, vals, w * .92, label=_label(key), **_fill_kw(key))
        ax.set_xticks(x)
        ax.set_xticklabels([_model_label(m) for m in MODEL_ORDER], rotation=20, ha="right")
        ax.set_ylabel(ylab)
        ax.axhline(0, color="0.25", lw=.8)
        style(ax)
    _legend(fig, loc="lower center", ncol=4, bbox_to_anchor=(.5, -.12))
    save(fig, "fig1_a_bars_per_model")


def fig1_b_plane():
    """Reuse against specificity, one point per model and config, with a
    per-model arrow from the component to the neuron granularity."""
    fig, ax = plt.subplots(figsize=(6.4, 5))
    pts = {}
    for key in CONFIG_ORDER:
        e = ext[(ext.method == key[0]) & (ext.granularity == key[1])]
        c = crs[(crs.method == key[0]) & (crs.granularity == key[1])]
        for m in MODEL_ORDER:
            pts[(key, m)] = (e[e.model == m].reuse.mean(),
                             c[c.model == m].specificity_gap_pp.mean())
    for method in ["eap_ig", "relp"]:
        for m in MODEL_ORDER:
            a = pts[((method, "head_mlp"), m)]
            b = pts[((method, "neuron"), m)]
            ax.annotate("", xy=b, xytext=a, arrowprops=dict(
                arrowstyle="-|>", color="0.7", lw=.9, shrinkA=5, shrinkB=5))
    for key in CONFIG_ORDER:
        xs = [pts[(key, m)][0] for m in MODEL_ORDER]
        ys = [pts[(key, m)][1] for m in MODEL_ORDER]
        ax.scatter(xs, ys, s=70, color=_color(key), edgecolor="0.25", lw=.7,
                   marker="o" if key[0] == "eap_ig" else "^", label=_label(key), zorder=3)
    ax.axhline(0, color="0.25", lw=.8)
    ax.set_xlabel("Reuse@$P$ (%) — consistency")
    ax.set_ylabel("Specificity gap (pp) — specificity")
    ax.set_ylim(-6, ax.get_ylim()[1] + 3)
    ax.text(.24, .97, "specific,\nnot consistent", transform=ax.transAxes,
            ha="left", va="top", fontsize=9, color="0.35")
    ax.text(.63, .02, "consistent, not specific", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=9, color="0.35")
    _legend(ax, loc="upper right")
    save(fig, "fig1_b_plane")


def fig1_c_forest():
    """The smallest possible teaser: four rows, two metrics, bootstrap intervals."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.1), sharey=True)
    ys = np.arange(len(CONFIG_ORDER))[::-1]
    for ax, (metric, xlab) in zip(axes, [("reuse", "Reuse@$P$ (%)"),
                                         ("specificity_gap_pp", "Specificity gap (pp)")]):
        for y, key in zip(ys, CONFIG_ORDER):
            mu, lo, hi = bmean(key[0], key[1], metric)
            ax.plot([lo, hi], [y, y], color=_color(key), lw=3, solid_capstyle="round")
            ax.plot([mu], [y], "o", color=_color(key), ms=9,
                    markeredgecolor="0.25", markeredgewidth=.7, zorder=3)
            ax.text(mu, y + .22, f"{mu:.1f}", ha="center", va="bottom", fontsize=9)
        ax.axvline(0, color="0.25", lw=.8)
        ax.set_xlabel(xlab)
        ax.grid(axis="x", alpha=.25, linewidth=.6)
        ax.set_ylim(-.6, len(CONFIG_ORDER) - .4)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([_label(k) for k in CONFIG_ORDER])
    save(fig, "fig1_c_forest")


def fig1_d_dumbbell():
    """One row per model; a line joining the component and neuron value."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    ys = np.arange(len(MODEL_ORDER))[::-1]
    for ax, (col, xlab) in zip(axes, [("reuse", "Reuse@$P$ (%)"),
                                      ("specificity_gap_pp", "Specificity gap (pp)")]):
        src = ext if col == "reuse" else crs
        for method, dy in [("eap_ig", .16), ("relp", -.16)]:
            for y, m in zip(ys, MODEL_ORDER):
                v = {}
                for g in ["head_mlp", "neuron"]:
                    s = src[(src.method == method) & (src.granularity == g) & (src.model == m)]
                    v[g] = s[col].mean()
                ax.plot([v["head_mlp"], v["neuron"]], [y + dy] * 2, color="0.7", lw=1.2, zorder=1)
                for g in ["head_mlp", "neuron"]:
                    ax.plot(v[g], y + dy, "o" if method == "eap_ig" else "^",
                            color=_color((method, g)), ms=8, markeredgecolor="0.25",
                            markeredgewidth=.6, zorder=3)
        ax.axvline(0, color="0.25", lw=.8)
        ax.set_xlabel(xlab)
        ax.grid(axis="x", alpha=.25, linewidth=.6)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([_model_label(m) for m in MODEL_ORDER])
    handles = [plt.Line2D([], [], marker="o" if k[0] == "eap_ig" else "^", ls="",
                          color=_color(k), ms=8, markeredgecolor="0.25",
                          markeredgewidth=.6, label=_label(k)) for k in CONFIG_ORDER]
    _legend(fig, handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(.5, -.14))
    save(fig, "fig1_d_dumbbell")


# --- slot 2: within-task consistency ------------------------------------

def fig2_a_reuse_vs_p():
    """Reuse against the consensus threshold, with the coverage that backs it."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    sub = extraction[extraction.K == K]
    for key in CONFIG_ORDER:
        s = sub[(sub.method == key[0]) & (sub.granularity == key[1])]
        g = s.groupby("p")
        ps = sorted(s.p.unique())
        mu = [g.get_group(p).reuse.mean() for p in ps]
        cov = [(g.get_group(p).circuit_size > 0).mean() * 100 for p in ps]
        kw = dict(color=_color(key), marker="o" if key[0] == "eap_ig" else "^",
                  ls="-" if key[0] == "eap_ig" else (0, (4, 1.6)), ms=4.5, lw=1.6)
        axes[0].plot(ps, mu, label=_label(key), **kw)
        axes[1].plot(ps, cov, **kw)
    axes[0].set_ylabel("Reuse@$P$ (%)")
    axes[1].set_ylabel("Non-empty shared circuits (%)")
    axes[1].axhline(80, color=CONTROL, ls=":", lw=1.2)
    axes[1].text(99, 82, "80% criterion", ha="right", fontsize=8, color="0.35")
    for ax, lab in zip(axes, ["(a) consistency", "(b) coverage"]):
        ax.axvline(P, color="0.75", lw=1, zorder=0)
        ax.set_xlabel("Consensus threshold $P$ (%)")
        _panel(ax, lab)
        style(ax)
    _legend(axes[0], loc="upper right")
    save(fig, "fig2_a_reuse_and_coverage_vs_p")


def fig2_b_reuse_by_task():
    """Reuse by task; shows the reversal is not carried by one task."""
    fig, ax = plt.subplots(figsize=(9, 3.4))
    offs, w = config_offsets(len(CONFIG_ORDER))
    x = np.arange(len(TASK_ORDER))
    for off, key in zip(offs, CONFIG_ORDER):
        s = ext[(ext.method == key[0]) & (ext.granularity == key[1])]
        vals = [s[s.task == t].reuse.mean() for t in TASK_ORDER]
        ax.bar(x + off, vals, w * .92, label=_label(key), **_fill_kw(key))
    ax.set_xticks(x)
    ax.set_xticklabels([_task_label(t) for t in TASK_ORDER])
    ax.set_ylabel("Reuse@$P$ (%)")
    style(ax)
    _legend(ax, loc="upper right", ncol=2)
    save(fig, "fig2_b_reuse_by_task")


def fig2_c_reuse_strip():
    """Every model-task cell as a dot, so the spread is visible."""
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    rng = np.random.default_rng(0)
    for i, key in enumerate(CONFIG_ORDER):
        s = ext[(ext.method == key[0]) & (ext.granularity == key[1])]
        ax.scatter(i + rng.uniform(-.16, .16, len(s)), s.reuse, s=26,
                   color=_color(key), edgecolor="0.3", lw=.4, alpha=.85, zorder=3)
        mu, lo, hi = bmean(key[0], key[1], "reuse")
        ax.plot([i - .3, i + .3], [mu, mu], color="0.15", lw=2, zorder=4)
        ax.plot([i, i], [lo, hi], color="0.15", lw=1, zorder=4)
    ax.set_xticks(range(len(CONFIG_ORDER)))
    ax.set_xticklabels([_label(k).replace(", ", "\n") for k in CONFIG_ORDER])
    ax.set_ylabel("Reuse@$P$ (%)")
    style(ax)
    save(fig, "fig2_c_reuse_strip")


# --- slot 3: cross-task specificity --------------------------------------

def fig3_a_own_other_2x2():
    """Own against other by task, one panel per config."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=True, sharey=True)
    for ax, key in zip(axes.ravel(), CONFIG_ORDER):
        s = crt[(crt.method == key[0]) & (crt.granularity == key[1])]
        s = s[s.donor == s.target]
        x = np.arange(len(TASK_ORDER))
        own = [s[s.target == t].own_drop_pp.mean() for t in TASK_ORDER]
        oth = [s[s.target == t].foreign_mean_drop_pp.mean() for t in TASK_ORDER]
        ax.bar(x - .2, own, .38, color=_color(key), label="Own circuit")
        ax.bar(x + .2, oth, .38, color=OTHER_COLOR, label="Other circuits (mean)")
        _panel(ax, _label(key))
        ax.axhline(0, color="0.25", lw=.8)
        style(ax)
    for ax in axes[1]:
        ax.set_xticks(np.arange(len(TASK_ORDER)))
        ax.set_xticklabels([_task_label(t) for t in TASK_ORDER], rotation=20, ha="right")
    for ax in axes[:, 0]:
        ax.set_ylabel("Accuracy drop (pp)")
    _legend(axes[0, 0], loc="upper right")
    save(fig, "fig3_a_own_other_2x2")


def fig3_b_gap_by_model():
    """One panel: the specificity gap alone, four bars per model."""
    fig, ax = plt.subplots(figsize=(8, 3.4))
    offs, w = config_offsets(len(CONFIG_ORDER))
    x = np.arange(len(MODEL_ORDER))
    for off, key in zip(offs, CONFIG_ORDER):
        s = crs[(crs.method == key[0]) & (crs.granularity == key[1])]
        vals = [s[s.model == m].specificity_gap_pp.mean() for m in MODEL_ORDER]
        ax.bar(x + off, vals, w * .92, label=_label(key), **_fill_kw(key))
    ax.axhline(0, color="0.25", lw=.9)
    ax.set_xticks(x)
    ax.set_xticklabels([_model_label(m) for m in MODEL_ORDER], rotation=15, ha="right")
    ax.set_ylabel("Specificity gap (pp)")
    ax.set_ylim(top=ax.get_ylim()[1] * 1.45)
    style(ax)
    _legend(ax, loc="upper left", ncol=2)
    save(fig, "fig3_b_gap_by_model")


def fig3_c_paired_own_other():
    """Own and other as a connected pair per task, four panels."""
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), sharey=True)
    for ax, key in zip(axes, CONFIG_ORDER):
        s = crt[(crt.method == key[0]) & (crt.granularity == key[1])]
        s = s[s.donor == s.target]
        ys = np.arange(len(TASK_ORDER))[::-1]
        for y, t in zip(ys, TASK_ORDER):
            own = s[s.target == t].own_drop_pp.mean()
            oth = s[s.target == t].foreign_mean_drop_pp.mean()
            ax.plot([oth, own], [y, y], color="0.75", lw=1.4, zorder=1)
            ax.plot(oth, y, "o", color=OTHER_COLOR, ms=8, markeredgecolor="0.25",
                    markeredgewidth=.6, zorder=3)
            ax.plot(own, y, "o", color=_color(key), ms=8, markeredgecolor="0.25",
                    markeredgewidth=.6, zorder=3)
        _panel(ax, _label(key).replace(", ", "\n"))
        ax.set_xlabel("Accuracy drop (pp)")
        ax.grid(axis="x", alpha=.25, linewidth=.6)
    axes[0].set_yticks(np.arange(len(TASK_ORDER))[::-1])
    axes[0].set_yticklabels([_task_label(t) for t in TASK_ORDER])
    handles = [plt.Line2D([], [], marker="o", ls="", color=GRAN_COLORS["head_mlp"],
                          ms=8, markeredgecolor="0.25", label="Own circuit"),
               plt.Line2D([], [], marker="o", ls="", color=OTHER_COLOR, ms=8,
                          markeredgecolor="0.25", label="Other circuits (mean)")]
    _legend(fig, handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(.5, -.16))
    save(fig, "fig3_c_paired_own_other")


def fig3_d_gap_vs_p():
    """The gap across the whole threshold sweep, not only the operating point."""
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    sub = cross_sum[cross_sum.K == K]
    for key in CONFIG_ORDER:
        s = sub[(sub.method == key[0]) & (sub.granularity == key[1])]
        g = s.groupby("p").specificity_gap_pp.mean()
        ax.plot(g.index, g.values, label=_label(key), color=_color(key),
                marker="o" if key[0] == "eap_ig" else "^", ms=4.5, lw=1.6,
                ls="-" if key[0] == "eap_ig" else (0, (4, 1.6)))
    ax.axhline(0, color="0.25", lw=.9)
    ax.axvline(P, color="0.75", lw=1, zorder=0)
    ax.set_xlabel("Consensus threshold $P$ (%)")
    ax.set_ylabel("Specificity gap (pp)")
    style(ax)
    _legend(ax, loc="upper left")
    save(fig, "fig3_d_gap_vs_p")


# --- slot 4: overlap ------------------------------------------------------

def fig4_a_heatmaps():
    """Pairwise overlap, one matrix per config."""
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.6))
    for ax, key in zip(axes, CONFIG_ORDER):
        s = overlap[(overlap.method == key[0]) & (overlap.granularity == key[1])]
        mat = np.full((len(TASK_ORDER), len(TASK_ORDER)), np.nan)
        for _, r in s.iterrows():
            i, j = TASK_ORDER.index(r.task_a), TASK_ORDER.index(r.task_b)
            mat[i, j] = mat[j, i] = r.observed
        im = ax.imshow(np.ma.masked_invalid(mat), cmap=_ramp(), vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(mat):
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}".lstrip("0"), ha="center", va="center",
                        fontsize=7, color="white" if v > .55 else "0.15")
        ax.set_xticks(range(len(TASK_ORDER)))
        ax.set_xticklabels([_task_label(t) for t in TASK_ORDER], rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(TASK_ORDER)))
        ax.set_yticklabels([_task_label(t) for t in TASK_ORDER] if ax is axes[0] else [], fontsize=8)
        ax.grid(False)
        _panel(ax, _label(key))
    fig.colorbar(im, ax=axes, shrink=.8, label="Jaccard overlap")
    save(fig, "fig4_a_heatmaps")


def _relatedness(row):
    return "related" if {row.task_a, row.task_b} == {"arc_easy", "arc_challenge"} else "unrelated"


def fig4_b_related_vs_unrelated():
    """The claim reduced to two bars per config."""
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ov = overlap.assign(kind=overlap.apply(_relatedness, axis=1))
    x = np.arange(len(CONFIG_ORDER))
    for off, kind, alpha, lab in [(-.19, "related", 1.0, "ARC Easy / ARC Challenge"),
                                  (.19, "unrelated", .45, "All other task pairs")]:
        vals = [ov[(ov.method == k[0]) & (ov.granularity == k[1]) & (ov.kind == kind)].observed.mean()
                for k in CONFIG_ORDER]
        ax.bar(x + off, vals, .36, label=lab,
               color=[_color(k) for k in CONFIG_ORDER], alpha=alpha,
               edgecolor="0.25", lw=.6)
        for xi, v in zip(x + off, vals):
            ax.text(xi, v + .015, f"{v:.2f}".lstrip("0"), ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([_label(k).replace(", ", "\n") for k in CONFIG_ORDER])
    ax.set_ylabel("Jaccard overlap")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor="0.35", alpha=a, edgecolor="0.25", label=l)
               for a, l in [(1.0, "ARC Easy / ARC Challenge"), (.45, "All other task pairs")]]
    _legend(ax, handles=handles, loc="upper right")
    style(ax)
    save(fig, "fig4_b_related_vs_unrelated")


def fig4_c_pair_strip():
    """All 15 task pairs as dots, the related pair marked."""
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    rng = np.random.default_rng(1)
    ov = overlap.assign(kind=overlap.apply(_relatedness, axis=1))
    for i, key in enumerate(CONFIG_ORDER):
        s = ov[(ov.method == key[0]) & (ov.granularity == key[1])]
        pair = s.groupby(["task_a", "task_b", "kind"])["observed"].mean().reset_index()
        un = pair[pair.kind == "unrelated"]
        re = pair[pair.kind == "related"]
        ax.scatter(i + rng.uniform(-.16, .16, len(un)), un.observed, s=30,
                   color=_color(key), edgecolor="0.3", lw=.4, alpha=.8, zorder=3)
        ax.scatter([i] * len(re), re.observed, s=110, marker="*",
                   color=_color(key), edgecolor="0.15", lw=.9, zorder=4)
    ax.set_xticks(range(len(CONFIG_ORDER)))
    ax.set_xticklabels([_label(k).replace(", ", "\n") for k in CONFIG_ORDER])
    ax.set_ylabel("Jaccard overlap")
    handles = [plt.Line2D([], [], marker="o", ls="", color="0.45", ms=6, label="Unrelated task pair"),
               plt.Line2D([], [], marker="*", ls="", color="0.45", ms=12, label="ARC Easy / ARC Challenge")]
    _legend(ax, handles=handles, loc="upper right")
    style(ax)
    save(fig, "fig4_c_pair_strip")


# --- slot 5: necessity ----------------------------------------------------

def fig5_a_lift_forest():
    """Excess damage over a capacity-matched random set, by config."""
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    ys = np.arange(len(CONFIG_ORDER))[::-1]
    for y, key in zip(ys, CONFIG_ORDER):
        mu, lo, hi = bmean(key[0], key[1], "lift")
        ax.plot([lo, hi], [y, y], color=_color(key), lw=3, solid_capstyle="round")
        ax.plot(mu, y, "o", color=_color(key), ms=9, markeredgecolor="0.25",
                markeredgewidth=.7, zorder=3)
        ax.text(hi + 1.5, y, f"{mu:.0f}", va="center", fontsize=9)
    ax.axvline(0, color="0.25", lw=.9)
    ax.set_yticks(ys)
    ax.set_yticklabels([_label(k) for k in CONFIG_ORDER])
    ax.set_xlabel("Excess accuracy drop over capacity-matched control (%)")
    ax.grid(axis="x", alpha=.25, linewidth=.6)
    ax.set_ylim(-.6, len(CONFIG_ORDER) - .4)
    save(fig, "fig5_a_lift_forest")


# --- slot 6: the selective-ablation correction ----------------------------

def fig6_a_selectivity_vs_control():
    """The task-only remainder against the random set it is size-matched to."""
    sel = selective.assign(sel=selective.target - selective.non_target)
    wide = sel.pivot_table(index=["model", "task_a", "task_b"], columns="condition", values="sel")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.6),
                             gridspec_kw=dict(wspace=.38))

    ax = axes[0]
    conds = ["shared_core", "residual_a", "residual_b", "random_control"]
    labels = ["Shared core", "$A$-only", "$B$-only", "Random control"]
    colors = [GRAN_COLORS["head_mlp"], GRAN_COLORS["neuron"], SECONDARY_LIGHT, CONTROL]
    x = np.arange(len(conds))
    tgt = [sel[sel.condition == c].target.mean() for c in conds]
    oth = [sel[sel.condition == c].non_target.mean() for c in conds]
    ax.bar(x - .2, tgt, .38, color=colors, edgecolor="0.25", lw=.6)
    ax.bar(x + .2, oth, .38, color=colors, edgecolor="0.25", lw=.6, hatch="////", alpha=.55)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Relative accuracy drop (%)")
    _panel(ax, "(a) drop by partition")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor="0.5", edgecolor="0.25", label="Target task"),
               plt.Rectangle((0, 0), 1, 1, facecolor="0.5", edgecolor="0.25", hatch="////",
                             alpha=.55, label="Other tasks (mean)")]
    _legend(ax, handles=handles, loc="upper right")
    style(ax)

    ax = axes[1]
    models = [m for m in MODEL_ORDER] + ["qwen3-8b"]
    ys = np.arange(len(models))[::-1]
    for y, m in zip(ys, models):
        w = wide.loc[m]
        a, c = w.residual_a.mean(), w.random_control.mean()
        ax.plot([c, a], [y, y], color="0.75", lw=1.4, zorder=1)
        ax.plot(c, y, "o", color=CONTROL, ms=8, markeredgecolor="0.25", markeredgewidth=.6, zorder=3)
        ax.plot(a, y, "o", color=GRAN_COLORS["neuron"], ms=8, markeredgecolor="0.25",
                markeredgewidth=.6, zorder=3)
    d = (wide.residual_a - wide.random_control).groupby(level=0).mean()
    ax.axvline(0, color="0.25", lw=.9)
    ax.set_yticks(ys)
    ax.set_yticklabels([_model_label(m) for m in models])
    ax.set_ylim(-1.1, len(models) - .4)
    ax.set_xlabel("Selectivity: target drop $-$ other-task drop (pp)")
    _panel(ax, f"(b) $A$-only is behind its own control in {int((d < 0).sum())}/{len(d)} models")
    handles = [plt.Line2D([], [], marker="o", ls="", color=GRAN_COLORS["neuron"], ms=8,
                          markeredgecolor="0.25", label="$A$-only set"),
               plt.Line2D([], [], marker="o", ls="", color=CONTROL, ms=8,
                          markeredgecolor="0.25", label="Size-matched random control")]
    _legend(ax, handles=handles, loc="upper center", bbox_to_anchor=(.5, -.28), ncol=2)
    ax.grid(axis="x", alpha=.25, linewidth=.6)
    save(fig, "fig6_a_selectivity_vs_control")


# --- slot 7: composition --------------------------------------------------

def all_task_intersections():
    """MLP share of the per-task shared circuit against the all-task intersection."""
    rows = []
    root = REPO / "results" / "granularity_parity" / "granularity_parity_extraction"
    for method in ["eap_ig", "relp"]:
        base = root / f"granularity_parity_{method}_head_mlp"
        for model in MODEL_ORDER:
            tag = model.replace("/", "_")
            sets = {}
            for d in sorted(glob.glob(str(base / f"{tag}__*"))):
                task = Path(d).name.split("__")[2]
                if task not in TASK_ORDER:
                    continue
                th = json.loads((Path(d) / "metrics.json").read_text())["by_k"][str(K)]["thresholds"]
                sets[task] = set(th[str(P)]["shared_components"])
            if len(sets) != len(TASK_ORDER):
                continue
            inter = set.intersection(*sets.values())
            per_task = np.mean([sum("mlp" in c for c in s) / len(s) for s in sets.values() if s])
            rows.append({"method": method, "model": model,
                         "per_task": per_task * 100,
                         "intersection": (sum("mlp" in c for c in inter) / len(inter) * 100)
                         if inter else np.nan,
                         "inter_size": len(inter)})
    return pd.DataFrame(rows)


def fig7_a_mlp_share(inter: pd.DataFrame):
    """Heads are shared within a task; MLP blocks are shared across tasks."""
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    x = np.arange(len(MODEL_ORDER))
    for off, method, hatch in [(-.2, "eap_ig", None), (.2, "relp", "////")]:
        s = inter[inter.method == method].set_index("model")
        pt = [s.per_task.get(m, np.nan) for m in MODEL_ORDER]
        it = [s.intersection.get(m, np.nan) for m in MODEL_ORDER]
        ax.bar(x + off, pt, .36, color=GRAN_COLORS["head_mlp"], alpha=.45,
               edgecolor="0.25", lw=.6, hatch=hatch)
        ax.plot(x + off, it, "D", color=GRAN_COLORS["neuron"], ms=8,
                markeredgecolor="0.25", markeredgewidth=.7, zorder=3)
        for xi, v, bar in zip(x + off, it, pt):
            # An empty intersection has no MLP share to plot, so it is named.
            ok = not np.isnan(v)
            ax.text(xi, (v if ok else bar) + 3, f"{v:.0f}" if ok else "empty",
                    ha="center", fontsize=8, color="0.15" if ok else "0.45",
                    style="normal" if ok else "italic")
    ax.set_xticks(x)
    ax.set_xticklabels([_model_label(m) for m in MODEL_ORDER], rotation=15, ha="right")
    ax.set_ylabel("MLP share (%)")
    ax.set_ylim(0, 128)
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=GRAN_COLORS["head_mlp"], alpha=.45,
                      edgecolor="0.25", label="Per-task shared circuit (EAP-IG)"),
        plt.Rectangle((0, 0), 1, 1, facecolor=GRAN_COLORS["head_mlp"], alpha=.45,
                      edgecolor="0.25", hatch="////", label="Per-task shared circuit (RelP)"),
        plt.Line2D([], [], marker="D", ls="", color=GRAN_COLORS["neuron"], ms=8,
                   markeredgecolor="0.25", label="All-task intersection")]
    _legend(ax, handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(.5, 1.02))
    style(ax)
    save(fig, "fig7_a_mlp_share")


# --- slot 8: agreement between the two attribution methods -----------------

AGREE_METRICS = [
    ("reuse", "Reuse@$P$", "which\ncomponents\nrecur"),
    ("overlap", "Task-pair overlap", "how much\ntasks\nshare"),
    ("lift", "Causal importance", "how much\nthe set\nmatters"),
    ("gap", "Specificity gap", "own minus\nforeign\ndrop"),
    ("own_drop_pp", "Own-circuit drop", "how much\nablation\nhurts"),
]


def agreement_cells(metric: str, gran: str) -> pd.DataFrame:
    """One row per unit of comparison, with a column per method."""
    if metric == "overlap":
        s = overlap[overlap.granularity == gran]
        idx, col = ["model", "task_a", "task_b"], "observed"
    elif metric in ("gap", "own_drop_pp"):
        s = crt[(crt.granularity == gran) & (crt.donor == crt.target)].copy()
        s["gap"] = s.own_drop_pp - s.foreign_mean_drop_pp
        idx, col = ["model", "target"], metric
    else:
        s = ext[ext.granularity == gran]
        idx, col = ["model", "task"], metric
    return s.pivot_table(index=idx, columns="method", values=col).dropna()


def fig8_a_agreement_bars():
    """Rank agreement between EAP-IG and RelP, by metric and granularity."""
    from scipy.stats import spearmanr
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    x = np.arange(len(AGREE_METRICS))
    for off, gran in [(-.2, "head_mlp"), (.2, "neuron")]:
        rhos, sig = [], []
        for metric, _, _ in AGREE_METRICS:
            w = agreement_cells(metric, gran)
            r = spearmanr(w.eap_ig, w.relp)
            rhos.append(r.statistic)
            sig.append(r.pvalue < .05)
        ax.bar(x + off, rhos, .36, color=GRAN_COLORS[gran], edgecolor="0.25", lw=.6,
               label="Component-level" if gran == "head_mlp" else "Neuron-level")
        for xi, v, s in zip(x + off, rhos, sig):
            ax.text(xi, v + .03, f"{v:.2f}".lstrip("0") + ("" if s else "*"),
                    ha="center", fontsize=8, color="0.15" if s else "0.5")
    ax.set_xticks(x)
    ax.set_xticklabels([lab.replace(" ", "\n", 1) for _, lab, _ in AGREE_METRICS], fontsize=9)
    for xi, (_, _, why) in zip(x, AGREE_METRICS):
        ax.text(xi, -.12, why, ha="center", va="top", fontsize=7.5, color="0.45",
                transform=ax.get_xaxis_transform())
    ax.set_ylabel("Rank agreement between\nEAP-IG and RelP ($\\rho$)")
    ax.set_ylim(0, 1.12)
    ax.axhline(0, color="0.25", lw=.8)
    ax.text(.012, .965, "* not significant at $p<0.05$", transform=ax.transAxes,
            fontsize=8, color="0.45", va="top")
    style(ax)
    _legend(ax, loc="upper right", ncol=2)
    save(fig, "fig8_a_method_agreement_bars")


def fig8_b_agreement_scatter():
    """Per-cell values under one method against the other, with the identity line."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9))
    for ax, (metric, lab, _) in zip(axes, AGREE_METRICS[:3]):
        for gran in ["head_mlp", "neuron"]:
            w = agreement_cells(metric, gran)
            ax.scatter(w.eap_ig, w.relp, s=30, color=GRAN_COLORS[gran],
                       edgecolor="0.3", lw=.4, alpha=.8, zorder=3,
                       label="Component-level" if gran == "head_mlp" else "Neuron-level")
        lim = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lim, lim, color="0.55", ls=(0, (4, 2)), lw=1, zorder=1)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel(f"{lab}, EAP-IG")
        ax.set_ylabel(f"{lab}, RelP")
        _panel(ax, lab)
        style(ax)
    axes[0].text(.96, .06, "dashed line:\nmethods agree", transform=axes[0].transAxes,
                 ha="right", va="bottom", fontsize=8, color="0.45")
    _legend(fig, handles=axes[0].get_legend_handles_labels()[0], loc="lower center",
            ncol=2, bbox_to_anchor=(.5, -.16))
    save(fig, "fig8_b_method_agreement_scatter")


def main():
    print(f"writing to {OUT}")
    fig1_a_bars_per_model()
    fig1_b_plane()
    fig1_c_forest()
    fig1_d_dumbbell()
    fig2_a_reuse_vs_p()
    fig2_b_reuse_by_task()
    fig2_c_reuse_strip()
    fig3_a_own_other_2x2()
    fig3_b_gap_by_model()
    fig3_c_paired_own_other()
    fig3_d_gap_vs_p()
    fig4_a_heatmaps()
    fig4_b_related_vs_unrelated()
    fig4_c_pair_strip()
    fig5_a_lift_forest()
    fig6_a_selectivity_vs_control()
    fig8_a_agreement_bars()
    fig8_b_agreement_scatter()
    inter = all_task_intersections()
    inter.to_csv(OUT / "all_task_intersection.csv", index=False)
    print(inter.round(1).to_string(index=False))
    fig7_a_mlp_share(inter)


if __name__ == "__main__":
    main()
