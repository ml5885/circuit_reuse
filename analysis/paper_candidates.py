"""Paper-shaped candidates for the within-task results: reuse, lift, circuit size.

The exploratory grids in paper2/explore show one panel per model and task, which
is too much for a paper. These keep every model-task cell visible but compress
the layout, either by drawing all 30 cells in one axes or by putting them on the
rows of a heatmap. Nothing is averaged over models or tasks.

Output goes to paper2/candidates/, one PNG per candidate.

Run: python -m analysis.paper_candidates
"""
from __future__ import annotations

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

from analysis.granularity_parity import (GRAN_COLORS, GRAN_COLORS_LIGHT,
                                         SECONDARY, _legend, _model_label,
                                         _ramp, _task_label)

REPO = Path(__file__).resolve().parent.parent
ANA = REPO / "results" / "granularity_parity_analysis"
OUT = REPO / "paper2" / "candidates"
OUT.mkdir(parents=True, exist_ok=True)

TASKS = ["addition", "boolean", "ioi", "mcqa", "arc_easy", "arc_challenge"]
TASK_COLORS = {"addition": "#5FA8D0", "boolean": "#E8B33C", "ioi": "#66C79C",
               "mcqa": "#DD7F72", "arc_easy": "#8E7CD0", "arc_challenge": "#B07AA1"}
MODELS = ["google/gemma-2-2b", "google/gemma-2-2b-it", "meta-llama/Llama-3.2-3B",
          "meta-llama/Llama-3.2-3B-Instruct", "qwen3-4b"]
KS = [1, 5, 10, 20, 30]
PS = [50, 75, 85, 90, 95, 96, 97, 98, 99, 100]
K_REF, P_REF = 10, 50

# A rotated model name has six heatmap rows of height to fit in, so the heatmap
# uses short forms rather than the names the other figures carry.
SHORT_MODEL = {"google/gemma-2-2b": "Gemma 2B", "google/gemma-2-2b-it": "Gemma 2B IT",
               "meta-llama/Llama-3.2-3B": "Llama 3B",
               "meta-llama/Llama-3.2-3B-Instruct": "Llama 3B It",
               "qwen3-4b": "Qwen3 4B"}
GRAN_NAME = {"head_mlp": "Component-level", "neuron": "Neuron-level"}
METHOD_NAME = {"eap_ig": "EAP-IG", "relp": "RelP"}
METHOD_MARK = {"eap_ig": "o", "relp": "^"}
METHOD_HATCH = {"eap_ig": "", "relp": "///"}

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                     "axes.titlesize": 15, "axes.labelsize": 15,
                     "xtick.labelsize": 13, "ytick.labelsize": 13,
                     "legend.fontsize": 13, "figure.titlesize": 16,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25, "grid.linewidth": .6,
                     "axes.axisbelow": True})


def save(fig, name):
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


def k_axis(ax):
    ax.set_xscale("log")
    ax.set_xticks(KS, [str(k) for k in KS])
    ax.minorticks_off()
    ax.set_xlabel("Top-$K$ (%)")


extraction = pd.read_csv(ANA / "extraction_tidy.csv")
extraction["nonempty"] = extraction.circuit_size > 0
ext = extraction.copy()
ext.loc[~ext.nonempty, ["reuse", "lift"]] = np.nan

# Lift is a difference of two accuracies measured on the same n examples, so its
# sampling error follows from the binomial error of each. Treating the two as
# independent overstates it slightly, since they share an evaluation set.
_n = ext.baseline_total
_var = (ext.ablation_accuracy * (1 - ext.ablation_accuracy)
        + ext.control_accuracy * (1 - ext.control_accuracy)) / _n
ext["lift_se"] = 100 * np.sqrt(_var) / ext.baseline_accuracy
ext.loc[~ext.nonempty, "lift_se"] = np.nan

# Chance-normalised necessity: each accuracy is first binned at chance,
# a+(S) = max(acc(S) - chance, 0), and the control-minus-circuit difference is
# taken relative to what the clean model had above chance. Cells whose clean
# margin is under 10% of the above-chance range are undefined.
CHANCE = {"addition": 0.0, "boolean": .5, "ioi": .5, "mcqa": .25, "arc_easy": .25,
          "arc_challenge": .25}
_c = ext.task.map(CHANCE)
_margin = ext.baseline_accuracy - _c
_ok = ext.nonempty & (_margin / (1 - _c) >= .1)
ext["lift_chance"] = np.where(_ok, 100 * ((ext.control_accuracy - _c).clip(lower=0)
                                          - (ext.ablation_accuracy - _c).clip(lower=0)) / _margin, np.nan)
ext["lift_chance_se"] = np.where(_ok, 100 * np.sqrt(_var) / _margin, np.nan)

cross = pd.read_csv(ANA / "cross_task_tidy.csv")
spec = (cross[cross.donor == cross.target]
        .rename(columns={"target": "task"})
        [["model", "method", "granularity", "K", "p", "task", "donor_size",
          "own_drop_pp", "foreign_mean_drop_pp", "specificity_gap_pp"]].copy())
spec.loc[spec.donor_size == 0, ["own_drop_pp", "specificity_gap_pp"]] = np.nan

SE_COLUMN = {"lift": "lift_se"}

METRICS = {"reuse": ("reuse", "Reuse@$P$ (%)", ext),
           "lift": ("lift", "Necessity (%)", ext),
           "size": ("circuit_size", "Shared circuit size", extraction),
           "specgap": ("specificity_gap_pp", "Specificity gap (accuracy points)", spec)}


def mark_empty(ax, positions, values):
    """Put a cross on the baseline wherever a bar is absent.

    A bar is absent only when the shared circuit is empty, which is a different
    statement from a bar of height zero and needs to look different from one.
    """
    missing = [x for x, v in zip(positions, values) if not np.isfinite(v)]
    ax.scatter(missing, [0] * len(missing), marker="x", s=42, linewidths=1.6,
               color="0.45", zorder=4)


def gran_legend(fig, markers=False, methods=False, pad=None, empty=False,
                extra_range=False):
    """Below the panels, never over the data.

    ``pad`` opens a gap between the x axis labels and the legend, which the
    layout engine otherwise leaves flush against them.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    kw = dict(ls="none", marker="s", ms=9) if markers else dict(lw=3)
    handles = [Line2D([], [], color=GRAN_COLORS[g], label=GRAN_NAME[g], **kw)
               for g in ("head_mlp", "neuron")]
    if methods:
        handles += [Patch(facecolor="0.75", edgecolor="white", hatch=METHOD_HATCH[m],
                          label=METHOD_NAME[m]) for m in ("eap_ig", "relp")]
    if extra_range:
        handles.append(Line2D([], [], color="0.3", lw=1.2, marker="_", ms=9,
                              label="95% interval"))
    if empty:
        handles.append(Line2D([], [], ls="none", marker="x", ms=8, mew=1.6,
                              color="0.45", label="Empty circuit"))
    _legend(fig, handles=handles, labels=[h.get_label() for h in handles],
            ncol=len(handles), loc="outside lower center")
    if pad is not None:
        fig.get_layout_engine().set(h_pad=pad)


# --- candidate A: every cell as one faint line ---------------------------

def cand_bands(metric: str):
    """Rows are methods, columns are the two sweeps. One line per cell.

    Thirty lines per granularity read as two bands, which is the claim. No
    summary line is drawn, so nothing is averaged.
    """
    column, ylabel, source = METRICS[metric]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), layout="constrained",
                             sharey=True)
    for r, method in enumerate(("eap_ig", "relp")):
        for c, (xcol, fixed, xlabel) in enumerate(
                [("K", source.p == P_REF, None),
                 ("p", source.K == K_REF, "Consensus threshold $P$ (%)")]):
            ax = axes[r][c]
            sub = source[(source.method == method) & fixed]
            for (model, task, gran), d in sub.groupby(["model", "task", "granularity"]):
                d = d.sort_values(xcol)
                ax.plot(d[xcol], d[column], lw=1.3, alpha=.45,
                        color=GRAN_COLORS[gran])
            if metric == "size":
                ax.set_yscale("symlog", linthresh=1)
            # Both rows share each column's x axis, so only the bottom row
            # names it, and one label at the figure level names the quantity.
            if xcol == "K":
                k_axis(ax)
                if r == 0:
                    ax.set_xlabel("")
            elif r == 1:
                ax.set_xlabel(xlabel)
            if r == 0:
                ax.set_title(f"Across top-$K$, at $P$={P_REF}%" if xcol == "K"
                             else f"Across $P$, at top-$K$={K_REF}%")
        axes[r][1].text(1.03, .5, METHOD_NAME[method], transform=axes[r][1].transAxes,
                        rotation=270, va="center", ha="left", fontsize=14)
    fig.supylabel(ylabel.replace("\n", " "), fontsize=15)
    gran_legend(fig, pad=.08)
    save(fig, f"{FIGNUM[metric + '_bands']}_{metric}_bands")


# --- candidate B: one dot per cell at each sweep value --------------------

def cand_dots(metric: str):
    """Categorical x, one dot per cell, granularities side by side.

    Rows are the methods. Pooling them would put 60 dots on each x position and
    hide the fact that the two methods disagree on some of them.
    """
    column, ylabel, source = METRICS[metric]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained", sharey=True)
    rng = np.random.default_rng(0)
    sweeps = [("K", KS, source.p == P_REF, "Top-$K$ (%)",
               f"Across top-$K$, at $P$={P_REF}%"),
              ("p", PS, source.K == K_REF, "Consensus threshold $P$ (%)",
               f"Across $P$, at top-$K$={K_REF}%")]
    for r, method in enumerate(("eap_ig", "relp")):
        for c, (xcol, values, fixed, xlabel, title) in enumerate(sweeps):
            ax = axes[r][c]
            sub = source[fixed & (source.method == method)]
            for i, gran in enumerate(("head_mlp", "neuron")):
                d = sub[sub.granularity == gran]
                for j, v in enumerate(values):
                    y = d[d[xcol] == v][column].dropna()
                    x = j + (i - .5) * .34 + rng.uniform(-.06, .06, len(y))
                    ax.scatter(x, y, s=26, alpha=.6, color=GRAN_COLORS[gran],
                               edgecolor="none")
            ax.set_xticks(range(len(values)),
                          [str(v) for v in values] if r == 1 else [])
            if r == 0:
                ax.set_title(title)
            else:
                ax.set_xlabel(xlabel)
            if metric == "size":
                ax.set_yscale("symlog", linthresh=1)
            if metric == "lift":
                ax.axhline(0, c="0.35", lw=1)
        axes[r][0].set_ylabel(ylabel.replace("\n", " "))
        axes[r][1].text(1.03, .5, METHOD_NAME[method], transform=axes[r][1].transAxes,
                        rotation=270, va="center", ha="left", fontsize=14)
    gran_legend(fig, markers=True, pad=.08)
    save(fig, f"{FIGNUM[metric + '_dots']}_{metric}_dots")


FIGNUM = {"reuse_dots": "01", "reuse_bands": "02",
          "lift_bars": "04", "lift_bars_pooled": "05", "lift_dots": "06",
          "lift_mean_bars": "07", "lift_mean_bars_pooled": "08",
          "own_other_eap_ig": "09", "own_other_relp": "10",
          "specgap_bars": "11", "specgap_bands": "12", "specgap_dots": "13",
          "specgap_mean_bars": "14", "specgap_bars_pooled": "15",
          "specgap_mean_bars_pooled": "15b",
          "size_bands": "19", "size_dots": "20"}

CONFIGS = [(g, m) for g in ("head_mlp", "neuron") for m in ("eap_ig", "relp")]


def cand_reuse_necessity(k=K_REF, p=P_REF, sweep="K", sub=None, necessity="lift"):
    """Reuse above, necessity below, one column per task.

    ``sweep`` picks which threshold the top row varies; the other is fixed at
    the value the bars use. ``necessity`` is the column the bars show: the
    paper's ``lift`` or the chance-binned ``lift_chance``.
    """
    reuse_src, nec_src = METRICS["reuse"][2], METRICS["lift"][2]
    nec_se = f"{necessity}_se"
    xcol, values = ("K", KS) if sweep == "K" else ("p", PS)
    fixed = (reuse_src.p == p) if sweep == "K" else (reuse_src.K == k)
    fig, axes = plt.subplots(2, len(TASKS), figsize=(19, 7.6), layout="constrained",
                             sharey="row")
    fig.get_layout_engine().set(h_pad=.09, w_pad=.04)
    x = np.arange(len(MODELS))
    width = .84 / 2
    any_empty = False
    for c, task in enumerate(TASKS):
        top, bot = axes[0][c], axes[1][c]
        d = reuse_src[(reuse_src.task == task) & fixed]
        for (model, gran, method), g in d.groupby(["model", "granularity", "method"]):
            g = g.sort_values(xcol)
            top.plot(g[xcol], g["reuse"], lw=1.5, alpha=.7, color=GRAN_COLORS[gran],
                     ls="-" if method == "eap_ig" else (0, (3, 1.4)))
        if sweep == "K":
            top.set_xscale("log")
            top.set_xticks(KS, [str(v) for v in KS])
            top.minorticks_off()
        else:
            top.set_xticks([50, 75, 90, 100])
        top.axvline(k if sweep == "K" else p, ls=":", c="0.45", lw=1.8, zorder=0)
        top.set_title(_task_label(task))
        top.set_xlabel("Top-$K$ (%)" if sweep == "K" else "$P$ (%)")

        n = nec_src[(nec_src.task == task) & (nec_src.K == k) & (nec_src.p == p)]
        for i, gran in enumerate(("head_mlp", "neuron")):
            cell = n[n.granularity == gran]
            heights = [cell.groupby("model")[necessity].mean().get(m, np.nan) for m in MODELS]
            any_empty |= not np.isfinite(heights).all()
            v = (cell.assign(v=cell[nec_se] ** 2)
                 .groupby("model").agg(v=("v", "sum"), k=("v", "size")))
            err = 1.96 * np.array([np.sqrt(v.v.get(m, np.nan)) / v.k.get(m, 1)
                                   for m in MODELS])
            bot.bar(x + (i - .5) * width, heights, width, color=GRAN_COLORS[gran],
                    edgecolor="white", linewidth=.6, yerr=err,
                    error_kw=dict(ecolor="0.3", elinewidth=1.2, capsize=2.5))
            mark_empty(bot, x + (i - .5) * width, heights)
        bot.axhline(0, c="0.35", lw=1.2)
        bot.set_xticks(x, [SHORT_MODEL[m] for m in MODELS], rotation=30, ha="right",
                       fontsize=11)
    # P is fixed when the top row sweeps top-K, so name the value it is fixed at.
    axes[0][0].set_ylabel(f"Reuse@{p} (%)" if sweep == "K" else "Reuse@$P$ (%)")
    axes[1][0].set_ylabel("Necessity (%)" if necessity == "lift" else "Necessity, above chance (%)")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=GRAN_COLORS[g], lw=3, label=GRAN_NAME[g])
               for g in ("head_mlp", "neuron")]
    handles += [Line2D([], [], color="0.35", lw=1.6, label=METHOD_NAME[m],
                       ls="-" if m == "eap_ig" else (0, (3, 1.4)))
                for m in ("eap_ig", "relp")]
    # The cross only means something when a bar is actually absent.
    if any_empty:
        handles.append(Line2D([], [], ls="none", marker="x", ms=8, mew=1.6,
                              color="0.45", label="Empty circuit" if necessity == "lift"
                              else "Empty circuit or clean model at chance"))
    _legend(fig, handles=handles, labels=[h.get_label() for h in handles],
            ncol=len(handles), loc="outside lower center")
    # The paper's caption states both thresholds, so the figure itself carries
    # them only in the sweep variants, where they are what tells them apart.
    if sub is not None:
        fig.suptitle(f"top-$K$={k}%, $P$={p}%")
    name = ("03_reuse_necessity" if sub is None else
            f"{sub}/across_{xcol}__K{k}_P{p}")
    save(fig, name + ("" if necessity == "lift" else "_chance"))


def cand_bars(metric: str, pool_methods=False):
    """One panel per task, models on x, one bar per configuration.

    Six tasks fill a 2 by 3 grid exactly, and putting the models on x makes a
    model's bars adjacent, which is the comparison to read. With
    ``pool_methods`` the two attribution methods are averaged into one bar per
    granularity and kept as dots, since that average hides a real disagreement
    on some tasks.
    """
    column, ylabel, source = METRICS[metric]
    sub = source[(source.K == K_REF) & (source.p == P_REF)]
    groups = ["head_mlp", "neuron"] if pool_methods else CONFIGS
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), layout="constrained", sharey=True)
    axes = axes.ravel()
    x = np.arange(len(MODELS))
    width = .84 / len(groups)
    for ax, task in zip(axes, TASKS):
        d = sub[sub.task == task]
        for i, group in enumerate(groups):
            gran = group if pool_methods else group[0]
            cell = d[d.granularity == gran]
            if not pool_methods:
                cell = cell[cell.method == group[1]]
            offset = (i - (len(groups) - 1) / 2) * width
            per_model = cell.groupby("model")[column]
            heights = [per_model.mean().get(m, np.nan) for m in MODELS]
            # 95% interval from the size of the evaluation set. Averaging two
            # methods averages two independent estimates, so the interval of the
            # mean shrinks by the usual factor.
            err = None
            if SE_COLUMN.get(metric):
                v = (cell.assign(v=cell[SE_COLUMN[metric]] ** 2)
                     .groupby("model").agg(v=("v", "sum"), k=("v", "size")))
                err = 1.96 * np.array([np.sqrt(v.v.get(m, np.nan)) / v.k.get(m, 1)
                                       for m in MODELS])
            ax.bar(x + offset, heights, width, color=GRAN_COLORS[gran],
                   hatch="" if pool_methods else METHOD_HATCH[group[1]],
                   edgecolor="white", linewidth=.6, yerr=err,
                   error_kw=dict(ecolor="0.3", elinewidth=1.2, capsize=2.5))
            mark_empty(ax, x + offset, heights)
        ax.axhline(0, c="0.35", lw=1.2)
        ax.set_xticks(x, [SHORT_MODEL[m] for m in MODELS], rotation=45, ha="right")
        ax.set_title(_task_label(task))
    for ax in (axes[0], axes[3]):
        ax.set_ylabel(ylabel.replace("\n", " "))
    gran_legend(fig, methods=not pool_methods, empty=True,
                extra_range=bool(SE_COLUMN.get(metric)))
    fig.suptitle(f"top-$K$={K_REF}%, $P$={P_REF}%")
    save(fig, f"{FIGNUM[metric + ('_bars_pooled' if pool_methods else '_bars')]}"
         f"_{metric}{'_bars_pooled' if pool_methods else '_bars'}")


def cand_bars_mean(metric: str, pool_methods=False):
    """The same bars averaged over the five models, with each model kept as a dot.

    The bar is the model mean and the dots are what it was taken over, so a bar
    that rests on one model can be told from one the models agree on.
    """
    column, ylabel, source = METRICS[metric]
    sub = source[(source.K == K_REF) & (source.p == P_REF)]
    groups = ["head_mlp", "neuron"] if pool_methods else CONFIGS
    fig, ax = plt.subplots(figsize=(13, 5.2), layout="constrained")
    x = np.arange(len(TASKS))
    width = .84 / len(groups)
    rng = np.random.default_rng(0)
    for i, group in enumerate(groups):
        gran = group if pool_methods else group[0]
        d = sub[sub.granularity == gran]
        if not pool_methods:
            d = d[d.method == group[1]]
        offset = (i - (len(groups) - 1) / 2) * width
        means = d.groupby("task")[column].mean()
        heights = [means.get(t, np.nan) for t in TASKS]
        mark_empty(ax, x + offset, heights)
        ax.bar(x + offset, heights, width,
               color=GRAN_COLORS[gran],
               hatch="" if pool_methods else METHOD_HATCH[group[1]],
               edgecolor="white", linewidth=.6)
        for j, task in enumerate(TASKS):
            y = d[d.task == task][column].dropna()
            ax.scatter(x[j] + offset + rng.uniform(-.02, .02, len(y)), y, s=16,
                       color="0.25", alpha=.8, zorder=3, edgecolor="none")
    ax.axhline(0, c="0.35", lw=1.2)
    ax.set_xticks(x, [_task_label(t) for t in TASKS])
    ax.set(ylabel=ylabel.replace("\n", " "),
           title=f"top-$K$={K_REF}%, $P$={P_REF}%")
    gran_legend(fig, methods=not pool_methods, empty=True)
    save(fig, f"{FIGNUM[metric + ('_mean_bars_pooled' if pool_methods else '_mean_bars')]}"
         f"_{metric}{'_mean_bars_pooled' if pool_methods else '_mean_bars'}")


def cand_own_other(method: str):
    """Own-circuit drop against the mean over other tasks' circuits.

    The gap is a difference, so a two-point gap on a ninety-point drop and a
    two-point gap on a three-point drop look the same until both are drawn.
    """
    sub = spec[(spec.K == K_REF) & (spec.p == P_REF) & (spec.method == method)]
    bars = [(g, c) for g in ("head_mlp", "neuron")
            for c in ("own_drop_pp", "foreign_mean_drop_pp")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), layout="constrained", sharey=True)
    axes = axes.ravel()
    x = np.arange(len(MODELS))
    width = .84 / len(bars)
    for ax, task in zip(axes, TASKS):
        d = sub[sub.task == task]
        for i, (gran, col) in enumerate(bars):
            vals = d[d.granularity == gran].set_index("model")[col]
            heights = [vals.get(m, np.nan) for m in MODELS]
            offset = (i - (len(bars) - 1) / 2) * width
            ax.bar(x + offset, heights, width, color=GRAN_COLORS[gran],
                   hatch="" if "own" in col else "///",
                   edgecolor="white", linewidth=.6)
            mark_empty(ax, x + offset, heights)
        ax.axhline(0, c="0.35", lw=1.2)
        ax.set_xticks(x, [SHORT_MODEL[m] for m in MODELS], rotation=45, ha="right")
        ax.set_title(_task_label(task))
    for ax in (axes[0], axes[3]):
        ax.set_ylabel("Accuracy drop (points)")
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=GRAN_COLORS[g], lw=3, label=GRAN_NAME[g])
               for g in ("head_mlp", "neuron")]
    handles += [Patch(facecolor="0.75", edgecolor="white", hatch=h, label=n)
                for h, n in (("", "Own circuit"), ("///", "Other tasks' circuits"))]
    handles.append(Line2D([], [], ls="none", marker="x", ms=8, mew=1.6, color="0.45",
                          label="Empty circuit"))
    _legend(fig, handles=handles, labels=[h.get_label() for h in handles],
            ncol=5, loc="outside lower center")
    fig.suptitle(f"{METHOD_NAME[method]}, top-$K$={K_REF}%, $P$={P_REF}%")
    save(fig, f"{FIGNUM['own_other_' + method]}_own_other_{method}")


FOCUS = "meta-llama/Llama-3.2-3B"
RELATED = ("arc_challenge", "arc_easy")


def _overlap_damage():
    """Pairwise overlap joined to the damage each donor circuit does."""
    overlap = pd.read_csv(REPO / "paper2" / "explore" / "overlap_sweep.csv")
    overlap = overlap[(overlap.K == K_REF) & (overlap.p == P_REF)]
    cross = pd.read_csv(ANA / "cross_task_tidy.csv")
    cross = cross[(cross.K == K_REF) & (cross.p == P_REF) & (cross.donor != cross.target)
                  & (cross.donor_size > 0) & (cross.own_drop_rel >= 20)].copy()
    key = ["model", "method", "granularity"]
    jac = {(*r[:3], *sorted(r[3:5])): r[5] for r in
           overlap[key + ["task_a", "task_b", "observed"]].itertuples(index=False)}
    cross["overlap"] = [jac.get((r.model, r.method, r.granularity,
                                 *sorted((r.donor, r.target))), np.nan)
                        for r in cross.itertuples()]
    # One means the other task's circuit is as damaging as the target's own.
    cross["specificity"] = 1 - cross.relative_drop_pct / cross.own_drop_rel
    return overlap, cross.dropna(subset=["overlap", "specificity"])


def _heatmap(ax, overlap, model, method, gran, annotate=True):
    d = overlap[(overlap.model == model) & (overlap.method == method)
                & (overlap.granularity == gran)]
    grid = np.full((len(TASKS), len(TASKS)), np.nan)
    for r in d.itertuples():
        i, j = TASKS.index(r.task_a), TASKS.index(r.task_b)
        grid[i, j] = grid[j, i] = r.observed
    im = ax.imshow(grid, cmap=_ramp(), vmin=0, vmax=1, interpolation="nearest")
    if annotate:
        for i in range(len(TASKS)):
            for j in range(len(TASKS)):
                if np.isfinite(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                            fontsize=8, color="white" if grid[i, j] > .55 else "0.2")
    ax.grid(False)
    return im


def cand_overlap_damage():
    """Overlap above, specificity below, one column per configuration."""
    overlap, cross = _overlap_damage()
    configs = [(g, m) for g in ("head_mlp", "neuron") for m in ("eap_ig", "relp")]
    fig, axes = plt.subplots(2, 4, figsize=(16, 9), layout="constrained",
                             height_ratios=[1, .95])
    fig.get_layout_engine().set(h_pad=.1, w_pad=.05)
    x = np.arange(len(MODELS))
    spec_vals: list[float] = []
    for c, (gran, method) in enumerate(configs):
        top, bot = axes[0][c], axes[1][c]
        im = _heatmap(top, overlap, FOCUS, method, gran)
        top.set_xticks(range(len(TASKS)), [_task_label(t) for t in TASKS],
                       rotation=30, ha="right", fontsize=10)
        top.set_yticks(range(len(TASKS)),
                       [_task_label(t) for t in TASKS] if c == 0 else [], fontsize=10)
        top.set_title(f"{GRAN_NAME[gran]}\n{METHOD_NAME[method]}", fontsize=14)

        d = cross[(cross.granularity == gran) & (cross.method == method)]
        med = d.groupby("model").specificity.median()
        heights = [med.get(m, np.nan) for m in MODELS]
        spec_vals.extend(h for h in heights if np.isfinite(h))
        bot.bar(x, heights, .7,
                color=GRAN_COLORS[gran], hatch=METHOD_HATCH[method],
                edgecolor="white", linewidth=.6)
        bot.axhline(0, c="0.35", lw=1.2)
        bot.set_xticks(x, [SHORT_MODEL[m] for m in MODELS], rotation=30, ha="right",
                       fontsize=11)
        if c:
            bot.sharey(axes[1][0])
            bot.tick_params(labelleft=False)
    # sharey copies column 0's limits, which only sees the component-level bars,
    # so set the range from every panel's values once they are all drawn.
    pad = .05 * max(1e-6, max(spec_vals) - min(0, min(spec_vals)))
    axes[1][0].set_ylim(min(0, min(spec_vals)) - pad, max(spec_vals) + pad)
    axes[1][0].set_ylabel("Specificity")
    fig.colorbar(im, ax=axes[0], location="right", fraction=.015, pad=.01,
                 label="Overlap")
    fig.suptitle(f"top-$K$={K_REF}%, $P$={P_REF}%")
    save(fig, "16_overlap_heatmap_specificity")


def cand_overlap_scatter():
    """All four configurations in one axes, filled for EAP-IG and open for RelP."""
    _, cross = _overlap_damage()
    fig, ax = plt.subplots(figsize=(8.5, 6), layout="constrained")
    for gran in ("head_mlp", "neuron"):
        for method in ("eap_ig", "relp"):
            d = cross[(cross.granularity == gran) & (cross.method == method)]
            ax.scatter(d.overlap, d.specificity, s=44, alpha=.6,
                       facecolor=GRAN_COLORS[gran] if method == "eap_ig" else "none",
                       edgecolor=GRAN_COLORS[gran], linewidth=1.2,
                       marker=METHOD_MARK[method])
    ax.axhline(0, c="0.35", lw=1.2, ls=":")
    ax.set(xlabel="Overlap", ylabel="Specificity", xlim=(-.05, 1.05),
           title=f"top-$K$={K_REF}%, $P$={P_REF}%")
    _config_marker_legend(fig)
    save(fig, "17_overlap_vs_specificity")


def cand_overlap_summary():
    """One point per model and configuration, so thirty pairs become five."""
    _, cross = _overlap_damage()
    fig, ax = plt.subplots(figsize=(8.5, 6), layout="constrained")
    for gran in ("head_mlp", "neuron"):
        for method in ("eap_ig", "relp"):
            d = (cross[(cross.granularity == gran) & (cross.method == method)]
                 .groupby("model")[["overlap", "specificity"]].median())
            ax.scatter(d.overlap, d.specificity, s=150, marker=METHOD_MARK[method],
                       facecolor=GRAN_COLORS[gran] if method == "eap_ig" else "none",
                       edgecolor=GRAN_COLORS[gran], linewidth=1.8, zorder=3)
    ax.axhline(0, c="0.35", lw=1.2, ls=":")
    ax.set(xlabel="Overlap", ylabel="Specificity", xlim=(-.05, 1.05),
           title=f"top-$K$={K_REF}%, $P$={P_REF}%")
    _config_marker_legend(fig)
    save(fig, "18_overlap_vs_specificity_by_model")


def _config_marker_legend(fig):
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], ls="none", marker=METHOD_MARK[m], ms=10,
                      markerfacecolor=GRAN_COLORS[g] if m == "eap_ig" else "none",
                      markeredgecolor=GRAN_COLORS[g], markeredgewidth=1.5,
                      label=f"{GRAN_NAME[g]}, {METHOD_NAME[m]}")
               for g in ("head_mlp", "neuron") for m in ("eap_ig", "relp")]
    _legend(fig, handles=handles, labels=[h.get_label() for h in handles],
            ncol=2, loc="outside lower center")


def _warm_ramp():
    """Yellow-to-deep-red sequential ramp, as the earlier overlap heatmaps used."""
    from matplotlib.colors import LinearSegmentedColormap
    from analysis.granularity_parity import MISSING_COLOR
    cm = LinearSegmentedColormap.from_list(
        "warm_overlap",
        ["#FFFCE8", "#FEE08B", "#FDAE61", "#F46D43", "#D73027", "#8C0B26"])
    return cm.with_extremes(bad=MISSING_COLOR)


DEPTH_BINS = 10
N_LAYERS = {"google/gemma-2-2b": 26, "google/gemma-2-2b-it": 26,
            "meta-llama/Llama-3.2-3B": 28,
            "meta-llama/Llama-3.2-3B-Instruct": 28, "qwen3-4b": 36}


def _depth_enrichment(comp: pd.DataFrame) -> pd.DataFrame:
    """Circuit share per depth bin, divided by the share of the pool in that bin.

    The bins are equal-width in relative depth, but layer counts do not divide
    evenly into ten, so a uniform circuit does not put 1/10 of itself in each bin.
    Gemma's 26 layers fall 3,2,3,2,3,3,2,2,3,3, and a uniform circuit would
    oscillate between .115 and .077 rather than sit at .10. Every layer offers the
    same number of components at both granularities, so dividing by the layers per
    bin removes the artifact and makes the null exactly 1 for every model.
    """
    bins = [f"bin{i}" for i in range(DEPTH_BINS)]
    out = comp.copy()
    for model, n in N_LAYERS.items():
        hist, _ = np.histogram(np.arange(n) / (n - 1), bins=DEPTH_BINS, range=(0, 1))
        rows = out.model == model
        out.loc[rows, bins] = out.loc[rows, bins].values / (hist / n)
    return out


def cand_circuit_anatomy():
    """What each basis is made of: component type on the left, depth on the right.

    Composition is only a question at the component granularity, since the neuron
    basis decomposes the MLPs alone and is 100% MLP by construction. Depth is a
    question at both, so the right panel is what actually compares the bases.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    comp = pd.read_csv(ANA / "circuit_composition.csv")
    depth = _depth_enrichment(comp)
    bins = [f"bin{i}" for i in range(DEPTH_BINS)]
    centers = (np.arange(DEPTH_BINS) + .5) / DEPTH_BINS
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2),
                             gridspec_kw={"width_ratios": [1, 1.05]},
                             layout="constrained")

    # Left: MLP against attention share, per model, averaged over the six tasks.
    # Pooling over models would report 62% MLP and hide that the split is really
    # between model families.
    ax = axes[0]
    hm = comp[comp.granularity == "head_mlp"]
    width = .38
    x = np.arange(len(MODELS))
    for i, method in enumerate(("eap_ig", "relp")):
        d = hm[hm.method == method].groupby("model").mlp_fraction.mean()
        mlp = np.array([d.get(m, np.nan) for m in MODELS])
        off = (i - .5) * width
        ax.bar(x + off, mlp, width, color=GRAN_COLORS["head_mlp"],
               hatch=METHOD_HATCH[method], edgecolor="white", linewidth=.8)
        ax.bar(x + off, 1 - mlp, width, bottom=mlp, color=SECONDARY,
               hatch=METHOD_HATCH[method], edgecolor="white", linewidth=.8)
        for xi, v in zip(x + off, mlp):
            if np.isfinite(v):
                ax.annotate(f"{v:.0%}", (xi, v), ha="center", va="bottom",
                            textcoords="offset points", xytext=(0, 3), fontsize=11)
    ax.set_xticks(x, [SHORT_MODEL[m] for m in MODELS], rotation=30, ha="right")
    # Headroom for the legend, so it never sits over a bar that reaches 100%.
    ax.set(ylabel="Fraction of shared circuit", ylim=(0, 1.42))
    ax.set_yticks(np.arange(0, 1.01, .25))
    # Colour names the component type, hatch names the attribution method, so the
    # legend lists the two encodings rather than their four combinations.
    handles = [Patch(facecolor=GRAN_COLORS["head_mlp"], edgecolor="white",
                     label="MLP blocks"),
               Patch(facecolor=SECONDARY, edgecolor="white", label="Attention heads")]
    handles += [Patch(facecolor="0.75", edgecolor="white", hatch=METHOD_HATCH[m],
                      label=METHOD_NAME[m]) for m in ("eap_ig", "relp")]
    _legend(ax, handles=handles, labels=[h.get_label() for h in handles],
            ncol=2, loc="upper center", fontsize=10.5)

    # Right: where the circuit sits in depth. Relative depth, since the models
    # range from 26 to 36 layers.
    ax = axes[1]
    # No band. Four min-max envelopes over six tasks overlap into noise, and the
    # per-cell spread is what the depth grid is for.
    for gran in ("head_mlp", "neuron"):
        for method in ("eap_ig", "relp"):
            sub = depth[(depth.granularity == gran) & (depth.method == method)]
            color = (GRAN_COLORS if method == "eap_ig" else GRAN_COLORS_LIGHT)[gran]
            ax.plot(centers, sub.groupby("task")[bins].mean().mean(), lw=2.4,
                    markersize=6, marker=METHOD_MARK[method],
                    ls="-" if method == "eap_ig" else (0, (4, 1.6)), color=color)
    ax.axhline(1, ls=":", c="0.4", lw=1.2, zorder=0)
    ax.annotate("uniform over depth", (.02, 1), textcoords="offset points",
                xytext=(0, 5), fontsize=10.5, color="0.4")
    ax.set(xlabel="Relative depth (layer / final layer)",
           ylabel="Circuit share $/$ pool share", ylim=(0, None))
    # Colour names the granularity, line style and marker name the method.
    handles = [Line2D([], [], color=GRAN_COLORS[g], lw=3, label=GRAN_NAME[g])
               for g in ("head_mlp", "neuron")]
    handles += [Line2D([], [], color="0.35", lw=1.8, marker=METHOD_MARK[m], ms=6,
                       ls="-" if m == "eap_ig" else (0, (4, 1.6)),
                       label=METHOD_NAME[m]) for m in ("eap_ig", "relp")]
    _legend(ax, handles=handles, labels=[h.get_label() for h in handles],
            ncol=2, fontsize=10.5)
    fig.suptitle(f"top-$K$={K_REF}%, $P$={P_REF}%")
    save(fig, "21_circuit_anatomy")


CDF_GRID = np.linspace(0, 1, 101)
EXTRACTION = REPO / "results" / "granularity_parity" / "granularity_parity_extraction"


def _depth_cdf() -> pd.DataFrame:
    """Empirical CDF of component depth for every model-task-method-granularity.

    Binning into ten equal-width depth bins is what forced the pool correction in
    `_depth_enrichment`; a CDF needs no bins, so the artifact cannot arise. Under a
    uniform circuit the CDF is the diagonal, because every layer offers the same
    number of components at both granularities.
    """
    cache = ANA / "circuit_depth_cdf.csv"
    cols = [f"q{i:03d}" for i in range(len(CDF_GRID))]
    if cache.exists():
        return pd.read_csv(cache)
    import re
    from analysis.circuit_overlap_chance import read_circuits
    layer_of = re.compile(r"\[layer=(\d+)")
    rows = []
    for r in read_circuits(EXTRACTION, K_REF, P_REF).itertuples():
        n = N_LAYERS.get(r.model)
        if not n or not r.components:
            continue
        depth = np.array([int(layer_of.search(c).group(1)) for c in r.components]) / (n - 1)
        cdf = (depth[None, :] <= CDF_GRID[:, None]).mean(axis=1)
        rows.append({"model": r.model, "task": r.task, "method": r.method,
                     "granularity": r.granularity, "size": len(depth),
                     **dict(zip(cols, cdf))})
    out = pd.DataFrame(rows)
    out.to_csv(cache, index=False)
    return out


def _composition_bars(ax, comp: pd.DataFrame, by: str, legend=True):
    """Stacked MLP/attention bars, one pair per model or per task."""
    from matplotlib.patches import Patch
    hm = comp[comp.granularity == "head_mlp"]
    groups, labels = ((MODELS, [SHORT_MODEL[m] for m in MODELS]) if by == "model"
                      else (TASKS, [_task_label(t) for t in TASKS]))
    width, x = .38, np.arange(len(groups))
    for i, method in enumerate(("eap_ig", "relp")):
        d = hm[hm.method == method].groupby(by).mlp_fraction.mean()
        mlp = np.array([d.get(g, np.nan) for g in groups])
        off = (i - .5) * width
        ax.bar(x + off, mlp, width, color=GRAN_COLORS["head_mlp"],
               hatch=METHOD_HATCH[method], edgecolor="white", linewidth=.8)
        ax.bar(x + off, 1 - mlp, width, bottom=mlp, color=SECONDARY,
               hatch=METHOD_HATCH[method], edgecolor="white", linewidth=.8)
        for xi, v in zip(x + off, mlp):
            if np.isfinite(v):
                ax.annotate(f"{v:.0%}", (xi, v), ha="center", va="bottom",
                            textcoords="offset points", xytext=(0, 3), fontsize=11)
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set(ylabel="% of shared circuit", ylim=(0, 1.42))
    ax.set_yticks(np.arange(0, 1.01, .25), ["0", "25", "50", "75", "100"])
    if not legend:
        return
    # Legends fill column-major, so interleave the two encodings to put the
    # component types on the first row and the methods on the second.
    handles = [Patch(facecolor=GRAN_COLORS["head_mlp"], edgecolor="white",
                     label="MLP blocks"),
               Patch(facecolor="0.75", edgecolor="white",
                     hatch=METHOD_HATCH["eap_ig"], label=METHOD_NAME["eap_ig"]),
               Patch(facecolor=SECONDARY, edgecolor="white", label="Attention heads"),
               Patch(facecolor="0.75", edgecolor="white",
                     hatch=METHOD_HATCH["relp"], label=METHOD_NAME["relp"])]
    _legend(ax, handles=handles, labels=[h.get_label() for h in handles],
            ncol=2, loc="upper center", fontsize=10.5)


def _depth_cdf_panel(ax, cdf: pd.DataFrame):
    from matplotlib.lines import Line2D
    cols = [f"q{i:03d}" for i in range(len(CDF_GRID))]
    ax.plot([0, 1], [0, 1], ls=":", c="0.4", lw=1.2, zorder=0)
    # Colour carries the granularity alone; the method is the dash pattern, so the
    # two RelP curves keep their granularity's hue rather than a lighter one.
    for gran in ("head_mlp", "neuron"):
        for method in ("eap_ig", "relp"):
            sub = cdf[(cdf.granularity == gran) & (cdf.method == method)]
            ax.plot(CDF_GRID, sub.groupby("task")[cols].mean().mean(), lw=2.4,
                    ls="-" if method == "eap_ig" else (0, (4, 1.6)),
                    color=GRAN_COLORS[gran])
    ax.set(xlabel="Relative depth", ylabel="Cumulative % of shared circuit",
           xlim=(0, 1), ylim=(0, 1))
    ax.set_yticks(np.arange(0, 1.01, .2), ["0", "20", "40", "60", "80", "100"])
    # Same interleave: granularities on the first row, methods on the second.
    handles = [Line2D([], [], color=GRAN_COLORS["head_mlp"], lw=3,
                      label=GRAN_NAME["head_mlp"]),
               Line2D([], [], color="0.35", lw=2.2, ls="-",
                      label=METHOD_NAME["eap_ig"]),
               Line2D([], [], color=GRAN_COLORS["neuron"], lw=3,
                      label=GRAN_NAME["neuron"]),
               Line2D([], [], color="0.35", lw=2.2, ls=(0, (4, 1.6)),
                      label=METHOD_NAME["relp"])]
    _legend(ax, handles=handles, labels=[h.get_label() for h in handles],
            ncol=3, loc="upper left", fontsize=10.5)


def cand_depth_cdf():
    """Composition on the left, depth as a CDF on the right.

    The CDF replaces the binned density of plot 21. It is monotone, so the four
    configurations separate without a spread band fighting them, and the uniform
    null is the diagonal rather than a level the binning cannot reach.
    """
    comp = pd.read_csv(ANA / "circuit_composition.csv")
    cdf = _depth_cdf()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2),
                             gridspec_kw={"width_ratios": [1, 1.05]},
                             layout="constrained")
    _composition_bars(axes[0], comp, "model")
    _depth_cdf_panel(axes[1], cdf)
    save(fig, "21c_circuit_anatomy_cdf")


def cand_overlap_own_other(pairwise=False):
    """Overlap above, own-against-other accuracy drop below, one column per config.

    With ``pairwise`` the bottom row keeps every (donor, target) pair as its own
    point instead of averaging the target tasks, so a circuit that destroys one
    other task and leaves four alone is visible rather than diluted to a fifth.

    Replaces the median-specificity bars of plot 16. Those collapse each model to
    one number and sit at zero wherever the foreign ablation drives the task to
    its chance floor, which is a saturated ratio rather than a measured tie. The
    scatter keeps every model-task cell and shows the tie directly: a point on the
    diagonal means the other tasks' circuits cost as much as the task's own.
    """
    from matplotlib.lines import Line2D
    overlap = pd.read_csv(ANA / "overlap_pairs.csv")
    overlap = overlap[(overlap.K == K_REF) & (overlap.p == P_REF)] \
        if "K" in overlap.columns else overlap
    cross = pd.read_csv(ANA / "cross_task_tidy.csv")
    own = cross[(cross.donor == cross.target) & (cross.K == K_REF)
                & (cross.p == P_REF) & (cross.donor_size > 0)]
    if pairwise:
        own = cross[(cross.donor != cross.target) & (cross.K == K_REF)
                    & (cross.p == P_REF) & (cross.donor_size > 0)
                    & cross.donor.isin(TASKS) & cross.target.isin(TASKS)]
        own = own.assign(foreign_mean_drop_pp=own.accuracy_drop_pp)
    configs = [(g, m) for g in ("head_mlp", "neuron") for m in ("eap_ig", "relp")]
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.6), layout="constrained",
                             height_ratios=[1, .95])

    for c, (gran, method) in enumerate(configs):
        top, bot = axes[0][c], axes[1][c]
        d = overlap[(overlap.method == method) & (overlap.granularity == gran)]
        grid = np.full((len(TASKS), len(TASKS)), np.nan)
        for (a, b), g in d.groupby(["task_a", "task_b"]):
            if a in TASKS and b in TASKS:
                i, j = TASKS.index(a), TASKS.index(b)
                grid[i, j] = grid[j, i] = g.observed.mean()
        # A circuit overlaps itself perfectly, so the diagonal is 1 by definition
        # rather than missing data.
        np.fill_diagonal(grid, 1.0)
        im = top.imshow(grid, cmap=_warm_ramp(), vmin=0, vmax=1, interpolation="nearest")
        for i in range(len(TASKS)):
            for j in range(len(TASKS)):
                if np.isfinite(grid[i, j]):
                    top.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                             fontsize=8, color="white" if grid[i, j] > .62 else "0.2")
        top.set_xticks(range(len(TASKS)), [_task_label(t) for t in TASKS],
                       rotation=30, ha="right", fontsize=10)
        top.set_yticks(range(len(TASKS)),
                       [_task_label(t) for t in TASKS] if c == 0 else [], fontsize=10)
        top.set_title(f"{METHOD_NAME[method]} \u2013 {GRAN_NAME[gran]}", fontsize=14)
        top.grid(False)

        s = own[(own.method == method) & (own.granularity == gran)]
        bot.plot([0, 100], [0, 100], ls=":", c="0.4", lw=1.2, zorder=0)
        for t in TASKS:
            st = s[s.target == t]
            bot.scatter(st.own_drop_pp, st.foreign_mean_drop_pp, s=110 if not pairwise else 40,
                        alpha=.85 if not pairwise else .6, color=TASK_COLORS[t],
                        edgecolor="black", linewidth=.6, zorder=3)
        bot.set(xlim=(-4, 104), ylim=(-4, 104))
        bot.set_xticks(range(0, 101, 25))
        bot.set_yticks(range(0, 101, 25), [str(v) for v in range(0, 101, 25)]
                       if c == 0 else [])
        bot.set_xlabel("Acc. drop from own circuit")
        if c == 0:
            bot.set_ylabel("Acc. drop on another task" if pairwise else "Acc. drop from other circuits")
    fig.colorbar(im, ax=axes[0], location="right", fraction=.015, pad=.01,
                 label="Overlap")
    handles = [Line2D([], [], ls="none", marker="o", ms=10, color=TASK_COLORS[t],
                      markeredgecolor="black", markeredgewidth=.6, label=_task_label(t)) for t in TASKS]
    _legend(fig, handles=handles, labels=[h.get_label() for h in handles], ncol=len(handles),
            loc="upper center", bbox_to_anchor=(.5, -.005), bbox_transform=fig.transFigure)
    save(fig, "23_overlap_own_other" + ("_pairwise" if pairwise else ""))


SEL_MODELS = ["google/gemma-2-2b", "google/gemma-2-2b-it", "meta-llama/Llama-3.2-3B",
              "meta-llama/Llama-3.2-3B-Instruct", "qwen3-4b", "qwen3-8b"]
SEL_SHORT = dict(SHORT_MODEL, **{"qwen3-8b": "Qwen3 8B"})
SEL_COND = [("shared_core", "Shared core"), ("residual_a", "Task-specific"),
            ("residual_b", "Task-complement"), ("random_control", "Random control")]
SEL_COLOR = {"shared_core": GRAN_COLORS["head_mlp"], "residual_a": SECONDARY,
             "residual_b": GRAN_COLORS_LIGHT["neuron"], "random_control": "#8A8A8A"}


def _selective(K: int | None = None):
    """Tidy selective-ablation rows. K=None uses the cached top-K=10% table.

    Every run of this experiment is at P=100%; only top-K varies.
    """
    if K is None or K == 10:
        return pd.read_csv(ANA / "selective_ablation_tidy.csv")
    from analysis.selective_ablation_summary import read
    return read(REPO / "results" / f"selective_ablation_k{K}")


def cand_selective_bars():
    """Option A: one bar pair per partition, with every model drawn on top."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    d = _selective()
    fig, ax = plt.subplots(figsize=(9, 5.2), layout="constrained")
    x = np.arange(len(SEL_COND))
    w = .36
    for k, (col, off, hatch) in enumerate([("target", -w / 2, ""),
                                           ("non_target", w / 2, "///")]):
        means = [d[d.condition == c][col].mean() for c, _ in SEL_COND]
        ax.bar(x + off, means, w, hatch=hatch, edgecolor="white", linewidth=.8,
               color=[SEL_COLOR[c] for c, _ in SEL_COND])
        for i, (c, _) in enumerate(SEL_COND):
            per_model = d[d.condition == c].groupby("model")[col].mean()
            ax.scatter(np.full(len(per_model), x[i] + off), per_model.values, s=22,
                       color="0.25", zorder=4, alpha=.8, linewidth=0)
    ax.set_xticks(x, [n for _, n in SEL_COND])
    ax.set_ylabel("Relative accuracy drop (%)")
    handles = [Patch(facecolor="0.75", edgecolor="white", label="Target task"),
               Patch(facecolor="0.75", edgecolor="white", hatch="///",
                     label="Other tasks"),
               Line2D([], [], ls="none", marker="o", ms=6, color="0.25",
                      label="One model")]
    _legend(ax, handles=handles, labels=[h.get_label() for h in handles], ncol=3,
            loc="upper right", fontsize=11)
    ax.set_ylim(0, max(d.groupby(["condition"]).target.mean()) * 1.55)
    save(fig, "24a_selective_bars")


def cand_selective_selectivity():
    """Option B: selectivity per partition, one dot per model, against zero.

    Selectivity is the target drop minus the mean drop on the other tasks. The
    claim is that the task-specific set does not beat its own random control, so
    the two rows should be read against each other, not against zero.
    """
    from matplotlib.lines import Line2D
    d = _selective().assign(sel=lambda f: f.target - f.non_target)
    fig, ax = plt.subplots(figsize=(9, 3.2), layout="constrained")
    for i, (c, name) in enumerate(SEL_COND):
        per_model = d[d.condition == c].groupby("model").sel.mean()
        y = len(SEL_COND) - 1 - i
        ax.scatter(per_model.values, np.full(len(per_model), y), s=70,
                   color=SEL_COLOR[c], edgecolor="white", linewidth=.8, zorder=3)
        ax.scatter([per_model.mean()], [y], s=200, marker="|", color="0.15",
                   linewidth=2.2, zorder=4)
    ax.axvline(0, ls=":", c="0.4", lw=1.2, zorder=0)
    ax.set_yticks(range(len(SEL_COND))[::-1], [n for _, n in SEL_COND])
    ax.set_xlabel("Selectivity: target drop $-$ other-task drop (points)")
    handles = [Line2D([], [], ls="none", marker="o", ms=8, color="0.6",
                      label="One model"),
               Line2D([], [], ls="none", marker="|", ms=12, color="0.15",
                      markeredgewidth=2.2, label="Mean")]
    _legend(fig, handles=handles, labels=[h.get_label() for h in handles], ncol=2,
            loc="outside lower center", fontsize=11)
    save(fig, "24b_selective_selectivity")


def cand_selective_by_task():
    """Option C: keep the per-task breakdown, drop the per-model panels."""
    from matplotlib.patches import Patch
    d = _selective()
    fig, axes = plt.subplots(1, len(TASKS), figsize=(19, 4.4), sharey=True,
                             layout="constrained")
    x = np.arange(len(SEL_COND))
    w = .36
    rng = np.random.default_rng(0)
    for ax, task in zip(axes, TASKS):
        sub = d[d.task_a == task]
        for col, off, hatch in [("target", -w / 2, ""), ("non_target", w / 2, "///")]:
            means, lo, hi = [], [], []
            for c, _ in SEL_COND:
                v = sub[sub.condition == c][col].dropna().values
                means.append(v.mean() if len(v) else np.nan)
                # Percentile bootstrap over the model-partner cells behind the bar.
                if len(v) > 1:
                    boot = rng.choice(v, (2000, len(v)), replace=True).mean(axis=1)
                    lo.append(means[-1] - np.percentile(boot, 2.5))
                    hi.append(np.percentile(boot, 97.5) - means[-1])
                else:
                    lo.append(0.)
                    hi.append(0.)
            ax.bar(x + off, means, w, hatch=hatch, edgecolor="white", linewidth=.8,
                   color=[SEL_COLOR[c] for c, _ in SEL_COND],
                   yerr=[lo, hi], error_kw=dict(ecolor="0.25", lw=1.1, capsize=2.5))
        # The partition names live in the legend, so the six panels do not repeat
        # four rotated tick labels each.
        ax.set_xticks(x, [""] * len(SEL_COND))
        ax.tick_params(axis="x", length=0)
        ax.set_title(_task_label(task), fontsize=22)
    axes[0].set_ylabel("Relative accuracy drop (%)", fontsize=17)
    for ax in axes:
        ax.tick_params(axis="y", labelsize=15)
    handles = [Patch(facecolor=SEL_COLOR[c], edgecolor="white", label=n)
               for c, n in SEL_COND]
    handles += [Patch(facecolor="0.75", edgecolor="white", label="Target task"),
                Patch(facecolor="0.75", edgecolor="white", hatch="///",
                      label="Other tasks")]
    # Draw once so the panel positions are final, then stretch the legend across
    # exactly the span the panels occupy rather than the whole figure width.
    fig.canvas.draw()
    x0 = axes[0].get_position().x0
    x1 = axes[-1].get_position().x1
    _legend(fig, handles=handles, labels=[h.get_label() for h in handles],
            ncol=len(handles), loc="upper left", mode="expand",
            bbox_to_anchor=(x0 - .012, -.13, (x1 - x0) + .024, .08),
            bbox_transform=fig.transFigure, fontsize=18)
    save(fig, "24c_selective_by_task")




def cand_selective_cells():
    """Selective ablation without averaging over models or tasks.

    One point per model and target task, averaged only over that task's five
    partners. A point below the diagonal means the partition costs the task it
    came from more than it costs the other tasks.
    """
    from matplotlib.lines import Line2D
    d = _selective()
    cell = (d.groupby(["condition", "model", "task_a"])[["target", "non_target"]]
            .mean().reset_index())
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.0), sharex=True, sharey=True,
                             layout="constrained")
    for ax, (cond, name) in zip(axes, SEL_COND):
        sub = cell[cell.condition == cond]
        lim = [-8, 104]
        ax.plot(lim, lim, ls=":", c="0.4", lw=1.2, zorder=0)
        for task in TASKS:
            s = sub[sub.task_a == task]
            ax.scatter(s.target, s.non_target, s=90, color=TASK_COLORS[task],
                       edgecolor="white", linewidth=.8, zorder=3, alpha=.9)
        ax.set(xlim=lim, ylim=lim)
        ax.set_title(name, fontsize=20)
        ax.set_xlabel("Acc. drop on own task", fontsize=16)
        ax.tick_params(labelsize=14)
    axes[0].set_ylabel("Acc. drop on other tasks", fontsize=16)
    handles = [Line2D([], [], ls="none", marker="o", ms=11,
                      color=TASK_COLORS[t], label=_task_label(t)) for t in TASKS]
    fig.canvas.draw()
    x0, x1 = axes[0].get_position().x0, axes[-1].get_position().x1
    _legend(fig, handles=handles, labels=[h.get_label() for h in handles],
            ncol=len(handles), loc="upper left", mode="expand",
            bbox_to_anchor=(x0 - .012, -.13, (x1 - x0) + .024, .08),
            bbox_transform=fig.transFigure, fontsize=17)
    save(fig, "25a_selective_cells")


def cand_selective_strip():
    """Selectivity per own task, one dot per model, one panel per partition."""
    from matplotlib.lines import Line2D
    d = _selective().assign(sel=lambda f: f.target - f.non_target)
    cell = d.groupby(["condition", "model", "task_a"]).sel.mean().reset_index()
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.4), sharex=True, sharey=True,
                             layout="constrained")
    for ax, (cond, name) in zip(axes, SEL_COND):
        sub = cell[cell.condition == cond]
        ax.axvline(0, ls=":", c="0.4", lw=1.2, zorder=0)
        for i, task in enumerate(TASKS):
            v = sub[sub.task_a == task].sel.values
            y = len(TASKS) - 1 - i
            ax.scatter(v, np.full(len(v), y), s=80, color=TASK_COLORS[task],
                       edgecolor="white", linewidth=.8, zorder=3, alpha=.9)
            if len(v):
                ax.scatter([v.mean()], [y], s=230, marker="|", color="0.15",
                           linewidth=2.4, zorder=4)
        ax.set_title(name, fontsize=20)
        ax.set_xlabel("Selectivity", fontsize=17)
        ax.tick_params(labelsize=14)
    axes[0].set_yticks(range(len(TASKS))[::-1], [_task_label(t) for t in TASKS],
                       fontsize=15)
    handles = [Line2D([], [], ls="none", marker="o", ms=10, color="0.6",
                      label="One model"),
               Line2D([], [], ls="none", marker="|", ms=14, color="0.15",
                      markeredgewidth=2.4, label="Mean over models")]
    _legend(fig, handles=handles, labels=[h.get_label() for h in handles], ncol=2,
            loc="upper center", bbox_to_anchor=(.5, -.02),
            bbox_transform=fig.transFigure, fontsize=17)
    save(fig, "25b_selective_strip")


SEL_MODEL_COLORS = {"google/gemma-2-2b": "#5FA8D0", "google/gemma-2-2b-it": "#AFD4E9",
                    "meta-llama/Llama-3.2-3B": "#E8B33C",
                    "meta-llama/Llama-3.2-3B-Instruct": "#F6DDA0",
                    "qwen3-4b": "#66C79C", "qwen3-8b": "#B3E3CD"}


def cand_selective_model_bars(K: int = 10):
    """Bar version of 25b. One panel per own task, one bar per model.

    Nothing is averaged over models or over own tasks. Each bar averages only the
    five partner tasks that define the partition. All runs are at P=100%.
    """
    from matplotlib.patches import Patch
    d = _selective(K).assign(sel=lambda f: f.target - f.non_target)
    cell = d.groupby(["condition", "model", "task_a"]).sel.mean().reset_index()
    fig, axes = plt.subplots(2, 3, figsize=(17, 8.4), sharey=True,
                             layout="constrained")
    x = np.arange(len(SEL_COND))
    w = .13
    for ax, task in zip(axes.ravel(), TASKS):
        sub = cell[cell.task_a == task]
        ax.axhline(0, c="0.35", lw=1.2, zorder=2)
        for m, model in enumerate(SEL_MODELS):
            vals = [sub[(sub.condition == c) & (sub.model == model)].sel.mean()
                    for c, _ in SEL_COND]
            ax.bar(x + (m - 2.5) * w, vals, w, color=SEL_MODEL_COLORS[model],
                   edgecolor="white", linewidth=.6, zorder=3)
        ax.set_xticks(x, [n for _, n in SEL_COND], fontsize=14, rotation=18,
                      ha="right")
        ax.set_title(_task_label(task), fontsize=20)
        ax.tick_params(axis="y", labelsize=14)
    for ax in axes[:, 0]:
        ax.set_ylabel("Selectivity", fontsize=17)
    handles = [Patch(facecolor=SEL_MODEL_COLORS[m], edgecolor="white",
                     label=SEL_SHORT[m]) for m in SEL_MODELS]
    _legend(fig, handles=handles, labels=[h.get_label() for h in handles],
            ncol=len(handles), loc="upper center", bbox_to_anchor=(.5, -.01),
            bbox_transform=fig.transFigure, fontsize=16)
    fig.suptitle(f"top-$K$={K}%, $P$=100%", fontsize=20)
    save(fig, "25c_selective_model_bars" if K == 10
         else f"sweeps/25c_selective_model_bars_K{K}")


def cand_depth_cdf_grid():
    """Per-model, per-task depth CDFs: the appendix grid behind plot 21c.

    Replaces the appendix layer figures, which are EAP at the component
    granularity on a six-model set that includes qwen3-8b.
    """
    from matplotlib.lines import Line2D
    cdf = _depth_cdf()
    cols = [f"q{i:03d}" for i in range(len(CDF_GRID))]
    fig, axes = plt.subplots(len(MODELS), len(TASKS), figsize=(20, 15),
                             sharex=True, sharey=True, layout="constrained")
    for r, model in enumerate(MODELS):
        for c, task in enumerate(TASKS):
            ax = axes[r][c]
            ax.plot([0, 1], [0, 1], ls=":", c="0.4", lw=1, zorder=0)
            for gran in ("head_mlp", "neuron"):
                for method in ("eap_ig", "relp"):
                    d = cdf[(cdf.model == model) & (cdf.task == task)
                            & (cdf.granularity == gran) & (cdf.method == method)]
                    if d.empty:
                        continue
                    ax.plot(CDF_GRID, d[cols].mean(), lw=2, color=GRAN_COLORS[gran],
                            ls="-" if method == "eap_ig" else (0, (4, 1.6)))
            ax.set(xlim=(0, 1), ylim=(0, 1))
            if r == 0:
                ax.set_title(_task_label(task))
            if c == 0:
                ax.set_ylabel(f"{SHORT_MODEL[model]}\nCumulative fraction")
            if r == len(MODELS) - 1:
                ax.set_xlabel("Relative depth")
    handles = [Line2D([], [], color=GRAN_COLORS[g], lw=3, label=GRAN_NAME[g])
               for g in ("head_mlp", "neuron")]
    handles += [Line2D([], [], color="0.35", lw=2.2,
                       ls="-" if m == "eap_ig" else (0, (4, 1.6)),
                       label=METHOD_NAME[m]) for m in ("eap_ig", "relp")]
    handles.append(Line2D([], [], color="0.4", lw=1.2, ls=":",
                          label="Uniform over depth"))
    _legend(fig, handles=handles, labels=[h.get_label() for h in handles], ncol=5,
            loc="upper center", bbox_to_anchor=(.5, -.01), bbox_transform=fig.transFigure)
    fig.suptitle(f"Depth CDF of the shared circuit, top-$K$={K_REF}%, $P$={P_REF}%")
    save(fig, "22c_depth_cdf_grid")


def cand_circuit_anatomy_split():
    """Variant of 21 that keeps a spread band by giving each method its own panel.

    One panel per method leaves two bands instead of four, and the interquartile
    range over tasks is narrow enough to read where a min-max envelope is not.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    comp = pd.read_csv(ANA / "circuit_composition.csv")
    depth = _depth_enrichment(comp)
    bins = [f"bin{i}" for i in range(DEPTH_BINS)]
    centers = (np.arange(DEPTH_BINS) + .5) / DEPTH_BINS
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2),
                             gridspec_kw={"width_ratios": [1, 1, 1]},
                             layout="constrained")

    ax = axes[0]
    hm = comp[comp.granularity == "head_mlp"]
    width, x = .38, np.arange(len(MODELS))
    for i, method in enumerate(("eap_ig", "relp")):
        d = hm[hm.method == method].groupby("model").mlp_fraction.mean()
        mlp = np.array([d.get(m, np.nan) for m in MODELS])
        off = (i - .5) * width
        ax.bar(x + off, mlp, width, color=GRAN_COLORS["head_mlp"],
               hatch=METHOD_HATCH[method], edgecolor="white", linewidth=.8)
        ax.bar(x + off, 1 - mlp, width, bottom=mlp, color=SECONDARY,
               hatch=METHOD_HATCH[method], edgecolor="white", linewidth=.8)
        for xi, v in zip(x + off, mlp):
            if np.isfinite(v):
                ax.annotate(f"{v:.0%}", (xi, v), ha="center", va="bottom",
                            textcoords="offset points", xytext=(0, 3), fontsize=11)
    ax.set_xticks(x, [SHORT_MODEL[m] for m in MODELS], rotation=30, ha="right")
    ax.set(ylabel="Fraction of shared circuit", ylim=(0, 1.42))
    ax.set_yticks(np.arange(0, 1.01, .25))
    handles = [Patch(facecolor=GRAN_COLORS["head_mlp"], edgecolor="white",
                     label="MLP blocks"),
               Patch(facecolor=SECONDARY, edgecolor="white", label="Attention heads")]
    handles += [Patch(facecolor="0.75", edgecolor="white", hatch=METHOD_HATCH[m],
                      label=METHOD_NAME[m]) for m in ("eap_ig", "relp")]
    _legend(ax, handles=handles, labels=[h.get_label() for h in handles],
            ncol=2, loc="upper center", fontsize=10.5)

    top = 0.
    for j, method in enumerate(("eap_ig", "relp")):
        ax = axes[1 + j]
        if j:
            ax.sharey(axes[1])
        for gran in ("head_mlp", "neuron"):
            sub = depth[(depth.granularity == gran) & (depth.method == method)]
            per_task = sub.groupby("task")[bins].mean()
            color = GRAN_COLORS[gran]
            ax.fill_between(centers, per_task.quantile(.25), per_task.quantile(.75),
                            color=color, alpha=.2, linewidth=0)
            ax.plot(centers, per_task.median(), lw=2.4, markersize=6,
                    marker=METHOD_MARK[method], color=color, label=GRAN_NAME[gran])
            top = max(top, per_task.quantile(.75).max())
        ax.axhline(1, ls=":", c="0.4", lw=1.2, zorder=0)
        ax.annotate(METHOD_NAME[method], (.03, .95), xycoords="axes fraction",
                    fontsize=13, va="top")
        ax.set_xlabel("Relative depth (layer / final layer)")
        if j == 0:
            ax.set_ylabel("Circuit share $/$ pool share")
        else:
            ax.tick_params(labelleft=False)
    # sharey copies the first panel's limits, so set the range once both are drawn.
    axes[1].set_ylim(0, top * 1.08)
    axes[1].annotate("uniform over depth", (.02, 1), textcoords="offset points",
                     xytext=(0, 5), fontsize=10.5, color="0.4")
    _legend(axes[2], fontsize=10.5)
    fig.suptitle(f"top-$K$={K_REF}%, $P$={P_REF}%")
    save(fig, "21b_circuit_anatomy_split")


def cand_depth_grid():
    """Per-model, per-task depth profiles: the appendix grid behind the summary.

    The existing appendix layer figures are EAP at the component granularity only.
    This covers all four configurations on the five main models.
    """
    from matplotlib.lines import Line2D
    comp = _depth_enrichment(pd.read_csv(ANA / "circuit_composition.csv"))
    bins = [f"bin{i}" for i in range(DEPTH_BINS)]
    centers = (np.arange(DEPTH_BINS) + .5) / DEPTH_BINS
    fig, axes = plt.subplots(len(MODELS), len(TASKS), figsize=(20, 15),
                             sharex=True, sharey="row", layout="constrained")
    for r, model in enumerate(MODELS):
        for c, task in enumerate(TASKS):
            ax = axes[r][c]
            for gran in ("head_mlp", "neuron"):
                for method in ("eap_ig", "relp"):
                    d = comp[(comp.model == model) & (comp.task == task)
                             & (comp.granularity == gran) & (comp.method == method)]
                    if d.empty:
                        continue
                    color = (GRAN_COLORS if method == "eap_ig"
                             else GRAN_COLORS_LIGHT)[gran]
                    ax.plot(centers, d[bins].mean(), lw=1.8, markersize=4,
                            marker=METHOD_MARK[method], color=color,
                            ls="-" if method == "eap_ig" else (0, (4, 1.6)))
            ax.axhline(1, ls=":", c="0.4", lw=1, zorder=0)
            if r == 0:
                ax.set_title(_task_label(task))
            if c == 0:
                ax.set_ylabel(f"{SHORT_MODEL[model]}\nCircuit $/$ pool")
            if r == len(MODELS) - 1:
                ax.set_xlabel("Relative depth")
    handles = [Line2D([], [], color=GRAN_COLORS[g], lw=3, label=GRAN_NAME[g])
               for g in ("head_mlp", "neuron")]
    handles += [Line2D([], [], color="0.35", lw=1.8, marker=METHOD_MARK[m], ms=6,
                       ls="-" if m == "eap_ig" else (0, (4, 1.6)),
                       label=METHOD_NAME[m]) for m in ("eap_ig", "relp")]
    # Anchored in figure coordinates below the axes, so it clears the bottom row's
    # x labels; save() uses bbox_inches="tight" and grows the canvas to include it.
    _legend(fig, handles=handles, labels=[h.get_label() for h in handles], ncol=4,
            loc="upper center", bbox_to_anchor=(.5, -.01), bbox_transform=fig.transFigure)
    fig.suptitle(f"Depth profile of the shared circuit, top-$K$={K_REF}%, $P$={P_REF}%")
    save(fig, "22_depth_grid")


# Candidates chosen for the paper, copied into its figures/ tree so the two
# never drift. Anything not listed here stays a candidate only.
PAPER = (REPO / "paper2" /
         "_ICLR27__How_Much_Do_Circuits_Tell_Us__"
         "Measuring_the_Consistency_and_Specificity_of_Language_Model_Circuits" / "figures")
CHOSEN = {"03_reuse_necessity": "within_task/reuse_necessity.png",
          "20_size_dots": "within_task/circuit_size.png",
          "21c_circuit_anatomy_cdf": "within_task/circuit_anatomy.png",
          "22c_depth_cdf_grid": "within_task/depth_cdf_grid.png",
          "23_overlap_own_other": "cross_task/overlap_own_other.png",
          "24c_selective_by_task": "cross_task/selective_by_task.png"}


def sync_paper():
    import shutil
    if not PAPER.exists():
        print(f"  paper figures/ not found at {PAPER}")
        return
    for name, dest in CHOSEN.items():
        target = PAPER / dest
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(OUT / f"{name}.png", target)
        print(f"  -> {target.relative_to(REPO)}")


def main():
    print("writing to", OUT)
    for metric in ("reuse", "size"):
        cand_bands(metric)
    for metric in ("reuse", "lift", "size"):
        cand_dots(metric)
    for pool in (False, True):
        cand_bars("lift", pool_methods=pool)
        cand_bars_mean("lift", pool_methods=pool)
    cand_reuse_necessity()
    (OUT / "sweeps").mkdir(exist_ok=True)
    for pv in (50, 75, 95, 100):
        for kv in KS:
            cand_reuse_necessity(k=kv, p=pv, sweep="K", sub="sweeps")
        cand_reuse_necessity(k=K_REF, p=pv, sweep="p", sub="sweeps")
    cand_bands("specgap")
    cand_dots("specgap")
    cand_bars("specgap")
    cand_bars_mean("specgap")
    for method in ("eap_ig", "relp"):
        cand_own_other(method)
    cand_overlap_damage()
    cand_overlap_scatter()
    cand_overlap_summary()
    sync_paper()


if __name__ == "__main__":
    main()
