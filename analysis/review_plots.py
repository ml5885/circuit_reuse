"""Plots requested by paper2/claude_review.md that need no GPU.

Inputs: results/granularity_parity (metrics.json, cross_task_*.json),
results/granularity_parity_analysis/*.csv, cache/*.jsonl (per-example
component scores), results/pretraining_v2 (OLMo checkpoints), and the task
generators in circuit_reuse.dataset (offline HF cache).

Run: python -m analysis.review_plots [A1 A2 ...]
Writes paper2/review_plots/*.png and paper2/review_plots.md.
"""
from __future__ import annotations

import glob
import itertools
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from analysis.granularity_parity import (CONFIG_ORDER, MODEL_COLORS, OTHER_COLOR,
                                         SECONDARY, _color, _fill_kw, _label,
                                         _legend, _model_label, _panel, _task_label)
from analysis.pat_confound_checks import (cache_path, jaccard, load_scores,
                                          out_degree, shared, topk_sets)

REPO = Path(__file__).resolve().parent.parent
ANA = REPO / "results" / "granularity_parity_analysis"
EXT = REPO / "results" / "granularity_parity" / "granularity_parity_extraction"
OLMO = REPO / "results" / "pretraining_v2" / "results_pretraining_v2"
OUT = REPO / "paper2" / "review_plots"
OUT.mkdir(parents=True, exist_ok=True)

K, P = 10, 50
KS = [1, 5, 10, 20, 30]
MODELS = ["google/gemma-2-2b", "google/gemma-2-2b-it", "meta-llama/Llama-3.2-3B",
          "meta-llama/Llama-3.2-3B-Instruct", "qwen3-4b"]
TASKS = ["addition", "boolean", "ioi", "mcqa", "arc_easy", "arc_challenge"]
CHANCE = {"addition": 0.0, "boolean": .5, "ioi": .5, "mcqa": .25, "arc_easy": .25,
          "arc_challenge": .25}
SAME_COUNTERFACTUAL = {frozenset(p) for p in itertools.combinations(["mcqa", "arc_easy", "arc_challenge"], 2)}
MODEL_COLOR = dict(zip(MODELS, MODEL_COLORS))
TASK_COLORS = dict(zip(TASKS, ["#2E6E96", "#5FA8D0", "#E8B33C", "#66C79C", "#DD7F72", "#8E7CD0"]))
HEAD_MLP = [("eap_ig", "head_mlp"), ("relp", "head_mlp")]
OWN_COLOR, A1_OTHER = "#2E6E96", "#D9711F"

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                     "axes.titlesize": 11, "axes.labelsize": 10,
                     "xtick.labelsize": 9, "ytick.labelsize": 9,
                     "legend.fontsize": 8.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": .25, "grid.linewidth": .6, "axes.axisbelow": True})

extraction = pd.read_csv(ANA / "extraction_tidy.csv")
cross = pd.read_csv(ANA / "cross_task_tidy.csv")
overlap = pd.read_csv(ANA / "overlap_pairs.csv")
NOTES: dict[str, list[str]] = {}


def note(item, text):
    NOTES.setdefault(item, []).append(text)


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}.png")


def md_table(df: pd.DataFrame, fmt="{:.2f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for r in df.astype(object).itertuples(index=False):
        lines.append("| " + " | ".join(fmt.format(v) if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)


# --- shared circuits from metrics.json --------------------------------------

def parse_component(s: str):
    m = re.match(r"(\w+)\[layer=(\d+), index=(\d+)\]", s)
    return int(m.group(2)), m.group(1), int(m.group(3))


def metrics(method, gran, model, task) -> dict:
    stem = model.replace("/", "_")
    m = "eap_ig__ig5" if method == "eap_ig" else method
    paths = glob.glob(str(EXT / f"granularity_parity_{method}_{gran}" / f"{stem}__main__{task}__{m}__*" / "metrics.json"))
    assert len(paths) == 1, (method, gran, model, task, paths)
    return json.loads(Path(paths[0]).read_text())


_SHARED = {}


def shared_set(method, gran, model, task, k=K, p=P) -> set:
    key = (method, gran, model, task, k, p)
    if key not in _SHARED:
        cell = metrics(method, gran, model, task)["by_k"][str(k)]["thresholds"][str(p)]
        _SHARED[key] = {parse_component(c) for c in cell["shared_components"]}
    return _SHARED[key]


_POOL = {}


def pool(model, gran):
    if (model, gran) not in _POOL:
        _POOL[(model, gran)] = int(overlap[(overlap.model == model) & (overlap.granularity == gran)].pool.iloc[0])
    return _POOL[(model, gran)]


_CACHE = {}


def scores(model, task, method):
    if (model, task, method) not in _CACHE:
        _CACHE[(model, task, method)] = load_scores(cache_path(model, task, method))
    return _CACHE[(model, task, method)]


def kind_counts(comps):
    return Counter(kind for _, kind, _ in comps)


# --- A1 dose-response -------------------------------------------------------

def a1():
    d = cross[(cross.p == P) & (cross.donor == cross.target)]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    rows = []
    for j, key in enumerate(CONFIG_ORDER):
        s = d[(d.method == key[0]) & (d.granularity == key[1])]
        for i, (col_own, col_oth, ylab) in enumerate([("own_drop_pp", "foreign_mean_drop_pp", "Accuracy drop (pp)"),
                                                     ("own_drop_rel", "foreign_mean_drop_rel", "Relative drop (%)")]):
            ax = axes[i, j]
            for m in MODELS:
                sm = s[s.model == m].groupby("K")[[col_own, col_oth]].mean().reindex(KS)
                ax.plot(range(len(KS)), sm[col_own], color=OWN_COLOR, lw=.8, alpha=.4)
                ax.plot(range(len(KS)), sm[col_oth], color=A1_OTHER, lw=.8, alpha=.4)
            mean = s.groupby("K")[[col_own, col_oth]].mean().reindex(KS)
            ax.plot(range(len(KS)), mean[col_own], "-o", color=OWN_COLOR, lw=2.2, label="Own circuit")
            ax.plot(range(len(KS)), mean[col_oth], "-o", color=A1_OTHER, lw=2.2, label="Other circuits (mean)")
            ax.axhline(0, color="0.25", lw=.8)
            if i == 0:
                _panel(ax, _label(key))
                for k in KS:
                    rows.append(dict(config=_label(key), K=k, own=mean.loc[k, col_own],
                                     other=mean.loc[k, col_oth], gap=mean.loc[k, col_own] - mean.loc[k, col_oth]))
            if j == 0:
                ax.set_ylabel(ylab)
        sizes = extraction[(extraction.method == key[0]) & (extraction.granularity == key[1]) & (extraction.p == P)]
        sizes = sizes.groupby("K").circuit_size.mean().reindex(KS)
        for ax in axes[:, j]:
            ax.set_xticks(range(len(KS)))
            ax.set_xticklabels([f"{k}\n({v:,.0f})" for k, v in zip(KS, sizes)])
        axes[1, j].set_xlabel("K (%)")
    _legend(axes[1, 0], loc="lower right")
    save(fig, "A1_dose_response")
    t = pd.DataFrame(rows)
    t = t.pivot(index="K", columns="config", values="gap").reindex(KS).reset_index().astype({"K": int})
    note("A1", "Specificity gap (own minus other, pp) by K at P=50:\n\n" + md_table(t, "{:.1f}"))


# --- A2 drop vs overlap regression -----------------------------------------

def a2():
    d = cross[(cross.K == K) & (cross.p == P) & (cross.donor != cross.target) & (cross.donor_size > 0)]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.4))
    rows = []
    for j, key in enumerate(CONFIG_ORDER):
        s = d[(d.method == key[0]) & (d.granularity == key[1])].copy()
        s["jaccard"] = [jaccard_sets(shared_set(key[0], key[1], r.model, r.target),
                                     shared_set(key[0], key[1], r.model, r.donor)) for r in s.itertuples()]
        s["log_size"] = np.log10(s.donor_size)
        s["size_frac"] = [r.donor_size / pool(r.model, key[1]) for r in s.itertuples()]
        y = s.relative_drop_pct.to_numpy()
        for name, cols in [("J only", ["jaccard"]), ("J + log size", ["jaccard", "log_size"]),
                           ("J + log size + model", ["jaccard", "log_size"] + [f"m_{m}" for m in MODELS[1:]])]:
            for m in MODELS[1:]:
                s[f"m_{m}"] = (s.model == m).astype(float)
            X = np.column_stack([np.ones(len(s))] + [s[c].to_numpy() for c in cols])
            beta, res, *_ = np.linalg.lstsq(X, y, rcond=None)
            yhat = X @ beta
            r2 = 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
            dof = len(y) - X.shape[1]
            sigma2 = ((y - yhat) ** 2).sum() / dof
            se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
            rows.append(dict(config=_label(key), model_=name, n=len(y),
                             beta_J=beta[1], t_J=beta[1] / se[1],
                             beta_logsize=beta[2] if "log_size" in cols else np.nan,
                             t_logsize=beta[2] / se[2] if "log_size" in cols else np.nan, R2=r2))
        rho_j = stats.spearmanr(s.jaccard, y).statistic
        rho_s = stats.spearmanr(s.log_size, y).statistic
        rho_js = stats.spearmanr(s.jaccard, s.log_size).statistic
        rows.append(dict(config=_label(key), model_="Spearman", n=len(y), beta_J=rho_j, t_J=np.nan,
                         beta_logsize=rho_s, t_logsize=np.nan, R2=rho_js))
        for i, (xcol, xlab) in enumerate([("jaccard", "Jaccard$(S^A, S^B)$"), ("size_frac", "$|S^B|$ / pool")]):
            ax = axes[i, j]
            for m in MODELS:
                sm = s[s.model == m]
                ax.scatter(sm[xcol], sm.relative_drop_pct, s=14, color=MODEL_COLOR[m], alpha=.8,
                           edgecolor="0.3", lw=.3, label=_model_label(m))
            if i == 0:
                _panel(ax, _label(key))
                ax.text(.98, .04, f"Spearman $\\rho$ = {rho_j:.2f}", transform=ax.transAxes, ha="right", fontsize=8.5)
            else:
                ax.set_xscale("log")
                ax.text(.98, .04, f"Spearman $\\rho$ = {rho_s:.2f}", transform=ax.transAxes, ha="right", fontsize=8.5)
            ax.set_xlabel(xlab)
            ax.axhline(0, color="0.25", lw=.8)
            if j == 0:
                ax.set_ylabel("Relative drop on task A from $S^B$ (%)")
    _legend(axes[0, 0], loc="upper left", fontsize=7.5)
    save(fig, "A2_drop_vs_overlap")
    t = pd.DataFrame(rows).rename(columns={"model_": "regression"})
    note("A2", "OLS of relative drop Δ^B_A (%) on Jaccard(S^A,S^B) and log10|S^B| (K=10, P=50, all ordered "
               "off-diagonal pairs, five models). The Spearman rows give ρ(J, drop), ρ(log size, drop) and, in the R2 "
               "column, ρ(J, log size).\n\n" + md_table(t, "{:.2f}"))


def jaccard_sets(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


# --- A3 pigeonhole ---------------------------------------------------------

def a3():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    rows, comp_rows = [], []
    for ax, key in zip(axes, HEAD_MLP):
        for m in MODELS:
            comps, _ = scores(m, "addition", key[0])
            n_kind = kind_counts(comps)
            sets = {t: shared_set(key[0], key[1], m, t) for t in TASKS}
            for t in TASKS:
                kc = kind_counts(sets[t])
                comp_rows.append(dict(config=_label(key), model=_model_label(m), task=_task_label(t),
                                      mlp=f"{kc['mlp']}/{n_kind['mlp']}", heads=f"{kc['head']}/{n_kind['head']}",
                                      control_shortfall=max(0, kc["mlp"] - (n_kind["mlp"] - kc["mlp"]))
                                      + max(0, kc["head"] - (n_kind["head"] - kc["head"]))))
            for a, b in itertools.combinations(TASKS, 2):
                ka, kb = kind_counts(sets[a]), kind_counts(sets[b])
                floor = sum(max(0, ka[k] + kb[k] - n_kind[k]) for k in n_kind)
                obs = len(sets[a] & sets[b])
                rows.append(dict(config=_label(key), model=m, a=a, b=b, observed=obs, floor=floor,
                                 union=len(sets[a] | sets[b]), excess=obs - floor))
                ax.scatter(floor, obs, color=MODEL_COLOR[m], s=26, edgecolor="0.3", lw=.4, alpha=.85,
                           label=_model_label(m) if (a, b) == ("addition", "boolean") else None)
        lim = max(max(r["observed"], r["floor"]) for r in rows if r["config"] == _label(key)) + 2
        ax.plot([0, lim], [0, lim], color="0.3", lw=.8, ls="--")
        ax.set_xlim(-1, lim); ax.set_ylim(-1, lim)
        ax.set_xlabel("Combinatorial minimum, summed over component types")
        ax.set_ylabel("Observed $|A \\cap B|$")
        _panel(ax, _label(key))
    _legend(axes[0], loc="upper left")
    save(fig, "A3_pigeonhole")
    t = pd.DataFrame(rows)
    summary = t.groupby(["config", "model"]).agg(pairs_at_floor=("excess", lambda x: int((x == 0).sum())),
                                                 n_pairs=("excess", "size"),
                                                 mean_observed=("observed", "mean"),
                                                 mean_floor=("floor", "mean")).reset_index()
    summary["model"] = summary.model.map(_model_label)
    note("A3", "Task pairs whose observed intersection equals the combinatorial minimum (K=10, P=50):\n\n"
               + md_table(summary, "{:.1f}"))
    c = pd.DataFrame(comp_rows)
    note("A3", "Composition of S_P and the size of the control that could be drawn from V \\ S_P "
               "(`control_shortfall` = components the type-matched control is short by):\n\n" + md_table(c))


# --- A4 enrichment ---------------------------------------------------------

def a4():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    rows = []
    x = np.arange(len(MODELS))
    for ax, (which, title) in zip(axes, [("shared", "Shared circuit $S_{50}$"), ("example", "Per-example circuits")]):
        for off, key in zip([-.2, .2], HEAD_MLP):
            vals = []
            for m in MODELS:
                comps, _ = scores(m, "addition", key[0])
                base = kind_counts(comps)["mlp"] / len(comps)
                fracs = []
                for t in TASKS:
                    if which == "shared":
                        s = shared_set(key[0], key[1], m, t)
                        if s:
                            fracs.append(kind_counts(s)["mlp"] / len(s))
                    else:
                        comps, S = scores(m, t, key[0])
                        M = topk_sets(S, K)
                        is_mlp = np.array([c[1] == "mlp" for c in comps])
                        fracs.append(float(np.broadcast_to(is_mlp, M.shape)[M].mean()))
                enr = np.log2(np.mean(fracs) / base)
                vals.append(enr)
                rows.append(dict(config=_label(key), model=_model_label(m), set=which, mlp_base_rate=base,
                                 mlp_fraction=float(np.mean(fracs)), log2_enrichment=float(enr)))
            ax.bar(x + off, vals, .38, label=_label(key), **_fill_kw(key))
        ax.axhline(0, color="0.25", lw=.8)
        ax.set_xticks(x)
        ax.set_xticklabels([_model_label(m) for m in MODELS], rotation=15, ha="right")
        _panel(ax, title)
    axes[0].set_ylabel("$\\log_2$ MLP enrichment over base rate")
    _legend(axes[0], loc="upper left")
    save(fig, "A4_mlp_enrichment")
    note("A4", "MLP share of the circuit against the MLP share of the node pool (mean over tasks with a non-empty circuit, K=10, P=50):\n\n"
               + md_table(pd.DataFrame(rows), "{:.2f}"))


# --- A5 per-task decomposition ---------------------------------------------

def a5():
    d = cross[(cross.K == K) & (cross.p == P) & (cross.donor == cross.target)]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharex=True, sharey=True)
    for ax, key in zip(axes, CONFIG_ORDER):
        s = d[(d.method == key[0]) & (d.granularity == key[1])]
        for t in TASKS:
            st = s[s.target == t]
            ax.scatter(st.foreign_mean_drop_pp, st.own_drop_pp, color=TASK_COLORS[t], s=34, edgecolor="0.3",
                       lw=.4, label=_task_label(t))
        ax.plot([-10, 100], [-10, 100], color="0.3", lw=.8, ls="--")
        ax.set_xlabel("Mean drop from other tasks' circuits (pp)")
        _panel(ax, _label(key))
    axes[0].set_ylabel("Drop from own circuit (pp)")
    _legend(axes[0], loc="upper left")
    save(fig, "A5a_own_vs_other_by_task")

    fig, ax = plt.subplots(figsize=(9, 3.6))
    x = np.arange(len(TASKS))
    w = .8 / len(CONFIG_ORDER)
    rows = []
    for i, key in enumerate(CONFIG_ORDER):
        s = d[(d.method == key[0]) & (d.granularity == key[1])]
        g = s.groupby("target").specificity_gap_pp
        means = g.mean().reindex(TASKS)
        sems = g.sem().reindex(TASKS)
        ax.bar(x + (i - 1.5) * w, means, w * .92, yerr=sems, label=_label(key),
               error_kw=dict(lw=.7, capsize=2), **_fill_kw(key))
        for t in TASKS:
            rows.append(dict(config=_label(key), task=_task_label(t), summed_gap=g.sum().get(t, np.nan),
                             mean_gap=means[t], share=g.sum().get(t, 0) / g.sum().sum() * 100))
    ax.axhline(0, color="0.25", lw=.8)
    ax.set_xticks(x)
    ax.set_xticklabels([_task_label(t) for t in TASKS])
    ax.set_ylabel("Specificity gap (pp), mean over models")
    _legend(ax, loc="upper right", ncol=2)
    save(fig, "A5b_gap_by_task")
    t = pd.DataFrame(rows)
    t = t.pivot(index="task", columns="config", values="share").reindex([_task_label(x) for x in TASKS])
    note("A5", "Share of the summed specificity gap carried by each task (%, K=10, P=50):\n\n"
               + md_table(t.reset_index(), "{:.0f}"))


# --- A6 two-way interaction ------------------------------------------------

def a6():
    d = cross[(cross.K == K) & (cross.p == P)]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    rows = []
    for j, key in enumerate(CONFIG_ORDER):
        s = d[(d.method == key[0]) & (d.granularity == key[1])]
        D = s.pivot_table(index="target", columns="donor", values="accuracy_drop_pp", aggfunc="mean")
        D = D.reindex(index=TASKS, columns=TASKS).to_numpy()
        R = D - D.mean(1, keepdims=True) - D.mean(0, keepdims=True) + D.mean()
        vmax = np.abs(D).max()
        for i, (M, ttl, lim) in enumerate([(D, "Drop (pp)", (0, vmax)), (R, "Interaction residual", (-np.abs(R).max(), np.abs(R).max()))]):
            ax = axes[i, j]
            im = ax.imshow(M, cmap="Blues" if i == 0 else "RdBu_r", vmin=lim[0], vmax=lim[1])
            for a in range(6):
                for b in range(6):
                    ax.text(b, a, f"{M[a, b]:.0f}", ha="center", va="center", fontsize=7.5,
                            color="white" if abs(M[a, b]) > .6 * max(abs(lim[0]), lim[1]) else "0.15")
            ax.set_xticks(range(6)); ax.set_yticks(range(6))
            ax.set_xticklabels([_task_label(t) for t in TASKS], rotation=45, ha="right", fontsize=7.5)
            ax.set_yticklabels([_task_label(t) for t in TASKS], fontsize=7.5)
            ax.grid(False)
            ax.set_xlabel("Donor circuit"); ax.set_ylabel("Target task")
            _panel(ax, _label(key) if i == 0 else ttl)
            fig.colorbar(im, ax=ax, fraction=.046, pad=.03)
        diag = np.diag(R)
        rows.append(dict(config=_label(key), gap_pp=float(np.diag(D).mean() - D[~np.eye(6, dtype=bool)].mean()),
                         interaction_diag_mean=float(diag.mean()),
                         **{_task_label(t): float(v) for t, v in zip(TASKS, diag)}))
    fig.tight_layout()
    save(fig, "A6_interaction_residuals")
    note("A6", "Row-wise gap versus the diagonal of the two-way interaction residual (drop − row mean − column mean + grand mean; "
               "pp, mean over five models, K=10, P=50):\n\n" + md_table(pd.DataFrame(rows), "{:.1f}"))


# --- A7 chance-normalised necessity ----------------------------------------

def a7():
    e = extraction[(extraction.K == K) & (extraction.p == P)].copy()
    c = e.task.map(CHANCE)
    above = lambda acc: ((acc - c).clip(lower=0)) / (1 - c)  # share of the above-chance range; chance or worse is 0
    e["nec_paper"] = (e.control_accuracy - e.ablation_accuracy) / e.baseline_accuracy
    e["nec_diff"] = above(e.control_accuracy) - above(e.ablation_accuracy)
    base = above(e.baseline_accuracy)
    e["nec_rel"] = np.where(base >= .1, e.nec_diff / base, np.nan)
    panels = [("nec_paper", "Paper: [acc($C^3$) − acc($S_P$)] / acc(∅)", "paper"),
              ("nec_diff", "Above-chance: $a^+(C^3) − a^+(S_P)$", "above-chance diff"),
              ("nec_rel", "Above-chance, relative: $[a^+(C^3) − a^+(S_P)]\\,/\\,a^+(\\varnothing)$", "above-chance relative")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 3.8), sharey=True)
    x = np.arange(len(TASKS))
    w = .8 / len(CONFIG_ORDER)
    rows = []
    for ax, (col, ttl, short) in zip(axes, panels):
        for i, key in enumerate(CONFIG_ORDER):
            g = e[(e.method == key[0]) & (e.granularity == key[1])].groupby("task")[col]
            ax.bar(x + (i - 1.5) * w, g.mean().reindex(TASKS), w * .92, yerr=g.sem().reindex(TASKS),
                   error_kw=dict(lw=.6, capsize=1.5), label=_label(key), **_fill_kw(key))
            for t in TASKS:
                rows.append(dict(config=_label(key), task=_task_label(t), norm=short,
                                 necessity=g.mean().get(t, np.nan), n_cells=int(g.count().get(t, 0))))
        _panel(ax, ttl.replace("acc(∅)", "acc($\\varnothing$)"))
        ax.axhline(0, color="0.25", lw=.8)
        ax.axhline(1, color="0.5", lw=.6, ls=":")
        ax.set_xticks(x); ax.set_xticklabels([_task_label(t) for t in TASKS], rotation=15, ha="right")
    axes[0].set_ylabel("Necessity")
    h, l = axes[0].get_legend_handles_labels()
    _legend(fig, handles=h, labels=l, loc="lower center", ncol=4, bbox_to_anchor=(.5, -.14))
    save(fig, "A7_necessity_chance_normalised")
    t = pd.DataFrame(rows)
    piv = t.pivot_table(index=["task", "config"], columns="norm", values="necessity")
    piv = piv.reindex(columns=["paper", "above-chance diff", "above-chance relative"])
    cells = t[t.norm == "above-chance relative"].set_index(["task", "config"]).n_cells
    piv["n_cells (relative)"] = cells
    note("A7", "a⁺(S) = max(acc(S) − chance, 0) / (1 − chance): the share of the above-chance range the model retains under ablation S, "
               "with chance or worse binned to 0. Chance levels: Addition 0, Boolean 0.5, IOI 0.5, MCQA/ARC 0.25. The relative version "
               "divides by a⁺(∅) and drops cells where a⁺(∅) < 0.1 (`n_cells` counts survivors out of 5 models).\n\n"
               + md_table(piv.reset_index(), "{:.2f}"))


# --- A8 OLMo step-0 floor --------------------------------------------------

def olmo_runs():
    rows = {}
    for path in glob.glob(str(OLMO / "*" / "*" / "metrics.json")):
        d = json.loads(Path(path).read_text())
        rev = d.get("hf_revision") or "final"
        if rev.startswith("stage2"):
            continue
        tokens = int(re.search(r"tokens(\d+)B", rev).group(1)) if "tokens" in rev else 4001
        rows[(tokens, d["task"])] = d
    return rows


def a8():
    runs = olmo_runs()
    tokens = sorted({t for t, _ in runs})
    tasks = [t for t in TASKS if any((tok, t) in runs for tok in tokens)]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    rows = []
    for k, ls in [(10, "-")]:
        for t in tasks:
            xs, reuse, rel, jac, nmlp = [], [], [], [], []
            cell0 = runs[(0, t)]["by_k"][str(k)]["thresholds"]["95"]
            s0 = {parse_component(c) for c in cell0["shared_components"]}
            for tok in tokens:
                if (tok, t) not in runs:
                    continue
                cell = runs[(tok, t)]["by_k"][str(k)]["thresholds"]["95"]
                s = {parse_component(c) for c in cell["shared_components"]}
                xs.append(tok); reuse.append(cell["reuse_percent"]); rel.append(cell["reuse_percent"] - cell0["reuse_percent"])
                jac.append(jaccard_sets(s, s0)); nmlp.append(kind_counts(s)["mlp"])
                if k == 10:
                    rows.append(dict(task=_task_label(t), tokens_B=tok, reuse95=cell["reuse_percent"],
                                     size=len(s), mlp=kind_counts(s)["mlp"], heads=kind_counts(s)["head"],
                                     jaccard_with_step0=jaccard_sets(s, s0),
                                     baseline_acc=runs[(tok, t)]["baseline_train_accuracy"]))
            xp = np.array(xs, dtype=float); xp[xp == 0] = 1
            for ax, ys in zip(axes, [reuse, rel, jac, nmlp]):
                ax.plot(xp, ys, ls, marker="o", ms=3, color=TASK_COLORS[t], lw=1.3,
                        label=_task_label(t) if k == 10 else None)
    for ax, yl in zip(axes, ["Reuse@95 (%)", "Reuse@95 minus step-0 value (pp)",
                             "Jaccard($S_{95}$ at step, $S_{95}$ at step 0)", "MLP blocks in $S_{95}$ (of 16)"]):
        ax.set_xscale("log"); ax.set_xlabel("Pretraining tokens (B)"); ax.set_ylabel(yl)
    axes[1].axhline(0, color="0.25", lw=.8)
    _legend(axes[0], loc="lower left", fontsize=7.5)
    save(fig, "A8_olmo_step0_floor")
    t = pd.DataFrame(rows).sort_values(["task", "tokens_B"])
    t = t[t.tokens_B.isin([0, 3, 17, 76, 399, 1196, 2391, 4001])]
    note("A8", "OLMo-2 1B stage-1 checkpoints, EAP, K=10, P=95 (the stage-2 51B checkpoint is excluded). At step 0 the shared set is MLP blocks 0-14 for every task.\n\n"
               + md_table(t, "{:.2f}"))


# --- A9 out-degree null -----------------------------------------------------

def a9(seed=0):
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    rows = []
    x = np.arange(len(MODELS))
    for off, key in zip([-.2, .2], HEAD_MLP):
        rho_m, jac_m, jac_typed_m, chance_m, chance_typed_m = [], [], [], [], []
        for m in MODELS:
            rhos, jacs, jacs_typed, chances, chances_typed = [], [], [], [], []
            for t in TASKS:
                comps, S = scores(m, t, key[0])
                d = out_degree(comps, m)
                kinds = np.array([c[1] for c in comps])
                rhos.append(np.mean([stats.spearmanr(S[i], d).statistic for i in range(0, len(S), max(1, len(S) // 100))]))
                sp = shared(topk_sets(S, K))
                n = int(sp.sum())
                top_d = np.zeros(len(comps), dtype=bool)
                top_d[np.argsort(-d, kind="stable")[:n]] = True
                top_typed = np.zeros(len(comps), dtype=bool)
                draws = np.zeros((200, len(comps)), dtype=bool)
                for kind in ("mlp", "head"):
                    idx = np.where(kinds == kind)[0]
                    n_kind = int((sp & (kinds == kind)).sum())
                    top_typed[idx[np.argsort(-d[idx], kind="stable")[:n_kind]]] = True
                    for r in range(200):
                        draws[r, rng.choice(idx, n_kind, replace=False)] = True
                jacs.append(jaccard(sp, top_d) if n else np.nan)
                jacs_typed.append(jaccard(sp, top_typed) if n else np.nan)
                chances.append(n / (2 * len(comps) - n) if n else np.nan)
                chances_typed.append(np.mean([jaccard(sp, dr) for dr in draws]) if n else np.nan)
                rows.append(dict(config=_label(key), model=_model_label(m), task=_task_label(t),
                                 spearman_score_vs_outdegree=rhos[-1], jaccard_SP_vs_top_outdegree=jacs[-1],
                                 jaccard_SP_vs_top_outdegree_typed=jacs_typed[-1], chance_jaccard=chances[-1],
                                 chance_jaccard_typed=chances_typed[-1], size=n))
            rho_m.append(np.nanmean(rhos)); jac_m.append(np.nanmean(jacs)); jac_typed_m.append(np.nanmean(jacs_typed))
            chance_m.append(np.nanmean(chances)); chance_typed_m.append(np.nanmean(chances_typed))
        axes[0].bar(x + off, rho_m, .38, label=_label(key), **_fill_kw(key))
        axes[1].bar(x + off, jac_m, .38, label=_label(key), **_fill_kw(key))
        axes[2].bar(x + off, jac_typed_m, .38, label=_label(key), **_fill_kw(key))
        axes[1].scatter(x + off, chance_m, marker="_", color="0.2", s=120, zorder=3, label="chance" if off > 0 else None)
        axes[2].scatter(x + off, chance_typed_m, marker="_", color="0.2", s=120, zorder=3)
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels([_model_label(m) for m in MODELS], rotation=15, ha="right")
        ax.axhline(0, color="0.25", lw=.8)
    axes[0].set_ylabel("Spearman(score, out-degree)")
    axes[1].set_ylabel("Jaccard($S_{50}$, top-$|S_{50}|$ by out-degree)")
    axes[2].set_ylabel("Jaccard($S_{50}$, earliest nodes of each type)")
    axes[2].set_ylim(0, 1.02)
    _panel(axes[0], "Score vs out-degree"); _panel(axes[1], "Circuit vs out-degree ranking")
    _panel(axes[2], "Circuit vs type-matched out-degree ranking")
    _legend(axes[1], loc="upper right")
    save(fig, "A9_outdegree_null")
    t = pd.DataFrame(rows).groupby("config")[["spearman_score_vs_outdegree", "jaccard_SP_vs_top_outdegree",
                                              "jaccard_SP_vs_top_outdegree_typed", "chance_jaccard", "chance_jaccard_typed"]].mean().reset_index()
    note("A9", "Ranking by outgoing-edge count alone is task-independent, so its reuse is 100 and its cross-task Jaccard is 1 "
               "by construction; the informative comparison is how much of the real circuit it recovers. The type-matched version takes the earliest "
               "|S_50 ∩ MLP| MLP blocks and the earliest |S_50 ∩ heads| heads; its chance level is a random draw with the same type composition "
               "(mean over models and tasks, K=10, P=50):\n\n"
               + md_table(t, "{:.2f}"))


# --- A10 permutation nulls -------------------------------------------------

def a10(n_perm=300, seed=0):
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), gridspec_kw=dict(width_ratios=[1.1, 1], wspace=.45))
    rows = []
    x = np.arange(len(MODELS))
    for off, key in zip([-.2, .2], HEAD_MLP):
        for xi, m in zip(x, MODELS):
            mats, labels = [], []
            for ti, t in enumerate(TASKS):
                _, S = scores(m, t, key[0])
                mats.append(topk_sets(S, K)); labels += [ti] * len(S)
            M = np.vstack(mats); labels = np.array(labels)

            def pair_jaccard(lab):
                sets = [shared(M[lab == ti]) for ti in range(6)]
                return np.mean([jaccard(a, b) for a, b in itertools.combinations(sets, 2)])
            obs = pair_jaccard(labels)
            null = np.array([pair_jaccard(rng.permutation(labels)) for _ in range(n_perm)])
            axes[0].scatter(xi + off + rng.uniform(-.06, .06, n_perm), null, s=4, color="0.6", alpha=.3)
            axes[0].scatter(xi + off, obs, color=_color(key), s=48, edgecolor="0.2", zorder=3,
                            label=_label(key) if xi == 0 else None)
            rows.append(dict(test="cross-task Jaccard of S_50", config=_label(key), model=_model_label(m),
                             observed=obs, null_mean=null.mean(), null_lo=np.percentile(null, 2.5),
                             null_hi=np.percentile(null, 97.5), p_value=float((null <= obs).mean())))
    axes[0].set_xticks(x); axes[0].set_xticklabels([_model_label(m) for m in MODELS], rotation=15, ha="right")
    axes[0].set_ylabel("Mean pairwise Jaccard of $S_{50}$ across tasks")
    _panel(axes[0], "Cross-task overlap of $S_{50}$")
    _legend(axes[0], loc="lower right")

    d = cross[(cross.K == K) & (cross.p == P)]
    ys = np.arange(len(CONFIG_ORDER))[::-1]
    for y, key in zip(ys, CONFIG_ORDER):
        s = d[(d.method == key[0]) & (d.granularity == key[1])]
        Ds = [s[s.model == m].pivot_table(index="target", columns="donor", values="accuracy_drop_pp")
              .reindex(index=TASKS, columns=TASKS).to_numpy() for m in MODELS]
        def gap(perms):
            g = []
            for D, p in zip(Ds, perms):
                Dp = D[:, p]
                g.append(np.diag(Dp).mean() - Dp[~np.eye(6, dtype=bool)].mean())
            return np.mean(g)
        obs = gap([np.arange(6)] * 5)
        null = np.array([gap([rng.permutation(6) for _ in MODELS]) for _ in range(5000)])
        axes[1].scatter(null, y + rng.uniform(-.25, .25, len(null)), s=3, color="0.6", alpha=.15)
        axes[1].scatter(obs, y, color=_color(key), s=50, edgecolor="0.2", zorder=3)
        rows.append(dict(test="specificity gap (pp)", config=_label(key), model="pooled", observed=obs,
                         null_mean=null.mean(), null_lo=np.percentile(null, 2.5), null_hi=np.percentile(null, 97.5),
                         p_value=float((null >= obs).mean())))
    axes[1].set_yticks(ys); axes[1].set_yticklabels([_label(k) for k in CONFIG_ORDER])
    axes[1].set_xlabel("Specificity gap (pp), mean over five models")
    axes[1].axvline(0, color="0.25", lw=.8)
    _panel(axes[1], "Specificity gap")
    save(fig, "A10_permutation_nulls")
    note("A10", f"Left: per-example top-10% circuits of all six tasks pooled, task labels permuted {n_perm} times, S_50 "
                "recomputed per pseudo-task. Right: within each model the donor columns of the 6×6 drop matrix are permuted "
                "(5000 draws); p is the fraction of draws with a gap at least as large.\n\n" + md_table(pd.DataFrame(rows), "{:.3f}"))


# --- A11 EAP vs EAP-IG top-K Jaccard ---------------------------------------

def a11():
    names = {"eap": "EAP", "eap_ig": "EAP-IG", "relp": "RelP"}
    rows = []
    for a, b in [("eap", "eap_ig"), ("relp", "eap_ig"), ("relp", "eap")]:
        for m in MODELS:
            for t in TASKS:
                comps_a, Sa = scores(m, t, a)
                comps_b, Sb = scores(m, t, b)
                assert comps_a == comps_b
                rho = np.mean([stats.spearmanr(x, y).statistic for x, y in zip(Sa[::20], Sb[::20])])
                for k in KS:
                    Ma, Mb = topk_sets(Sa, k), topk_sets(Sb, k)
                    rows.append(dict(pair=f"{names[a]} vs {names[b]}", model=_model_label(m), task=t, K=k,
                                     per_example_jaccard=np.mean([jaccard(x, y) for x, y in zip(Ma, Mb)]),
                                     shared_circuit_jaccard=jaccard(shared(Ma), shared(Mb)), spearman_all_nodes=rho))
    t = pd.DataFrame(rows)
    by_k = t.groupby(["pair", "K"])[["per_example_jaccard", "shared_circuit_jaccard"]].mean().reset_index().astype({"K": int})
    by_k = by_k.pivot(index="K", columns="pair", values=["per_example_jaccard", "shared_circuit_jaccard"])
    by_k.columns = [f"{'per-example' if a == 'per_example_jaccard' else 'S_50'}: {b}" for a, b in by_k.columns]
    by_model = t[t.K == K].groupby(["pair", "model"])[["spearman_all_nodes", "per_example_jaccard", "shared_circuit_jaccard"]].mean().reset_index()
    by_model = by_model.rename(columns={"spearman_all_nodes": "Spearman (all nodes)", "per_example_jaccard": "per-example Jaccard",
                                        "shared_circuit_jaccard": "S_50 Jaccard"})
    note("A11", "Top-K Jaccard between the component sets two attribution methods select on the same model and task, "
                "mean over five models and six tasks, by K:\n\n" + md_table(by_k.reset_index(), "{:.2f}")
                + "\n\nAt K=10, per model. The Spearman column is the rank correlation of the two methods' scores over all nodes, "
                "mean over examples:\n\n" + md_table(by_model, "{:.2f}"))


# --- A12 cross-fit reuse ---------------------------------------------------

def a12(n_splits=20, seed=0):
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    rows = []
    for ax, key in zip(axes, HEAD_MLP):
        for m in MODELS:
            ins, outs, paper = [], [], []
            for k in KS:
                a, b, c = [], [], []
                for t in TASKS:
                    _, S = scores(m, t, key[0])
                    M = topk_sets(S, k)
                    size = M[0].sum()
                    for _ in range(n_splits):
                        idx = rng.permutation(len(M)); half = len(M) // 2
                        A, B = M[idx[:half]], M[idx[half:]]
                        sp = shared(A)
                        a.append((A & sp).sum(1).mean() / size)
                        b.append((B & sp).sum(1).mean() / size)
                    c.append(min(shared(M).sum(), size) / size)
                ins.append(np.mean(a)); outs.append(np.mean(b)); paper.append(np.mean(c))
                rows.append(dict(config=_label(key), model=_model_label(m), K=k, reuse_paper=paper[-1] * 100,
                                 coverage_in_sample=ins[-1] * 100, coverage_cross_fit=outs[-1] * 100))
            ax.plot(KS, np.array(paper) * 100, "-", color=MODEL_COLOR[m], lw=1, alpha=.5)
            ax.plot(KS, np.array(ins) * 100, "--", color=MODEL_COLOR[m], lw=1)
            ax.plot(KS, np.array(outs) * 100, "-o", ms=3.5, color=MODEL_COLOR[m], label=_model_label(m))
        ax.set_xticks(KS); ax.set_xlabel("K (%)"); ax.set_ylim(0, 102)
        _panel(ax, _label(key))
    axes[0].set_ylabel("Reuse@50 (%)")
    _legend(axes[1], loc="lower right", fontsize=7.5)
    save(fig, "A12_crossfit_reuse")
    t = pd.DataFrame(rows).groupby(["config", "K"])[["reuse_paper", "coverage_in_sample", "coverage_cross_fit"]].mean().reset_index()
    note("A12", f"S_50 fitted on a random half of the extraction examples, coverage |S_50 ∩ C_i|/|C_i| measured on the other half "
                f"({n_splits} splits; mean over models and tasks):\n\n" + md_table(t, "{:.1f}"))


# --- A13 within vs between-task similarity ---------------------------------

def a13(n_pairs=3000, seed=0):
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    rows = []
    x = np.arange(len(MODELS))
    for off, key in zip([-.2, .2], HEAD_MLP):
        accs = []
        for xi, m in zip(x, MODELS):
            mats, labels = [], []
            for ti, t in enumerate(TASKS):
                _, S = scores(m, t, key[0])
                mats.append(topk_sets(S, K)); labels += [ti] * len(S)
            M = np.vstack(mats).astype(np.float32); labels = np.array(labels)
            inter = M @ M.T
            sizes = M.sum(1)
            J = inter / (sizes[:, None] + sizes[None, :] - inter)
            i, j = rng.integers(0, len(M), (2, n_pairs * 4))
            keep = i != j
            i, j = i[keep], j[keep]
            same = labels[i] == labels[j]
            within, between = J[i[same], j[same]][:n_pairs], J[i[~same], j[~same]][:n_pairs]
            for vals, c, o in [(within, _color(key), -.08), (between, OTHER_COLOR, .08)]:
                parts = axes[0].violinplot([vals], [xi + off + o], widths=.16, showmedians=True, showextrema=False)
                for b in parts["bodies"]:
                    b.set_facecolor(c); b.set_alpha(.75)
                parts["cmedians"].set_color("0.2")
            # leave-one-out nearest-centroid classifier on membership vectors
            cents = np.stack([M[labels == ti].mean(0) for ti in range(6)])
            counts = np.bincount(labels, minlength=6)
            correct = 0
            for n in range(len(M)):
                c = cents.copy()
                c[labels[n]] = (c[labels[n]] * counts[labels[n]] - M[n]) / max(1, counts[labels[n]] - 1)
                sim = (c @ M[n]) / (np.linalg.norm(c, axis=1) * np.linalg.norm(M[n]) + 1e-9)
                correct += int(sim.argmax() == labels[n])
            acc = correct / len(M)
            accs.append(acc)
            rows.append(dict(config=_label(key), model=_model_label(m), within_median=float(np.median(within)),
                             between_median=float(np.median(between)),
                             auc=float(stats.mannwhitneyu(within, between).statistic / (len(within) * len(between))),
                             classifier_acc=acc, majority_baseline=counts.max() / len(M)))
        axes[1].bar(x + off, accs, .38, label=_label(key), **_fill_kw(key))
    axes[0].set_xticks(x); axes[0].set_xticklabels([_model_label(m) for m in MODELS], rotation=15, ha="right")
    axes[0].set_ylabel("Jaccard between per-example circuits")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, alpha=.75, label=l)
               for c, l in [(_color(HEAD_MLP[0]), "Within task, EAP-IG"), (_color(HEAD_MLP[1]), "Within task, RelP"),
                            (OTHER_COLOR, "Between tasks")]]
    _legend(axes[0], handles=handles, loc="upper right")
    _panel(axes[0], "Pairwise similarity of per-example circuits")
    axes[1].set_xticks(x); axes[1].set_xticklabels([_model_label(m) for m in MODELS], rotation=15, ha="right")
    axes[1].set_ylabel("Leave-one-out task classification accuracy")
    axes[1].axhline(1 / 6, color="0.3", lw=.8, ls="--")
    axes[1].set_ylim(0, 1.05)
    _panel(axes[1], "Nearest-centroid classifier on membership vectors")
    _legend(axes[1], loc="lower right")
    save(fig, "A13_within_between_jaccard")
    note("A13", "Per-example top-10% circuits, component level. `auc` is P(within-task Jaccard > between-task Jaccard). "
                "The classifier is leave-one-out nearest-centroid (cosine) on the 0/1 membership vector; chance is 1/6, "
                "`majority_baseline` is the largest class share (IOI, Boolean and Addition have 800 examples, ARC-C 469, MCQA 40).\n\n"
                + md_table(pd.DataFrame(rows), "{:.3f}"))


# --- A14 Boolean corruption audit -----------------------------------------

def a14():
    from circuit_reuse.dataset import BooleanDataset
    random.seed(42)
    ds = list(BooleanDataset(num_examples=1000))
    n_lit = np.array([len(re.findall(r"\b(true|false)\b", e.prompt)) for e in ds])
    same = np.array([e.target == e.corrupted_target for e in ds])
    fig, ax = plt.subplots(figsize=(6, 3.4))
    lits = sorted(set(n_lit))
    frac = [same[n_lit == n].mean() for n in lits]
    cnt = [(n_lit == n).sum() for n in lits]
    ax.bar(lits, frac, color=SECONDARY, edgecolor="0.3", lw=.5)
    for n, f, c in zip(lits, frac, cnt):
        ax.text(n, f + .02, f"n={c}", ha="center", fontsize=8)
    ax.axhline(same.mean(), color="0.25", lw=1, ls="--", label=f"all examples: {same.mean():.2f}")
    ax.set_xlabel("Literals in the expression"); ax.set_ylabel("Fraction of pairs with unchanged answer")
    ax.set_ylim(0, 1)
    _legend(ax, loc="upper left")
    save(fig, "A14_boolean_flip_audit")
    tgt = Counter(e.target for e in ds)
    note("A14", f"Regenerated with seed 42, n=1000 (the extraction set is the first 800 after the seeded shuffle). "
                f"{same.mean():.1%} of clean/corrupt pairs share the answer; the extractor does not filter them. "
                f"Clean answers: true {tgt['true']}, false {tgt['false']}.")


# --- A15 overlap vs task relatedness ---------------------------------------

def task_vocab():
    from circuit_reuse.dataset import get_dataset
    vocab = {}
    for t in TASKS:
        random.seed(42)
        ds = list(get_dataset(t, num_examples=1000, digits=3 if t == "addition" else 0))
        vocab[t] = Counter(w for e in ds for w in re.findall(r"[A-Za-z]+|\d|[^\sA-Za-z\d]", e.prompt.lower()))
    return vocab


def a15():
    vocab = task_vocab()

    def weighted_jaccard(a, b):
        keys = set(a) | set(b)
        return sum(min(a[k], b[k]) for k in keys) / sum(max(a[k], b[k]) for k in keys)
    rel = {frozenset((a, b)): weighted_jaccard(vocab[a], vocab[b]) for a, b in itertools.combinations(TASKS, 2)}
    ov = overlap[overlap.model.isin(MODELS)].copy()
    ov["vocab_overlap"] = [rel[frozenset((r.task_a, r.task_b))] for r in ov.itertuples()]
    ov["same_cf"] = [frozenset((r.task_a, r.task_b)) in SAME_COUNTERFACTUAL for r in ov.itertuples()]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharex=True)
    rows = []
    for ax, key in zip(axes, CONFIG_ORDER):
        s = ov[(ov.method == key[0]) & (ov.granularity == key[1])]
        for m in MODELS:
            sm = s[s.model == m]
            ax.scatter(sm.vocab_overlap.clip(lower=2e-3), sm.observed, color=MODEL_COLOR[m], s=[46 if c else 18 for c in sm.same_cf],
                       marker="o", edgecolor="0.3", lw=.4, alpha=.85, label=_model_label(m))
        rho = stats.spearmanr(s.vocab_overlap, s.observed)
        rows.append(dict(config=_label(key), spearman=rho.statistic, p=rho.pvalue,
                         mean_overlap_same_cf=s[s.same_cf].observed.mean(),
                         mean_overlap_other=s[~s.same_cf].observed.mean()))
        ax.text(.98, .96, f"$\\rho$ = {rho.statistic:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=8.5)
        ax.set_xscale("log")
        ax.set_xlabel("Prompt vocabulary overlap")
        _panel(ax, _label(key))
    axes[0].set_ylabel("Jaccard of $S_{50}$ across tasks")
    _legend(axes[0], loc="center left", fontsize=7.5)
    save(fig, "A15_overlap_vs_vocab")
    note("A15", "Larger markers are the three pairs that share the answer-position counterfactual (ARC-E, ARC-C, CopyColors).\n\n"
                + md_table(pd.DataFrame(rows), "{:.3f}"))


# --- A16 IOI counterfactual audit ------------------------------------------

def a16():
    from datasets import load_dataset
    from transformers import AutoTokenizer
    ds = load_dataset("mib-bench/ioi", split="test")
    n = len(ds)
    cf = [it["s2_io_flip_counterfactual"] for it in ds]
    same_answer = sum(c["choices"][c["answerKey"]] == it["choices"][it["answerKey"]] for it, c in zip(ds, cf))
    order = Counter()
    for it in ds:
        io, sub = it["metadata"]["indirect_object"], it["metadata"]["subject"]
        mentions = [m for m in re.findall(r"\b[A-Z][a-z]+\b", it["prompt"]) if m in (io, sub)]
        order[("IO first" if mentions[0] == io else "S first") + ", " + ("IO last" if mentions[-1] == io else "S last")] += 1
    rows = [dict(check="clean/corrupt pairs with the same answer", value=f"{same_answer}/{n}"),
            dict(check="answerKey always 0 (choices ordered [IO, S])", value=f"{sum(it['answerKey'] == 0 for it in ds)}/{n}"),
            dict(check="name order in the prompt", value="; ".join(f"{k}: {v}" for k, v in order.items())),
            dict(check="templates / unique prompts / unique names", value=f"{len(set(it['template'] for it in ds))} / {len(set(it['prompt'] for it in ds))} / {len(set(it['metadata']['indirect_object'] for it in ds))}")]
    for name in ["google/gemma-2-2b", "meta-llama/Llama-3.2-3B", "Qwen/Qwen3-4B"]:
        tok = AutoTokenizer.from_pretrained(name)
        diff = sum(len(tok(it["prompt"].rstrip() + " ")["input_ids"]) != len(tok(c["prompt"].rstrip() + " ")["input_ids"]) for it, c in zip(ds, cf))
        multi = sum(len(tok(" " + it["metadata"]["indirect_object"], add_special_tokens=False)["input_ids"]) > 1
                    or len(tok(" " + it["metadata"]["subject"], add_special_tokens=False)["input_ids"]) > 1 for it in ds)
        rows.append(dict(check=f"{name}: token-length mismatch / multi-token names", value=f"{diff}/{n} / {multi}/{n}"))
    note("A16", md_table(pd.DataFrame(rows)))


CAPTIONS = {
    "A1": "Accuracy drop (top) and relative drop (bottom) on a task when its own S_50 is ablated, against the mean drop when the other five "
          "tasks' circuits are ablated, as a function of K. Thin lines are single models (mean over six tasks), thick lines the mean over models. "
          "The number in parentheses under each K is the mean |S_50| at that K.",
    "A2": "Each point is one ordered task pair (A, B) in one model: the relative accuracy drop on task A when S^B is ablated, against the "
          "Jaccard overlap of S^A and S^B (top) and the size of S^B as a fraction of the node pool (bottom, log scale). The Spearman ρ in each panel is over all points.",
    "A3": "Each point is one task pair in one model. The x-axis is the smallest intersection two sets of those sizes can have, summed over "
          "component types (max(0, |A|+|B|−N) for MLP blocks plus the same for heads). The dashed line is y = x.",
    "A4": "log2 of the MLP share of the circuit divided by the MLP share of the node pool, mean over tasks. Left: the shared circuit S_50; "
          "right: the per-example top-10% circuits. Zero means no enrichment.",
    "A5": "Top: Figure 4 of the paper coloured by task; each point is one model. The dashed line is y = x. Bottom: the specificity gap "
          "(own drop minus mean other-task drop) per task, mean over five models, error bars are the s.e.m.",
    "A6": "Top: the 6×6 drop matrix (target task by donor circuit, pp, mean over five models). Bottom: the same matrix after removing row "
          "means and column means and adding back the grand mean. The diagonal of the bottom row is the part of the own-task drop not "
          "explained by how fragile the target is or how damaging the donor is in general.",
    "A7": "Three versions of necessity at K=10, P=50. Left: the paper's Eq. 5. Middle: accuracies are first mapped to the share of the "
          "above-chance range they retain, a⁺(S) = max(acc(S) − chance, 0) / (1 − chance), so chance-level or worse is 0 and a perfect "
          "model is 1; necessity is a⁺(C³) − a⁺(S_50). Right: the same difference divided by the clean model's a⁺(∅), so 1 means the "
          "ablation removes everything the model had above chance; cells with a⁺(∅) < 0.1 are dropped. Bars are means over five models "
          "with s.e.m.; the dotted line is 1.",
    "A8": "OLMo-2 1B stage-1 checkpoints, EAP, K=10, P=95. Step 0 is drawn at 1 on the log axis. From left: Reuse@95; Reuse@95 minus its "
          "step-0 value; Jaccard between S_95 at the checkpoint and S_95 at step 0; number of MLP blocks in S_95 (of 16).",
    "A9": "Left: Spearman correlation between a node's attribution score and its number of outgoing edges, mean over examples. Middle: Jaccard "
          "between S_50 and the |S_50| nodes with the most outgoing edges. Right: Jaccard between S_50 and the earliest |S_50 ∩ MLP| MLP blocks "
          "plus the earliest |S_50 ∩ heads| heads. Black ticks are the chance level of each comparison (random sets of the same size, and of the same type composition on the right).",
    "A10": "Grey points are null draws, coloured points the observed value. Left, per model and method: every example's top-10% "
           "circuit from all six tasks is pooled (3,709 membership vectors, each tagged with its task); the observed value builds S_50 per "
           "task and averages the Jaccard over the 15 task pairs; each null draw reassigns the task tags at random (keeping the number of "
           "examples per task), rebuilds the six S_50 sets and averages their pairwise Jaccard; 300 draws. Right, per configuration: D is "
           "the 6×6 matrix of accuracy drops (target task by donor circuit) for one model, and the gap is mean(diagonal) − "
           "mean(off-diagonal), averaged over the five models; each null draw permutes the donor columns of every model's D independently "
           "and recomputes the gap; 5,000 draws.",
    "A11": "",
    "A12": "Thin lines: the paper's Reuse@50 = |S_50| / |C_i|. Dashed: coverage |S_50 ∩ C_i| / |C_i| on the examples S_50 was fitted on. "
           "Solid with markers: the same coverage on the held-out half (20 random splits). One colour per model, mean over tasks.",
    "A13": "Left: distributions of Jaccard between pairs of per-example top-10% circuits from the same task (blue tones, one per method) and "
           "from different tasks (green); black ticks are medians. Right: leave-one-out accuracy of a nearest-centroid classifier that "
           "predicts the task from the 0/1 membership vector; the dashed line is chance (1/6).",
    "A14": "Fraction of Boolean clean/corrupt pairs whose answer does not change when one literal is flipped, by number of literals in the "
           "expression (n is the number of expressions in each bin). The dashed line is the fraction over all 1000 examples.",
    "A16": "",
    "A15": "Each point is one task pair in one model: Jaccard of the two tasks' S_50 against the overlap of their prompt vocabularies "
           "(weighted Jaccard over word and character types, log scale, zero drawn at 0.002). Larger markers are the three pairs that share the "
           "answer-position counterfactual. ρ is the Spearman correlation over all points in the panel.",
}

# Items whose numeric tables add something the figure does not show.
TABLES = {"A1", "A2", "A5", "A6", "A7", "A9", "A10", "A11", "A12", "A14", "A15", "A16"}

INTERPRETATIONS = {
    "A1": "The component-level gap is a small-K effect. Under EAP-IG it is 6.7 pp at K=1 (own 38 pp, other 31 pp) and gone by K=5, "
          "where both own and other circuits remove about 50 pp; under RelP it falls from 15.7 pp at K=1 to 3.3 pp at K=30. The neuron-level "
          "gap is flat at 20-24 pp from 174 to 15,000 neurons. So the headline K=10 sits on the saturated part of the component curve, as the "
          "review says, and the paper should report the curve rather than one K. The curves also give the matched-damage comparison the "
          "review asks for (Sec. 2.5): at an own-task drop near 23 pp, component circuits (RelP, K=1) cost 9 pp on other tasks and neuron "
          "circuits (EAP-IG, K=1) cost 3 pp; at an own-task drop near 45 pp, component circuits (EAP-IG, K=5) cost 48 pp and neuron circuits "
          "(EAP-IG, K=30) cost 24 pp. Neuron circuits do about half the collateral damage at equal own-task damage. What the curves cannot "
          "show is anything past the accuracy floor; both component curves flatten at 50 pp because the model is at zero or chance.",
    "A2": "For EAP-IG component circuits the review is right: cross-task damage is unrelated to overlap (Spearman -0.11) and to donor size "
          "(0.12), and the regression explains nothing (R² ≤ 0.02). Every donor circuit in this configuration holds 20-28 MLP blocks, so "
          "ablating any of them is close to the same intervention, and the sentence 'this difference is explained by overlap' has no support "
          "here. For the other three configurations overlap does predict damage (Spearman 0.27-0.49) and its coefficient survives controlling "
          "for donor size and model (t = 3.7-6.6). Donor size matters on its own for EAP-IG neurons (t = 5.0). The mediation claim should be "
          "restricted to RelP and neuron-level circuits, and the EAP-IG component result described as generic fragility.",
    "A3": "For Gemma the review's arithmetic holds: S_50 is 20-24 of the 26 MLP blocks and no heads, the mean pairwise intersection is 20.5 "
          "against a combinatorial floor of 18.3, and every pair is within 1-3 components of the floor. For Llama and Qwen the observed "
          "intersections are about twice the floor (37 vs 19 and 66 vs 29), and much of the shared set is heads (31-43 of Llama's 672), so the "
          "overlap there is not forced. RelP circuits have a floor of zero and one third of Gemma pairs sit on it. The G.4 figures of 83% and "
          "88% are therefore a counting artefact for Gemma and a finding for the other three models; the text should say so. On the control "
          "in Eq. (5): with 20-28 MLP blocks in S_50, V \\ S_50 has 0-6 MLP blocks left, and the code samples min(n, |pool|) of each type, so "
          "the control for these cells is 14-28 components smaller than S_50 (for Gemma it is 2-6 random MLP blocks and no heads). The "
          "necessity values in Figure 2 for Gemma and for Llama Addition compare S_50 against a control that is not capacity-matched, and the "
          "paper must say this; the parity-sampled control in results/c3_parity is the alternative.",
    "A4": "Relative to base rate MLP blocks are 8-11x over-represented in EAP-IG component circuits in every model (log2 enrichment "
          "3.1-3.4 for S_50, 3.1-3.2 for per-example circuits) and 3-12x in RelP circuits. Gemma is the extreme case: 95-100% MLP from an 11% "
          "base rate. The sentence 'circuits are majority attention heads' is true of raw counts for Llama and Qwen and false as a description "
          "of the selection, and the tension with the introduction disappears once enrichment is reported. The figure replaces the composition "
          "bars in Section 4.",
    "A5": "The decomposition confirms the review's table. Under EAP-IG neurons Addition carries 62% of the summed gap and CopyColors 29%; "
          "ARC-E, ARC-C, IOI and Boolean together carry 9%. Under RelP neurons the effect is broader (Addition 59%, ARC-E 21%, ARC-C 16%) but "
          "IOI and Boolean still contribute nothing. In the scatter the Addition points sit far above the diagonal in the neuron panels, the "
          "ARC points a little above it, and IOI and Boolean on it with own-task drops near zero, so for those two tasks the neuron circuits are "
          "not necessary and specificity is undefined rather than absent. The conclusion should say that neuron circuits are specific for "
          "Addition and CopyColors, weakly for ARC, and untestable for IOI and Boolean, and Figure 4 should be the coloured version.",
    "A6": "Removing row and column means lowers every gap but keeps the ordering: the interaction diagonal is 1.0 / 5.8 / 18.4 / 18.3 pp "
          "against the row-wise 1.3 / 8.3 / 22.1 / 22.0 pp. So about 4 pp of the neuron-level gap is the row-mean effect the review describes "
          "and 18 pp is not. Per task the neuron interaction is 62 pp for Addition, 25 pp for CopyColors (EAP-IG), 6-13 pp for ARC and Boolean "
          "and -2 to 5 pp for IOI. Boolean's residual is positive only because its column mean is near zero (empty donor circuits), which is "
          "the inflation the review worries about acting in the other direction. Reporting both numbers is the honest fix; the interaction "
          "version should go in the appendix next to Table 6.",
    "A7": "Binning at chance removes the arithmetic the review objects to: under Eq. 5 a 4-way task can never show more than 0.75 and a "
          "60%-accurate model no more than 0.58, so the left panel reads Addition (1.0) against ARC (0.5-0.7) as a difference in "
          "necessity when much of it is a difference in chance level. In the middle panel the numbers are in one unit (share of the "
          "above-chance range lost), and the right panel puts every task on the same 0-to-1 scale. On that scale component-level EAP-IG "
          "circuits remove 0.7-0.9 of what the model had on the MCQA tasks, neuron-level circuits 0.35-0.85, and both are 1.0 on "
          "Addition. RelP component circuits stay at 0.1-0.2 because their capacity-matched control already takes most of the margin. "
          "Boolean keeps 4 of 5 models in the relative panel (Gemma 2 2B sits below chance) and gives about 0.27 for EAP-IG components and "
          "zero otherwise; in the middle panel the near-chance cells simply contribute 0 rather than being dropped. IOI stays at 0.05-0.3 "
          "everywhere. The ordering across configurations does not change, so the paper's conclusions hold, but the cross-task "
          "comparisons in the text should use the right-hand version.",
    "A8": "The conjecture is right: at step 0 the shared set is MLP blocks 0-14 for all six tasks, which is 15 of 27 nodes and the 55.6% in "
          "Table 22. Reuse stays on that floor to 32-76B tokens and the shared set keeps a Jaccard above 0.85 with the step-0 set until 53B. "
          "After 400B tokens reuse is 15-48% for five tasks and zero for Boolean, and the shared set retains 2-11 of the step-0 MLP blocks. "
          "Reuse minus the floor is negative at every checkpoint past 76B. Consistency does not emerge during training; the untrained "
          "network gives the maximum this metric can show, and training moves circuits away from the architectural default. Section J should be "
          "rewritten around this floor, and the same step-0 comparison is what the review asks for on the five main models (needs GPU). "
          "This is EAP with P=95, the configuration of the original OLMo runs.",
    "A9": "The depth prior is in the scores but not in the circuits. EAP-IG node scores correlate with out-degree at Spearman 0.44-0.61; "
          "RelP scores do not. But the |S_50| nodes with the most outgoing edges (layer-0 heads) overlap S_50 at chance (0.03 vs 0.05), and "
          "the earliest MLP blocks and heads, matched to the type composition of S_50, overlap it at chance for Llama and Qwen (0.20-0.25 vs "
          "0.21-0.23). Only Gemma is above chance (0.92 vs 0.73-0.79), and only because S_50 contains almost every MLP block. So the "
          "contiguous early block in Appendix I is not evidence of an edge-count prior in general, and the review's prediction that ranking by "
          "out-degree reproduces the circuits fails. results/pat_confound_checks already shows that dividing scores by out-degree moves the "
          "early-layer share from 0.43 to 0.21 without changing reuse or overlap; the two results together answer Sec. 2.3's depth-prior "
          "point without a GPU. The signed-versus-absolute question (cancellation between edges) still needs the node-level EAP-IG run.",
    "A10": "Both panels ask whether a headline number could have arisen with no task structure. Left: if the task tags carried no "
           "information, each pseudo-task would be a random mixture of all six real tasks, its S_50 would be close to the model-wide "
           "consensus set, and the six S_50 sets would nearly coincide. That is what the null shows: Jaccard 0.90-1.0 for EAP-IG and "
           "0.41-0.89 for RelP. The observed overlap (0.46-0.88 EAP-IG, 0.16-0.29 RelP) is below the null in every model, p < 0.004. Real "
           "tasks therefore produce more distinct shared circuits than arbitrary groupings of the same examples, so the reported overlap is "
           "not the maximum the data allow. This does not say how much of the observed overlap is forced by counting; that is A3. Right: "
           "under the null 'own' is a random one of the six circuits, so the gap is zero in expectation and its spread measures how much "
           "damage varies across donors. The 95% range is ±2 pp for EAP-IG components, ±6 pp for RelP components and ±8 pp for neurons. "
           "The EAP-IG component gap of 1.3 pp is inside its null (p = 0.09), the RelP component gap of 8.3 pp is outside (p = 0.004), and "
           "both neuron gaps of 22 pp are far outside (p < 0.001). The paper should state the 1.3 pp figure as indistinguishable from zero "
           "under permutation. The left panel uses the per-example scores in cache/ and so exists only for component circuits; the right "
           "uses the ablation results and covers all four configurations.",
    "A11": "Top-10% membership agrees between EAP and EAP-IG at Jaccard 0.85 per example and 0.92 for S_50, against 0.66 and 0.73 at "
           "K=1, with an all-node Spearman of 0.98. So substituting EAP for EAP-IG in Figure 5 and Section J is defensible at K=10 and "
           "weaker at K=1, and footnote 1 should give the Jaccard numbers instead of the Spearman. RelP against either EAP method is a "
           "different picture: Spearman near zero, per-example Jaccard 0.15-0.16 and S_50 Jaccard about 0.10. The two families select "
           "nearly disjoint component sets, which restates the sign-rule finding in results/pat_confound_checks and means the "
           "EAP-IG-versus-RelP comparisons in Section 6.2 compare two different selections, not two estimates of one circuit.",
    "A12": "Cross-fit coverage is within 1 pp of in-sample coverage at every K and for both methods, so S_50 is not overfit to the "
           "examples it was built from and the in-sample objection has no numerical consequence. The reported statistic |S_50| / |C_i| is "
           "9-12 pp above coverage because S_50 contains components that are absent from some examples' circuits. Coverage on held-out "
           "examples is the cleaner definition of consistency and should replace the ratio in Eq. (2); its values are 81% (EAP-IG) and 35% "
           "(RelP) at K=10.",
    "A13": "Per-example component circuits carry task identity. Within-task Jaccard exceeds between-task Jaccard in every model (AUC "
           "0.70-0.93 for EAP-IG, 0.78-0.90 for RelP) and a nearest-centroid classifier recovers the task from the membership vector at "
           "66-87% against 17% chance and 22% majority. Gemma EAP-IG is weakest (AUC 0.70, 66%) for the pigeonhole reason. This is the unified "
           "measurement the review asks for, and it says something the ablation results do not: component circuits are consistent and also "
           "measurably task-specific in membership, even though zero-ablation accuracy cannot register the difference. The specificity "
           "failure at the component level is partly a failure of the accuracy metric, which supports moving to graded metrics.",
    "A14": "The review is right and the damage is larger than it guessed. Flipping one literal leaves the truth value unchanged in 79% of "
           "pairs, rising from 71% with three literals to 84% with six. Only one pair in five is a real counterfactual; for the rest the "
           "attribution metric compares two prompts with the same answer, so the scores measure sensitivity to one token rather than to the "
           "computation. Clean answers are 59% true. Combined with clean accuracies of 48-73% on a binary task, Boolean cannot support any "
           "claim in the paper. Drop it from the main results, or fix the corruption (flip until the value changes) and re-extract, which "
           "needs GPU.",
    "A16": "The same audit as A14, run on IOI because the two tasks share the templated construction. IOI passes. The s2-IO-flip "
           "counterfactual changes the answer in 1000 of 1000 pairs (the clean answer is the indirect object, the corrupt answer is the "
           "subject), every name is a single token under all three tokenizers, and clean and corrupt prompts have identical token length in "
           "every case, so the position-by-position interpolation in EAP-IG is aligned. The structural regularity the review suspects is "
           "real, though: in all 1000 prompts the subject is mentioned first and last and the indirect object second, so 'pick the name "
           "that is not the most recent mention' scores 100%. That is the standard IOI construction and it is what the known IOI circuit "
           "(S-inhibition and name-mover heads) computes, so it does not invalidate the task, but it means the task can be solved by "
           "attention alone. That is the most likely reason Gemma's IOI accuracy survives zeroing 23 of 26 MLP blocks (Sec. 2.8), and it is "
           "consistent with the neuron-level IOI circuits being unnecessary (A5). Whether the hook is also at fault cannot be checked "
           "without a GPU. The MCQA alignment question in the same review paragraph is a separate check on the ARC and CopyColors "
           "corruptions and is not done here.",
    "A15": "Prompt vocabulary overlap is near zero for every pair except ARC-Easy / ARC-Challenge (0.56), so a lexical relatedness measure "
           "has one informative pair and the Spearman correlations (0.18, 0.09, -0.04, 0.07) are not significant. The same-counterfactual "
           "pairs have higher circuit overlap on average, but that is entirely ARC-E / ARC-C: neuron circuits overlap at 0.41-0.48 for that "
           "pair and at 0.03-0.09 for CopyColors against either ARC set, which shares the counterfactual but not the content. This supports "
           "the review's hypothesis in Sec. 2.6 that neuron circuits track the input distribution rather than the answer-binding mechanism. "
           "Component-level overlap is high for all three MCQA pairs (0.76-0.94 EAP-IG), so it does not separate the two. Testing the "
           "hypothesis properly needs the format-by-content design, which is GPU work.",
}

ITEMS = {
    "A1": ("Dose-response curves", a1, ["A1_dose_response.png"],
           "I would like to see: Dose-response curves (own-task vs. other-task damage as a function of ablated set size)."),
    "A2": ("Regression of cross-task damage on overlap", a2, ["A2_drop_vs_overlap.png"],
           "A regression of Delta^B_A on J(S^A, S^B) controlling for |S^B|, before any causal language about overlap is used. "
           "[...] Overlap is therefore not necessary for cross-task damage. The more parsimonious explanation is generic fragility to zero-ablating a handful of MLP blocks."),
    "A3": ("Pigeonhole check: observed overlap vs combinatorial minimum", a3, ["A3_pigeonhole.png"],
           "Two sets of sizes 24 and 23 drawn from 26 items must intersect in at least 24 + 23 - 26 = 21 elements. Insofar as these are all MLP blocks, "
           "the observed overlap equals the combinatorial minimum. [...] The control in Eq. (5) cannot be constructed as described. [...] For Gemma, S_P contains about 23 MLP blocks and V \\ S_P contains about 3."),
    "A4": ("MLP enrichment over base rate", a4, ["A4_mlp_enrichment.png"],
           "l. 295 says \"For nearly all models, circuits are majority attention heads\". This ignores base rates. Heads are about 96% of nodes in Llama, "
           "so a circuit that is 41% MLP represents roughly tenfold enrichment of MLPs. Report enrichment, not raw fractions."),
    "A5": ("Per-task decomposition of the specificity gap", a5, ["A5a_own_vs_other_by_task.png", "A5b_gap_by_task.png"],
           "Show the per-task decomposition in the main text. The scatter in Figure 4 should be colored by task, because pooled gaps conceal that two tasks carry the result. "
           "[...] Addition and CopyColors contribute about 91% of the effect."),
    "A6": ("Two-way interaction residuals of the drop matrix", a6, ["A6_interaction_residuals.png"],
           "Row-wise gap. The gap compares circuits of wildly different sizes. [...] Boolean's near-empty circuits cannot damage anything, which inflates everyone else's gap. "
           "Use a two-way interaction (remove row and column means), or size-matched controls per cell."),
    "A7": ("Necessity normalised by (accuracy − chance)", a7, ["A7_necessity_chance_normalised.png"],
           "Dividing by acc(M, empty) ignores chance. For four-way MCQA the maximum attainable necessity is about 0.75 (0.58 for ARC-C on Gemma), "
           "so values are not comparable across tasks. Normalize by (acc - chance)."),
    "A8": ("Reuse relative to the OLMo step-0 floor", a8, ["A8_olmo_step0_floor.png"],
           "Table 22 shows Reuse@95 at K=10% at the 0B checkpoint is 56, 56, 56, 56, 56, 59 across the six tasks. [...] Note that 56% = 15/27 [...] "
           "I would conjecture these are 15 of the 16 MLP blocks. [...] report all reuse and overlap numbers relative to that null."),
    "A9": ("Out-degree ranking null", a9, ["A9_outdegree_null.png"],
           "Null models: [...] ranking by outgoing-edge count or activation norm alone. [...] The number of outgoing edges from a node grows with the number of "
           "downstream nodes. Summing absolute values means noise does not cancel, so early-layer nodes receive a large task-independent bonus."),
    "A10": ("Task-label permutation nulls", a10, ["A10_permutation_nulls.png"],
            "Null models: [...] task-label permutation of per-example circuits to obtain null distributions for Jaccard and the specificity gap."),
    "A11": ("Top-K agreement between EAP, EAP-IG and RelP", a11, [],
            "Footnote 1. A Spearman of 0.977 over all components says little about top-K membership. Report top-K Jaccard."),
    "A12": ("Cross-fit reuse", a12, ["A12_crossfit_reuse.png"],
            "Reuse is in-sample. S_P is computed from the same examples as Eq. (2). Cross-fit it."),
    "A13": ("Within- vs between-task circuit similarity; task classifier", a13, ["A13_within_between_jaccard.png"],
            "A unified alternative: the distribution of pairwise Jaccard between per-example circuits within vs. between tasks, "
            "or simply the accuracy of a classifier predicting task from the circuit membership vector."),
    "A14": ("Boolean corruption audit", a14, ["A14_boolean_flip_audit.png"],
            "The Boolean flip claim is false. [...] \"true and (false or true)\" with false -> true still evaluates to true. "
            "Unless you filter, a good share of clean/corrupt pairs share an answer and the attribution signal is noise."),
    "A16": ("IOI counterfactual audit", a16, [],
            "Positional alignment in the MCQA corruptions. Permuting answer choices of unequal token length misaligns positions between clean "
            "and corrupt inputs [...] Either the IOI evaluation is insensitive (a pairwise candidate comparison plus a template position bias would do it), "
            "or something is wrong with the hook."),
    "A15": ("Circuit overlap vs task relatedness", a15, ["A15_overlap_vs_vocab.png"],
            "The genuinely interesting claim is that component overlap is flat with respect to relatedness [...] correlate overlap with a behavioral or "
            "representational task-similarity measure. [...] Plot pairwise circuit similarity against prompt token overlap."),
}


def write_md(items):
    lines = ["# Review plots (no GPU)", "",
             "Source: `analysis/review_plots.py`. Figures in `paper2/review_plots/`. Quotes are from `claude_review.md`.",
             "All at K=10%, P=50% unless stated. Models: the five main ones. Tasks: the six main ones.", ""]
    for item in items:
        title, _, figs, quote = ITEMS[item]
        lines += [f"## {item}. {title}", "", f"> {quote}", ""]
        lines += [f"![{f}](review_plots/{f})" for f in figs] + ([f"", f"*{CAPTIONS[item]}*"] if CAPTIONS[item] else []) + [""]
        if item in TABLES:
            lines += [n + "\n" for n in NOTES.get(item, [])]
        lines += [INTERPRETATIONS[item], ""]
    (REPO / "paper2" / "review_plots.md").write_text("\n".join(lines))
    print("wrote paper2/review_plots.md")


def main():
    items = sys.argv[1:] or list(ITEMS)
    for item in items:
        print(item, ITEMS[item][0])
        ITEMS[item][1]()
    if not sys.argv[1:]:
        write_md(items)
    else:
        for item in items:
            for n in NOTES.get(item, []):
                print(n)


if __name__ == "__main__":
    main()
