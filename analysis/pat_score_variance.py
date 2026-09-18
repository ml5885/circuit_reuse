"""Per-node attribution-score variance across examples.

Component level: from cache/*.jsonl (800 train examples per model x task).
Neuron level: no per-example scores are stored locally, so the only view of
cross-example variability is membership stability from metrics.json (how many
units are in the top-K% circuit of at least P% of examples).
"""

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.granularity_parity import CONFIG_COLORS, CONFIG_ORDER
from analysis.pat_confound_checks import MODELS, TASKS, cache_path, load_scores, topk_sets
from circuit_reuse.dataset import get_model_display_name, get_task_display_name

matplotlib.use("Agg")
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                     "axes.titlesize": 11, "axes.labelsize": 10,
                     "xtick.labelsize": 9, "ytick.labelsize": 9,
                     "legend.fontsize": 8.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": .25, "grid.linewidth": .6})

KIND_COLORS = {"head": "#5FA8D0", "mlp": "#E8B33C"}
METHOD_NAME = {"eap_ig": "EAP-IG", "relp": "RelP"}
K = 10


def per_node_stats(comps, S):
    M = topk_sets(S, K)
    return pd.DataFrame({
        "layer": [c[0] for c in comps], "kind": [c[1] for c in comps], "index": [c[2] for c in comps],
        "mean": S.mean(0), "std": S.std(0), "abs_mean": np.abs(S).mean(0),
        "q10": np.quantile(S, .1, axis=0), "q90": np.quantile(S, .9, axis=0),
        "freq": M.mean(0),
    })


def plot_node_variance(model, task, out):
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2))
    for row, method in enumerate(["eap_ig", "relp"]):
        comps, S = load_scores(cache_path(model, task, method))
        st = per_node_stats(comps, S)
        M = topk_sets(S, K)
        cut = np.sort(S, axis=1)[:, -M[0].sum()]  # per-example K% cutoff score

        # (a) mean vs std per node
        ax = axes[row, 0]
        for kind, c in KIND_COLORS.items():
            s = st[st.kind == kind]
            ax.scatter(s["mean"], s["std"], s=12 if kind == "head" else 30, alpha=.6, color=c, label=kind,
                       marker="o" if kind == "head" else "s", edgecolor="none")
        ax.axvline(cut.mean(), color="0.4", lw=.8, ls="--", label=f"mean K={K}% cutoff")
        if method == "eap_ig":
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.set_xlabel("mean score across examples")
        ax.set_ylabel("standard deviation across examples")
        ax.set_title(f"{METHOD_NAME[method]}: each node's mean and standard deviation")
        ax.legend(frameon=True, framealpha=.9, edgecolor="none", loc="best")

        # (b) score distributions for nodes ranked by mean: top 6, 6 at cutoff, 6 far below
        ax = axes[row, 1]
        order = np.argsort(-st["mean"].values)
        take = M[0].sum()
        picks = list(order[:6]) + list(order[take - 3:take + 3]) + list(order[-6:])
        groups = ["top 6"] * 6 + ["at cutoff"] * 6 + ["bottom 6"] * 6
        pos = np.arange(len(picks)) + np.repeat([0, 1, 2], 6)
        for p, i, g in zip(pos, picks, groups):
            vals = S[:, i]
            c = KIND_COLORS[comps[i][1]]
            ax.scatter(np.full(vals.shape, p) + np.random.default_rng(0).uniform(-.25, .25, vals.shape),
                       vals, s=2, alpha=.15, color=c, rasterized=True)
            ax.plot([p - .3, p + .3], [np.median(vals)] * 2, color="k", lw=1.2)
        ax.axhline(cut.mean(), color="0.4", lw=.8, ls="--")
        ax.set_xticks([2.5, 9.5, 16.5])
        ax.set_xticklabels(["top 6\nby mean", f"6 nodes at\nK={K}% cutoff", "bottom 6\nby mean"])
        if method == "eap_ig":
            ax.set_yscale("log")
        ax.set_ylabel("score (one dot per example)")
        ax.set_title(f"{METHOD_NAME[method]}: all 800 scores for 18 nodes")

        # (c) membership frequency curve
        ax = axes[row, 2]
        f = np.sort(st["freq"].values)[::-1]
        ax.plot(np.arange(1, len(f) + 1), f * 100, color=CONFIG_COLORS[(method, "head_mlp")], lw=1.4)
        ax.axvline(take, color="0.4", lw=.8, ls="--", label=f"$|C_i|$ = {take}")
        ax.axhline(50, color="0.6", lw=.6, ls=":")
        ax.set_xscale("log")
        ax.set_xlabel("nodes, sorted from most to least often selected")
        ax.set_ylabel(f"% of examples on which node is in top {K}%")
        ax.set_title(f"{METHOD_NAME[method]}: how often each node is selected")
        ax.legend(frameon=True, framealpha=.9, edgecolor="none", loc="lower left")
    fig.suptitle(f"{get_model_display_name(model)}, {get_task_display_name(task)}: how much each component's score varies across examples")
    fig.tight_layout()
    stem = f"{model.replace('/', '_')}__{task}"
    fig.savefig(out / f"node_variance__{stem}.png", dpi=200)
    plt.close(fig)


def variance_decomposition(out):
    """Fraction of score variance explained by node identity (eta^2) and per-node CV, all runs."""
    rows = []
    for model in MODELS:
        for task in TASKS:
            for method in ("eap_ig", "relp"):
                comps, S = load_scores(cache_path(model, task, method))
                st = per_node_stats(comps, S)
                eta2 = S.mean(0).var() / S.var()
                top = st.nlargest(int(len(comps) * K / 100), "mean")
                rows.append(dict(model=model, task=task, method=method, eta2=float(eta2),
                                 cv_top=float((top["std"] / top["abs_mean"]).median()),
                                 cv_all=float((st["std"] / st["abs_mean"]).median()),
                                 freq_top_median=float(top["freq"].median())))
                print(f"[var] {model} {task} {method} eta2={eta2:.2f}")
    df = pd.DataFrame(rows)
    df.to_csv(out / "component_variance_summary.csv", index=False)
    return df


def membership_stability(out):
    """Fraction of the per-example circuit that is in the top-K of >= P% of examples, all four configs."""
    df = pd.read_csv("results/granularity_parity_analysis/extraction_tidy.csv")
    pools = json.load(open("results/granularity_parity_analysis/component_pools.json"))
    df = df[df.model.isin(MODELS) & df.task.isin(TASKS) & (df.K == K)].copy()
    df["size"] = [max(1, int(pools[m][g] * K / 100)) for m, g in zip(df.model, df.granularity)]
    df["stable_frac"] = df.circuit_size / df["size"] * 100
    names = {("eap_ig", "head_mlp"): "EAP-IG component", ("relp", "head_mlp"): "RelP component",
             ("eap_ig", "neuron"): "EAP-IG neuron", ("relp", "neuron"): "RelP neuron"}
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for cfg in CONFIG_ORDER:
        s = df[(df.method == cfg[0]) & (df.granularity == cfg[1])].groupby("p").stable_frac
        ax.errorbar(s.mean().index, s.mean(), yerr=s.std(), marker="o", ms=4, capsize=2,
                    color=CONFIG_COLORS[cfg], label=names[cfg])
    ax.set_xlabel("P: required % of examples that select the unit")
    ax.set_ylabel(f"units selected on $\\geq$P% of examples,\nas % of circuit size")
    ax.set_ylim(0, 100)
    ax.set_title(f"How much of a K={K}% circuit recurs across examples\n(mean $\\pm$ s.d. over 30 model-task runs)")
    fig.legend(*ax.get_legend_handles_labels(), ncol=2, frameon=False, loc="lower center")
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(out / "membership_stability_vs_p.png", dpi=200)
    tab = df.groupby(["method", "granularity", "p"]).stable_frac.mean().unstack("p").round(1)
    tab.to_csv(out / "membership_stability_vs_p.csv")
    return tab


def score_tail(out):
    """How far into the score distribution the top-K% cutoff reaches, component level.

    Left: the sorted |score| profile of one example, divided by that example's top score,
    averaged over examples and tasks, one line per model. Right: cumulative share of
    total |score| held by the top x% of components. K = 1, 5, 10% marked."""
    rows, profiles, masses = [], {}, {}
    grid = np.logspace(-3, 0, 60)
    for model in MODELS:
        for method in ("eap_ig", "relp"):
            prof, mass = [], []
            for task in TASKS:
                comps, S = load_scores(cache_path(model, task, method))
                A = np.abs(S) if method == "relp" else S
                srt = np.sort(A, axis=1)[:, ::-1]
                n = srt.shape[1]
                frac = (np.arange(n) + 1) / n
                rel = srt / srt[:, :1]
                cum = np.cumsum(srt, axis=1) / srt.sum(1, keepdims=True)
                prof.append(np.array([np.interp(grid, frac, r) for r in rel]).mean(0))
                mass.append(np.array([np.interp(grid, frac, c) for c in cum]).mean(0))
                for k in (1, 5, 10):
                    m = max(1, int(n * k / 100))
                    rows.append(dict(model=model, task=task, method=method, K=k,
                                     mass_in_topk=float(cum[:, m - 1].mean()),
                                     cut_over_top=float(np.median(srt[:, m - 1] / srt[:, 0])),
                                     cut_over_median=float(np.median(srt[:, m - 1] / np.median(srt, axis=1)))))
            profiles[(model, method)] = np.mean(prof, 0)
            masses[(model, method)] = np.mean(mass, 0)
    df = pd.DataFrame(rows)
    df.to_csv(out / "score_tail_components.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    from analysis.granularity_parity import MODEL_COLORS
    for row, method in enumerate(["eap_ig", "relp"]):
        for ax, store, ylabel in ((axes[row, 0], profiles, "score at this rank / top score"),
                                  (axes[row, 1], masses, "share of total |score| in top x%")):
            for model, color in zip(MODELS, MODEL_COLORS):
                ax.plot(grid * 100, store[(model, method)], color=color, lw=1.3, label=get_model_display_name(model))
            for k in (1, 5, 10):
                ax.axvline(k, color="0.5", lw=.7, ls="--")
                ax.text(k, .02, f"K={k}%", fontsize=7.5, ha="left", va="bottom", color="0.3",
                        transform=ax.get_xaxis_transform())
            ax.set_xscale("log")
            ax.set_xlabel("rank as % of all components")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{METHOD_NAME[method]}" + (" (|score| for RelP)" if method == "relp" else ""))
        axes[row, 0].set_yscale("log")
        axes[row, 1].set_ylim(0, 1.02)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), ncol=5, frameon=False, loc="lower center")
    fig.suptitle("Where the top-K% cutoff falls in the component score distribution (mean over 6 tasks x 800 examples)")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out / "score_tail_components.png", dpi=200)
    tab = df.groupby(["method", "K"])[["mass_in_topk", "cut_over_top", "cut_over_median"]].mean().round(3)
    print(tab.to_string())
    return tab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/pat_confound_checks")
    ap.add_argument("--examples", nargs="*", default=["meta-llama/Llama-3.2-3B:ioi", "google/gemma-2-2b:addition",
                                                      "qwen3-4b:arc_easy"])
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    for spec in args.examples:
        model, task = spec.split(":")
        plot_node_variance(model, task, out)
    print(membership_stability(out).to_string())
    score_tail(out)
    df = variance_decomposition(out)
    print(df.groupby("method")[["eta2", "cv_top", "cv_all", "freq_top_median"]].agg(["mean", "min", "max"]).round(2).to_string())


if __name__ == "__main__":
    main()
