"""Zero vs mean ablation: does the ablation choice change any finding?

Loads the K=10, P=100 cross-task matrices produced under zero ablation
(results/cross_task_ablation_k10) and mean ablation
(results/cross_task_ablation_mean_k10), builds a tidy cell-level table, and
writes plots and summary statistics to results2/zero_vs_mean_ablation.
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": .25,
    "grid.linewidth": .6,
    "axes.axisbelow": True,
})

REPO = Path(__file__).resolve().parent.parent
ZERO_DIR = REPO / "results" / "cross_task_ablation_k10"
MEAN_DIR = REPO / "results" / "cross_task_ablation_mean_k10"
OUT = REPO / "results2" / "zero_vs_mean_ablation"
OUT.mkdir(parents=True, exist_ok=True)

TASKS = ["addition", "boolean", "ioi", "mcqa", "arc_easy", "arc_challenge"]
TASK_LABEL = {"addition": "Addition", "boolean": "Boolean", "ioi": "IOI",
              "mcqa": "MCQA", "arc_easy": "ARC-E", "arc_challenge": "ARC-C"}
# Zissou1 / Darjeeling1 / Rushmore1
TASK_COLORS = ["#F21A00", "#3B9AB2", "#E1AF00", "#00A08A", "#F98400", "#35274A"]
# one hue per family, base model dark and instruct/larger model light
MODEL_COLORS = ["#9A4F0A", "#E5A650",   # Gemma, FantasticFox1 orange
                "#1F6E8C", "#78B7C5",   # Llama, Zissou1 blue
                "#35274A", "#9986A5"]   # Qwen, Rushmore1 / IsleofDogs1 purple
MODELS = {
    "google/gemma-2-2b": "Gemma-2-2B",
    "google/gemma-2-2b-it": "Gemma-2-2B-IT",
    "meta-llama/Llama-3.2-3B": "Llama-3.2-3B",
    "meta-llama/Llama-3.2-3B-Instruct": "Llama-3.2-3B-IT",
    "qwen3-4b": "Qwen3-4B",
    "qwen3-8b": "Qwen3-8B",
}
# Wes Anderson palettes: Zissou1 for zero/mean and other, GrandBudapest1 for own
C_ZERO, C_MEAN = "#3B9AB2", "#E1AF00"
C_OWN, C_OTHER = "#FD6467", "#78B7C5"
TITLE = "$K$=10%, $P$=100%"
GAP_THRESHOLD = 5


def save(fig, name):
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def load_cells():
    """Both JSONs index accuracy_drop_pp[source][target]; own and other drops
    are grouped by target, as in granularity_parity.py."""
    rows = []
    for model in MODELS:
        stem = f"cross_task_{model.replace('/', '_')}_K10_t100"
        zero = json.load(open(ZERO_DIR / f"{stem}.json"))
        mean = json.load(open(MEAN_DIR / f"{stem}_meanabl.json"))
        for tgt in TASKS:
            for src in TASKS:
                rows.append(dict(
                    model=model, target=tgt, source=src, diag=tgt == src,
                    size=zero["circuit_sizes"][src],
                    base_zero=100 * zero["baseline_accuracy"][tgt],
                    base_mean=100 * mean["baseline_accuracy"][tgt],
                    zero=zero["accuracy_drop_pp"][src][tgt],
                    mean=mean["accuracy_drop_pp"][src][tgt],
                ))
    df = pd.DataFrame(rows)
    df["delta"] = df["mean"] - df["zero"]
    return df


def summarise(df):
    """Per (model, target): own drop, mean other drop, gap, under each ablation."""
    out = []
    for (model, tgt), g in df.groupby(["model", "target"], sort=False):
        own = g[g.diag].iloc[0]
        other = g[~g.diag & (g["size"] > 0)]
        row = dict(model=model, target=tgt, size=own["size"], base=own["base_zero"])
        for ab in ("zero", "mean"):
            row[f"own_{ab}"] = own[ab]
            row[f"other_{ab}"] = other[ab].mean()
            row[f"gap_{ab}"] = own[ab] - other[ab].mean()
        out.append(row)
    return pd.DataFrame(out)


def corr_block(x, y):
    d = y - x
    return dict(n=int(len(x)),
                pearson=float(stats.pearsonr(x, y)[0]),
                spearman=float(stats.spearmanr(x, y)[0]),
                mean_abs_diff=float(np.mean(np.abs(d))),
                mean_diff=float(np.mean(d)),
                median_diff=float(np.median(d)),
                n_nonzero_diff=int((d != 0).sum()),
                wilcoxon_p=float(stats.wilcoxon(x, y).pvalue) if np.any(d != 0) else 1.0)


# --- plots --------------------------------------------------------------

def plot_cell_scatter(df):
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    for model, col in zip(MODELS, MODEL_COLORS):
        g = df[df.model == model]
        ax.scatter(g.zero, g["mean"], s=40, color=col, edgecolor="black", linewidth=.4,
                   zorder=2, label=MODELS[model])
    lim = (-25, 105)
    ax.plot(lim, lim, color="grey", linewidth=.8, linestyle="--", zorder=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Accuracy drop, zero ablation (pp)")
    ax.set_ylabel("Accuracy drop, mean ablation (pp)")
    ax.set_title(TITLE)
    ax.legend(fontsize=8, loc="upper left", frameon=True, edgecolor="grey")
    save(fig, "cell_scatter_zero_vs_mean")


def plot_own_vs_other(summ):
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.3), sharex=True, sharey=True)
    lim = (-25, 105)
    for ax, ab, title in zip(axes, ("zero", "mean"), ("Zero ablation", "Mean ablation")):
        for task, col in zip(TASKS, TASK_COLORS):
            g = summ[summ.target == task]
            ax.scatter(g[f"other_{ab}"], g[f"own_{ab}"], s=50, color=col,
                       edgecolor="black", linewidth=.4, zorder=3, label=TASK_LABEL[task])
        ax.plot(lim, lim, color="grey", linewidth=.8, linestyle="--", zorder=1)
        for off in (GAP_THRESHOLD, -GAP_THRESHOLD):
            ax.plot(lim, [v + off for v in lim], color="grey", linewidth=.6,
                    linestyle=":", zorder=1)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("Mean drop from other circuits (pp)")
        ax.set_title(title)
    axes[0].set_ylabel("Drop from own circuit (pp)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(TASKS), fontsize=9,
               frameon=True, edgecolor="grey", bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save(fig, "own_vs_other_scatter")


def plot_gap_by_model(summ):
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 5.2), sharex=True)
    for ax, model in zip(axes.flat, MODELS):
        s = summ[summ.model == model].set_index("target").loc[TASKS]
        y = np.arange(len(TASKS))[::-1]
        ax.hlines(y, s.gap_zero, s.gap_mean, color="grey", linewidth=1.2, zorder=1)
        ax.scatter(s.gap_zero, y, color=C_ZERO, s=36, zorder=2, label="Zero ablation")
        ax.scatter(s.gap_mean, y, color=C_MEAN, s=36, zorder=2, label="Mean ablation")
        ax.axvline(0, color="black", linewidth=.7)
        ax.axvspan(-GAP_THRESHOLD, GAP_THRESHOLD, color="grey", alpha=.08, zorder=0)
        ax.set_yticks(y); ax.set_yticklabels([TASK_LABEL[t] for t in TASKS], fontsize=8)
        ax.set_title(MODELS[model])
        ax.set_xlim(-80, 45)
    for ax in axes[1]:
        ax.set_xlabel("Specificity gap (pp)")
    axes[0, 0].legend(fontsize=8, loc="lower left", frameon=True, edgecolor="grey")
    fig.tight_layout()
    save(fig, "specificity_gap_by_model")


def plot_necessity_bars(summ):
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 5.2), sharey=True)
    x = np.arange(len(TASKS)); w = .38
    for ax, model in zip(axes.flat, MODELS):
        s = summ[summ.model == model].set_index("target").loc[TASKS]
        ax.bar(x - w / 2, s.own_zero, w, color=C_ZERO, label="Zero ablation", zorder=2)
        ax.bar(x + w / 2, s.own_mean, w, color=C_MEAN, label="Mean ablation", zorder=2)
        ax.scatter(x, s.base, marker="_", s=180, color="black", linewidth=1.2,
                   label="Baseline accuracy", zorder=3)
        ax.axhline(0, color="black", linewidth=.7)
        ax.set_xticks(x); ax.set_xticklabels([TASK_LABEL[t] for t in TASKS],
                                             rotation=35, ha="right", fontsize=8)
        ax.set_title(MODELS[model])
        ax.set_ylim(-25, 105)
    for ax in axes[:, 0]:
        ax.set_ylabel("Own-circuit drop (pp)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=9, loc="lower center", ncol=3, frameon=True,
               edgecolor="grey", bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout()
    save(fig, "necessity_own_drop_by_model")


def plot_target_by_source(df, target, name):
    """Drop on one target task from each of the six circuits, zero vs mean, per model."""
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 5.2), sharey=True)
    x = np.arange(len(TASKS)); w = .38
    for ax, model in zip(axes.flat, MODELS):
        g = df[(df.model == model) & (df.target == target)].set_index("source").loc[TASKS]
        ax.bar(x - w / 2, g.zero, w, color=C_ZERO, label="Zero ablation", zorder=2)
        ax.bar(x + w / 2, g["mean"], w, color=C_MEAN, label="Mean ablation", zorder=2)
        ax.axhline(0, color="black", linewidth=.7)
        ax.set_xticks(x); ax.set_xticklabels([TASK_LABEL[t] for t in TASKS],
                                             rotation=35, ha="right", fontsize=8)
        ax.set_title(MODELS[model])
    lo, hi = df[df.target == target][["zero", "mean"]].to_numpy().min(), \
        df[df.target == target][["zero", "mean"]].to_numpy().max()
    axes[0, 0].set_ylim(min(lo, 0) - 5, hi + 5)
    for ax in axes[:, 0]:
        ax.set_ylabel(f"Drop on {TASK_LABEL[target]} (pp)")
    for ax in axes[1]:
        ax.set_xlabel("Ablated circuit")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=9, loc="lower center", ncol=2, frameon=True,
               edgecolor="grey", bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout()
    save(fig, name)


# --- statistics ---------------------------------------------------------

def compute_stats(df, summ, paired):
    s = {}
    s["cells"] = {
        "all": corr_block(df.zero.to_numpy(), df["mean"].to_numpy()),
        "own": corr_block(df[df.diag].zero.to_numpy(), df[df.diag]["mean"].to_numpy()),
        "other": corr_block(df[~df.diag].zero.to_numpy(), df[~df.diag]["mean"].to_numpy()),
    }
    s["per_target_paired_tests"] = paired
    s["baseline_max_abs_diff_pp"] = float((df.base_zero - df.base_mean).abs().max())
    s["gap"] = corr_block(summ.gap_zero.to_numpy(), summ.gap_mean.to_numpy())
    s["gap"][f"within_{GAP_THRESHOLD}pp"] = (
        f"{int(((summ.gap_zero - summ.gap_mean).abs() <= GAP_THRESHOLD).sum())}/{len(summ)}")
    for ab in ("zero", "mean"):
        s["gap"][f"gt{GAP_THRESHOLD}_{ab}"] = int((summ[f"gap_{ab}"] > GAP_THRESHOLD).sum())
        s["gap"][f"lt-{GAP_THRESHOLD}_{ab}"] = int((summ[f"gap_{ab}"] < -GAP_THRESHOLD).sum())
    both = (summ.gap_zero > GAP_THRESHOLD) & (summ.gap_mean > GAP_THRESHOLD)
    s["gap"][f"gt{GAP_THRESHOLD}_both"] = [f"{MODELS[r.model]} / {TASK_LABEL[r.target]}"
                                           for r in summ[both].itertuples()]
    for thr in (10, 20):
        s[f"own_drop_gt{thr}pp"] = {ab: f"{int((summ[f'own_{ab}'] > thr).sum())}/{len(summ)}"
                                    for ab in ("zero", "mean")}
    s["own_drop_le10pp_both"] = [f"{MODELS[r.model]} / {TASK_LABEL[r.target]}" for r in
                                 summ[(summ.own_zero <= 10) & (summ.own_mean <= 10)].itertuples()]
    rng = df.groupby(["model", "target"]).agg(zero=("zero", lambda v: v.max() - v.min()),
                                              mean=("mean", lambda v: v.max() - v.min()))
    s["rows_all_sources_within_5pp"] = {ab: f"{int((rng[ab] <= 5).sum())}/{len(rng)}"
                                        for ab in ("zero", "mean")}
    s["row_range_median_pp"] = {ab: float(rng[ab].median()) for ab in ("zero", "mean")}
    per_model = summ.groupby("model")[["own_zero", "own_mean", "other_zero", "other_mean",
                                       "gap_zero", "gap_mean"]].mean().round(1)
    s["per_model"] = per_model.to_dict("index")
    moves = summ[(summ.gap_zero - summ.gap_mean).abs() > GAP_THRESHOLD]
    s[f"gap_moves_gt{GAP_THRESHOLD}pp"] = [
        dict(model=r.model, target=r.target, gap_zero=round(r.gap_zero, 1),
             gap_mean=round(r.gap_mean, 1)) for r in moves.itertuples()]
    moves = summ[(summ.own_zero - summ.own_mean).abs() > GAP_THRESHOLD]
    s[f"own_moves_gt{GAP_THRESHOLD}pp"] = [
        dict(model=r.model, target=r.target, own_zero=round(r.own_zero, 1),
             own_mean=round(r.own_mean, 1)) for r in moves.itertuples()]
    return s


def main():
    df = load_cells()
    summ = summarise(df)
    df.to_csv(OUT / "cells_tidy.csv", index=False)
    summ.to_csv(OUT / "per_target_summary.csv", index=False)

    plot_cell_scatter(df)
    plot_own_vs_other(summ)
    plot_gap_by_model(summ)
    plot_necessity_bars(summ)
    plot_target_by_source(df, "ioi", "ioi_drop_by_source")
    paired = {m: corr_block(summ[f"{m}_zero"].to_numpy(), summ[f"{m}_mean"].to_numpy())
              for m in ("own", "other")}

    s = compute_stats(df, summ, paired)
    with open(OUT / "summary_stats.json", "w") as f:
        json.dump(s, f, indent=2)
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
