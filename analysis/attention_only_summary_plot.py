"""Per-model bar-plot grid of head+MLP vs heads-only cross-task overlap.

One subplot per model. Y is Overlap Coefficient. Error bars are 95% CIs
across the 15 unordered task pairs for that (model, K, granularity).
"""

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
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

REPO = Path("/Users/michaelli/Desktop/Research/circuit_reuse")
SRC = REPO / "results2" / "attention_only_overlap" / "pairs_P85_eap.csv"
OUT = REPO / "results2" / "attention_only_overlap" / "per_model_bars_P85_eap.png"

COLOR_ALL = "#E76F51"
COLOR_HEADS = "#5DA9C9"
COLOR_CHANCE = "#000000"


def ci95(arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return float("nan")
    return stats.t.ppf(0.975, len(arr) - 1) * arr.std(ddof=1) / np.sqrt(len(arr))


def aggregate(df, model, K, granularity):
    sub = df[(df["model"] == model) & (df["K"] == K) & (df["granularity"] == granularity)]
    vals = sub["jaccard"].values
    return (float(np.nanmean(vals)) if len(vals) else float("nan"),
            ci95(vals))


def main():
    df = pd.read_csv(SRC)
    df = df[df["K"] != 1]
    models = list(df["model"].unique())
    ks = sorted(df["K"].unique())

    n = len(models)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.4 * nrows),
                             sharey=True)
    axes = np.atleast_2d(axes).reshape(nrows, ncols)

    for idx, model in enumerate(models):
        r, c = idx // ncols, idx % ncols
        ax = axes[r, c]

        all_mean, all_err = [], []
        heads_mean, heads_err = [], []
        for K in ks:
            m_a, e_a = aggregate(df, model, K, "head+MLP")
            m_h, e_h = aggregate(df, model, K, "heads only")
            all_mean.append(m_a); all_err.append(e_a)
            heads_mean.append(m_h); heads_err.append(e_h)

        def clipped_yerr(means, errs):
            means = np.asarray(means, dtype=float)
            errs = np.asarray(errs, dtype=float)
            lower = np.clip(errs, 0, means)
            upper = np.clip(errs, 0, 1.0 - means)
            return np.vstack([lower, upper])

        x = np.arange(len(ks))
        w = 0.38
        b1 = ax.bar(
            x - w / 2, all_mean, w, yerr=clipped_yerr(all_mean, all_err), capsize=3,
            color=COLOR_ALL, edgecolor="white", linewidth=0.5,
            label="head + MLP (paper)",
            error_kw={"elinewidth": 1.0, "ecolor": "#444"},
            zorder=2,
        )
        b2 = ax.bar(
            x + w / 2, heads_mean, w, yerr=clipped_yerr(heads_mean, heads_err), capsize=3,
            color=COLOR_HEADS, edgecolor="white", linewidth=0.5,
            label="heads only",
            error_kw={"elinewidth": 1.0, "ecolor": "#444"},
            zorder=2,
        )

        for bar, v in zip(b1, all_mean):
            if np.isnan(v):
                continue
            ax.text(bar.get_x() + bar.get_width() / 2, v / 2,
                    f"{v:.2f}", ha="center", va="center",
                    color="white", fontsize=9, fontweight="bold")
        for bar, v in zip(b2, heads_mean):
            if np.isnan(v):
                continue
            ax.text(bar.get_x() + bar.get_width() / 2, v / 2,
                    f"{v:.2f}", ha="center", va="center",
                    color="white", fontsize=9, fontweight="bold")

        ks_frac = np.array(ks, dtype=float) / 100.0
        chance = ks_frac / (2.0 - ks_frac)
        ax.plot(
            x, chance, linestyle="--", color=COLOR_CHANCE, linewidth=1.6,
            marker="o", markersize=5, markeredgecolor="white", markeredgewidth=0.7,
            label="random baseline $K/(2{-}K)$", zorder=6,
        )

        ax.set_xticks(x)
        ax.set_xticklabels([f"{k}%" for k in ks])
        ax.set_ylim(0, 1.05)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.set_title(model, fontsize=12)
        ax.grid(True, axis="y", alpha=0.25, linestyle=":")
        ax.set_axisbelow(True)
        if c == 0:
            ax.set_ylabel("Cross-task circuit overlap")
        if r == nrows - 1:
            ax.set_xlabel("Circuit size $K$")

    for idx in range(n, nrows * ncols):
        r, c = idx // ncols, idx % ncols
        axes[r, c].axis("off")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", bbox_to_anchor=(0.5, -0.08),
        ncol=3, fontsize=12,
    )

    fig.suptitle(
        "Per-model cross-task circuit overlap with vs. without MLP layers "
        "(EAP, $P=85\\%$, error bars: 95% CI across task pairs)",
        fontsize=13, y=1.00,
    )
    fig.tight_layout()
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
