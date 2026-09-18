"""Teaser figure: consistency and specificity reverse between the two granularities.

The contribution is a reversal, so the figure shows both halves of it at once,
and it shows them per model. Averaging the five models would assert a
population-level claim the paper does not make, and would hide that the neuron
specificity gap ranges from 15 to 30 points across them.

Every number is read from the analysis outputs at the operating point rather
than typed in, so the figure cannot drift from the table.
"""
from __future__ import annotations

import argparse
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
                                         GRAN_SHORT, _fill_kw, _label, _legend,
                                         _model_label)


def _numbers(analysis: Path, K: int, p: int):
    at = lambda d: d[(d.K == K) & (d.p == p)]
    within = at(pd.read_csv(analysis / "within_task_summary.csv"))
    cross = at(pd.read_csv(analysis / "cross_task_summary.csv"))
    return within, cross, sorted(within.model.unique())


def _panel(ax, frame, column, models, ylabel):
    """One group per model, one bar per configuration within it."""
    x = np.arange(len(models))
    width = .8 / len(CONFIG_ORDER)
    for i, key in enumerate(CONFIG_ORDER):
        sub = frame[(frame.method == key[0]) & (frame.granularity == key[1])]
        vals = [sub[sub.model == m][column].mean() for m in models]
        ax.bar(x + i * width - .4 + width / 2, vals, width * .92,
               label=_label(key), **_fill_kw(key))
    ax.axhline(0, c="0.3", lw=.8)
    ax.set_xticks(x, [_model_label(m) for m in models], rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=.25, linewidth=.6, axis="y")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def make(analysis: Path, out: Path, K: int, p: int):
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                         "axes.labelsize": 10, "xtick.labelsize": 8.5,
                         "ytick.labelsize": 9, "legend.fontsize": 8.5})
    within, cross, models = _numbers(analysis, K, p)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8))
    _panel(axes[0], within, "reuse", models, "Reuse@$P$ (%)")
    _panel(axes[1], cross, "specificity_gap_pp", models, "Specificity gap (pp)")
    # Below the axes rather than inside them: at five models the bars reach the
    # top of the left panel and any in-axes legend sits on data.
    handles, labels = axes[0].get_legend_handles_labels()
    _legend(fig, handles=handles, labels=labels, ncol=4, loc="lower center",
            bbox_to_anchor=(.5, -.22))

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", default="results/granularity_parity_analysis")
    ap.add_argument("--out", default="results/granularity_parity_analysis/figure1_granularity.png")
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--p", type=int, default=50)
    args = ap.parse_args()
    make(Path(args.analysis_dir), Path(args.out), args.K, args.p)


if __name__ == "__main__":
    main()
