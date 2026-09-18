"""What shared circuits are made of, at both granularities.

Two structural questions that the ablation and overlap summaries leave open.
First, ablation-based specificity responds to the *fraction* of a task's circuit
that another task also uses, not to symmetric Jaccard overlap, so we compute that
fraction directly and relate it to the measured specificity gap.  Second, the
composition of a shared circuit -- how it splits between attention heads and MLP
layers, and where it sits in depth -- has only ever been reported over heads and
MLP layers, and the neuron granularity answers the same question.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import tempfile
from pathlib import Path

_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "circuit_reuse_mplconfig"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd

from analysis.circuit_overlap_chance import read_circuits
from analysis.granularity_parity import (CONFIG, METHOD_MARKER, METHOD_NAMES,
                                         SECONDARY, _color, _label, _legend,
                                         _line_kw, _ordered)

COMPONENT = re.compile(r"(\w+)\[layer=(\d+)")
DEPTH_BINS = 10


def parse(component: str) -> tuple[str, int]:
    kind, layer = COMPONENT.match(component).groups()
    return kind, int(layer)


def shared_fractions(circuits: pd.DataFrame) -> pd.DataFrame:
    """|C_A n C_B| / |C_A| for every ordered task pair, which is what ablating
    B's circuit removes from A's."""
    rows = []
    for (model, method, gran), group in circuits.groupby(["model"] + CONFIG):
        sets = {r.task: r.components for r in group.itertuples()}
        for a, b in itertools.permutations(sorted(sets), 2):
            if not sets[a]:
                continue
            rows.append({"model": model, "method": method, "granularity": gran,
                         "task": a, "other": b,
                         "shared_fraction": len(sets[a] & sets[b]) / len(sets[a])})
    return pd.DataFrame(rows)


def composition(circuits: pd.DataFrame) -> pd.DataFrame:
    """Component-type split and normalized depth profile of each shared circuit."""
    rows = []
    for r in circuits.itertuples():
        parsed = [parse(c) for c in r.components]
        if not parsed:
            continue
        layers = np.array([l for _, l in parsed])
        # Models differ in depth, so bin by relative depth to make the profiles
        # comparable across a 26-layer and a 36-layer model.
        depth = layers / max(layers.max(), 1)
        hist, _ = np.histogram(depth, bins=DEPTH_BINS, range=(0, 1))
        rows.append({"model": r.model, "task": r.task, "method": r.method,
                     "granularity": r.granularity, "size": len(parsed),
                     "mlp_fraction": np.mean([k == "mlp" for k, _ in parsed]),
                     "head_fraction": np.mean([k == "head" for k, _ in parsed]),
                     **{f"bin{i}": v / len(parsed) for i, v in enumerate(hist)}})
    return pd.DataFrame(rows)


def _setup():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                         "axes.titlesize": 11, "axes.labelsize": 10,
                         "xtick.labelsize": 9, "ytick.labelsize": 9,
                         "legend.fontsize": 9, "figure.titlesize": 12})
    return plt


def style(ax):
    ax.grid(alpha=.25, linewidth=.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_shared_fraction(fracs: pd.DataFrame, gaps: pd.DataFrame | None, out: Path):
    """The bridge from overlap to specificity: what fraction of a task's circuit
    a foreign ablation takes away, and what that costs in specificity."""
    plt = _setup()
    configs = _ordered(sorted(fracs.groupby(CONFIG).groups))
    colors = [_color(k) for k in configs]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    ax = axes[0]
    data = [fracs[(fracs.method == k[0]) & (fracs.granularity == k[1])].shared_fraction.values
            for k in configs]
    parts = ax.violinplot(data, showextrema=False, widths=.8)
    for body, c in zip(parts["bodies"], colors):
        body.set_facecolor(c)
        body.set_edgecolor(c)
        body.set_alpha(.35)
    for i, (vals, key, c) in enumerate(zip(data, configs, colors), start=1):
        ax.scatter(np.full(len(vals), i) + (np.arange(len(vals)) % 7 - 3) * .022, vals,
                   s=7, color=c, alpha=.45, linewidth=0, zorder=3)
        ax.scatter([i], [vals.mean()], s=90, color=c, edgecolor="white", linewidth=1.2,
                   marker=METHOD_MARKER.get(key[0], "o"), zorder=4)
        ax.annotate(f"{vals.mean():.2f}", (i, vals.mean()), textcoords="offset points",
                    xytext=(11, -3), fontsize=9, fontweight="bold")
    ax.set_xticks(range(1, len(configs) + 1), [_label(k) for k in configs],
                  rotation=12, ha="right")
    ax.set(ylabel="Fraction of a task's circuit that\nanother task also uses",
           ylim=(-.03, 1.03))
    style(ax)

    ax = axes[1]
    if gaps is not None and not gaps.empty:
        joined = (fracs.groupby(CONFIG + ["model"]).shared_fraction.mean()
                  .to_frame().join(gaps).dropna().reset_index())
        for key in configs:
            g = joined[(joined.method == key[0]) & (joined.granularity == key[1])]
            ax.scatter(g.shared_fraction, g.specificity_gap_pp, s=64, alpha=.85,
                       color=_color(key), marker=METHOD_MARKER.get(key[0], "o"),
                       label=_label(key), edgecolor="white", linewidth=.8)
        ax.axhline(0, ls="--", c="grey", lw=1, zorder=0)
        ax.set(xlabel="Mean shared fraction", ylabel="Specificity gap (pp)")
        _legend(ax)
        style(ax)
    fig.tight_layout()
    fig.savefig(out / "shared_fraction_by_granularity.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_composition(comp: pd.DataFrame, out: Path):
    """What the shared circuit is built from, and where in depth it sits."""
    plt = _setup()
    configs = _ordered(sorted(comp.groupby(CONFIG).groups))
    bins = [f"bin{i}" for i in range(DEPTH_BINS)]
    centers = (np.arange(DEPTH_BINS) + .5) / DEPTH_BINS
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4),
                             gridspec_kw={"width_ratios": [.7, 1]})

    # Composition is only a question at the head/MLP granularity; the neuron
    # basis has one component type by construction.
    ax = axes[0]
    hm = [k for k in configs if k[1] == "head_mlp"]
    x = np.arange(len(hm))
    mlp = [comp[(comp.method == k[0]) & (comp.granularity == k[1])].mlp_fraction.mean()
           for k in hm]
    ax.bar(x, mlp, .5, color=[_color(k) for k in hm], label="MLP layers")
    ax.bar(x, [1 - v for v in mlp], .5, bottom=mlp, color=SECONDARY,
           label="Attention heads")
    for xi, v in zip(x, mlp):
        ax.annotate(f"{v:.0%} MLP", (xi, v), ha="center", va="top",
                    textcoords="offset points", xytext=(0, -5), fontsize=9,
                    color="white", fontweight="bold")
    ax.set_xticks(x, [METHOD_NAMES.get(k[0], k[0]) for k in hm])
    ax.set(ylabel="Share of the shared circuit", ylim=(0, 1))
    _legend(ax, loc="lower right", fontsize=8)
    style(ax)

    # Depth profile per configuration, banded over tasks. Splitting by task
    # instead would ask a question about tasks; the comparison here is between
    # granularities, and the band shows the task spread is smaller than the split.
    ax = axes[1]
    for key in configs:
        sub = comp[(comp.method == key[0]) & (comp.granularity == key[1])]
        per_task = sub.groupby("task")[bins].mean()
        ax.fill_between(centers, per_task.min(), per_task.max(), color=_color(key),
                        alpha=.12, linewidth=0)
        ax.plot(centers, per_task.mean(), markersize=4, lw=1.8, label=_label(key),
                **_line_kw(key))
    ax.axhline(1 / DEPTH_BINS, ls="--", c="0.4", lw=1, zorder=0)
    ax.annotate("uniform over depth", (.02, 1 / DEPTH_BINS), textcoords="offset points",
                xytext=(0, 4), fontsize=8, color="0.4")
    ax.set(xlabel="Relative depth (layer / final layer)",
           ylabel="Share of the shared circuit", ylim=(0, None))
    style(ax)
    _legend(ax, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "circuit_composition_by_granularity.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", required=True, help="raw extraction root")
    ap.add_argument("--analysis-dir", required=True,
                    help="directory holding cross_task_tidy.csv; figures land here")
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--p", type=int, default=50)
    args = ap.parse_args()
    out = Path(args.analysis_dir)
    out.mkdir(parents=True, exist_ok=True)

    circuits = read_circuits(Path(args.results_root), args.K, args.p)
    fracs = shared_fractions(circuits)
    comp = composition(circuits)
    fracs.to_csv(out / "shared_fraction_pairs.csv", index=False)
    comp.to_csv(out / "circuit_composition.csv", index=False)

    gaps = None
    cross_path = out / "cross_task_tidy.csv"
    if cross_path.exists():
        cross = pd.read_csv(cross_path)
        diag = cross[(cross.donor == cross.target) & (cross.K == args.K) & (cross.p == args.p)]
        gaps = diag.groupby(CONFIG + ["model"]).specificity_gap_pp.mean()

    plot_shared_fraction(fracs, gaps, out)
    plot_composition(comp, out)
    print(fracs.groupby(CONFIG).shared_fraction.mean().to_string())
    print(comp[comp.granularity == "head_mlp"].groupby(CONFIG).mlp_fraction.mean().to_string())


if __name__ == "__main__":
    main()
