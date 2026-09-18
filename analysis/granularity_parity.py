"""Aggregate, audit, and plot the granularity-parity experiment.

The reader accepts the schema emitted by :mod:`cross_task_experiment` and the
existing extraction ``metrics.json`` files.  It deliberately keeps empty
circuits as rows, while masking specificity for an empty own circuit and
excluding empty foreign donors from foreign means.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

# Must be set before numpy/pandas (or any plotting backend) is imported.
_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "circuit_reuse_mplconfig"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd

EXCLUDED_MODELS = ("qwen3-8b",)
EXCLUDED_TASKS = ("mmlu",)
CONFIG = ["method", "granularity"]
EXTRACTION_METRICS = ["circuit_size", "reuse", "necessity_gap", "lift"]
CROSS_METRICS = ["own_drop_pp", "foreign_mean_drop_pp", "specificity_gap_pp",
                 "own_drop_rel", "foreign_mean_drop_rel",
                 "normalized_specificity_gap"]


def _excluded(model: str | None, task: str | None) -> bool:
    return (any(x in str(model) for x in EXCLUDED_MODELS)
            or str(task) in EXCLUDED_TASKS)


def infer_granularity(data: dict, path: Path | str = "") -> str:
    """Extraction metrics predate the explicit field; read it off a component."""
    if data.get("granularity"):
        return data["granularity"]
    for entry in data.get("by_k", {}).values():
        for cell in entry.get("thresholds", {}).values():
            components = cell.get("shared_components") or []
            if components:
                return "neuron" if str(components[0]).startswith("neuron[") else "head_mlp"
    return "neuron" if "neuron" in str(path) else "head_mlp"


def read_cross_task(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.rglob("cross_task_*.json")):
        try:
            data = json.loads(path.read_text())
            tasks = [t for t in data["tasks"] if t not in EXCLUDED_TASKS]
            if data.get("schema_version", 0) < 2 or data.get("skipped_donors"):
                continue
            if _excluded(data["model_name"], None):
                continue
            for donor in tasks:
                for target in tasks:
                    cell = data["cells"][donor][target]
                    rows.append({
                        "source": str(path), "model": data["model_name"],
                        "method": data.get("method", "eap"),
                        "granularity": data.get("granularity", "head_mlp"),
                        "K": int(data["K"]), "p": int(data["threshold"]),
                        "donor": donor, "target": target,
                        "donor_size": int(data["circuit_sizes"][donor]),
                        **cell,
                    })
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            continue
    return pd.DataFrame(rows)


def read_extraction(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.rglob("metrics.json")):
        try:
            d = json.loads(path.read_text())
            if d.get("skipped_examples", 0) or d.get("extraction_processed_examples", d.get("num_examples")) != d.get("extraction_expected_examples", d.get("num_examples")):
                continue
            if _excluded(d.get("model_name"), d.get("task")):
                continue
            granularity = infer_granularity(d, path)
            for K, entry in d.get("by_k", {}).items():
                for p, cell in entry.get("thresholds", {}).items():
                    train = cell.get("train", {})
                    baseline = d.get("baseline_train_accuracy", np.nan)
                    scale = (baseline if baseline is not None
                             and np.isfinite(baseline) and baseline != 0 else np.nan)
                    ablation = train.get("ablation_accuracy", np.nan)
                    control = train.get("control_accuracy", np.nan)
                    necessity_gap = (train.get("accuracy_drop_ablation", np.nan)
                                     - train.get("accuracy_drop_control", np.nan)) * 100
                    rows.append({
                        "source": str(path), "model": d.get("model_name"),
                        "task": d.get("task"), "method": d.get("method", "eap"),
                        "granularity": granularity,
                        "K": int(K), "p": int(p),
                        "circuit_size": int(cell.get("shared_circuit_size", 0)),
                        "reuse": float(cell.get("reuse_percent", np.nan)),
                        "baseline_correct": d.get("baseline_train_correct"),
                        "baseline_total": d.get("baseline_train_total"),
                        "baseline_accuracy": baseline,
                        "ablation_accuracy": ablation,
                        "control_accuracy": control,
                        "necessity_gap": necessity_gap,
                        # Lift is the necessity gap normalized by clean accuracy;
                        # the two are the same quantity on different scales.
                        "lift": (control - ablation) / scale * 100,
                    })
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return pd.DataFrame(rows)


def add_cross_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    keys = ["model", "method", "granularity", "K", "p"]
    diag = out[out["donor"] == out["target"]].set_index(keys + ["donor"])
    for src, dest in [("accuracy_drop_pp", "own_drop_pp"),
                      ("relative_drop_pct", "own_drop_rel")]:
        col = diag[src]
        out[dest] = [col.get((r.model, r.method, r.granularity, r.K, r.p, r.target), np.nan)
                     for r in out.itertuples()]
    nonempty = out[(out["donor_size"] > 0) & (out["donor"] != out["target"])]
    for src, dest in [("accuracy_drop_pp", "foreign_mean_drop_pp"),
                      ("relative_drop_pct", "foreign_mean_drop_rel")]:
        col = nonempty.groupby(keys + ["target"])[src].mean()
        out[dest] = [col.get((r.model, r.method, r.granularity, r.K, r.p, r.target), np.nan)
                     for r in out.itertuples()]
    out["specificity_gap_pp"] = np.where(out["donor_size"] > 0,
                                          out["own_drop_pp"] - out["foreign_mean_drop_pp"], np.nan)
    out["normalized_drop"] = out["relative_drop_pct"]
    # Own and foreign drops share a target, hence a baseline: the normalized gap
    # is the difference of the relative drops (equivalently, gap_pp / clean acc).
    out["normalized_specificity_gap"] = np.where(out["donor_size"] > 0,
                                                   out["own_drop_rel"] - out["foreign_mean_drop_rel"], np.nan)
    return out


def bootstrap_mean(df: pd.DataFrame, column: str, n: int = 10000, seed: int = 42) -> dict:
    """Hierarchical paired bootstrap: models first, tasks within each model."""
    clean = df.dropna(subset=[column])
    if clean.empty:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    rng = np.random.default_rng(seed)
    models = sorted(clean["model"].unique())
    task_col = "task" if "task" in clean else "target"
    # Reduce each model/task block to its mean before resampling. This is
    # exactly the paired hierarchical bootstrap unit and avoids materializing
    # 10,000 copies of large tidy tables.
    block_means = (clean.groupby(["model", task_col], sort=False)[column].mean()
                   .reset_index())
    by_model = {m: block_means[block_means.model == m].set_index(task_col)[column]
                for m in models}
    vals = []
    for _ in range(n):
        pieces = []
        for m in rng.choice(models, size=len(models), replace=True):
            block = by_model[m]
            pieces.extend(block.loc[rng.choice(block.index.values, size=len(block), replace=True)].to_numpy())
        vals.append(float(np.mean(pieces)))
    return {"mean": float(clean[column].mean()), "lo": float(np.quantile(vals, .025)),
            "hi": float(np.quantile(vals, .975)), "n": int(len(clean)), "resamples": n, "seed": seed}


def bootstrap_table(df: pd.DataFrame, metrics: list[str], n: int = 10000,
                    seed: int = 42) -> pd.DataFrame:
    rows = []
    for key, group in df.groupby(CONFIG):
        for metric in metrics:
            if metric in group:
                rows.append({**dict(zip(CONFIG, key)), "metric": metric,
                             **bootstrap_mean(group, metric, n=n, seed=seed)})
    return pd.DataFrame(rows)


def select_operating_point(extraction: pd.DataFrame, K: int = 10, min_coverage: float = .8):
    sub = extraction[extraction.K == K]
    if sub.empty:
        return {"K": K, "p": None, "reason": "no extraction rows"}
    configs = sub[["model", "method", "granularity"]].drop_duplicates()
    for p in sorted(sub.p.unique(), reverse=True):
        cov = sub[sub.p == p].groupby(["model", "method", "granularity"]).circuit_size.agg(lambda x: (x > 0).mean())
        if len(cov) and cov.reindex(pd.MultiIndex.from_frame(configs)).fillna(0).ge(min_coverage).all():
            return {"K": K, "p": int(p), "min_nonempty_coverage": float(cov.min()),
                    "min_coverage_criterion": min_coverage}
    return {"K": K, "p": int(sub.p.min()), "min_nonempty_coverage": 0.0,
            "min_coverage_criterion": min_coverage}


def summarize(extraction: pd.DataFrame, cross: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    within = extraction.groupby(["model", "method", "granularity", "K", "p"], as_index=False).agg(
        circuit_size=("circuit_size", "mean"),
        nonempty_coverage=("circuit_size", lambda x: (x > 0).mean()),
        reuse=("reuse", "mean"), necessity_gap=("necessity_gap", "mean"),
        lift=("lift", "mean")) if not extraction.empty else pd.DataFrame()
    # One row per target task: the diagonal carries own/foreign/specificity, and
    # its donor_size column is exactly the per-task non-empty indicator.
    diagonal = cross[cross.donor == cross.target] if not cross.empty else cross
    across = diagonal.groupby(["model", "method", "granularity", "K", "p"], as_index=False).agg(
        own_drop_pp=("own_drop_pp", "mean"),
        foreign_mean_drop_pp=("foreign_mean_drop_pp", "mean"),
        own_drop_rel=("own_drop_rel", "mean"),
        foreign_mean_drop_rel=("foreign_mean_drop_rel", "mean"),
        specificity_gap_pp=("specificity_gap_pp", "mean"),
        normalized_specificity_gap=("normalized_specificity_gap", "mean"),
        coverage=("donor_size", lambda x: (x > 0).mean())) if not cross.empty else pd.DataFrame()
    return within, across


# Display names and a fixed style, so every figure names methods, granularities,
# and tasks the way the prose does.
#
# Granularity is the paper's independent variable and attribution method only a
# robustness check, so the two get different visual channels rather than four
# equally weighted colours. Hue carries granularity, taken from the two ends of
# cividis, and shade carries method, with line style and marker as a redundant
# second cue. Navy against red is a far larger perceptual step than dark against
# light within one hue, so every figure reads granularity-first. Gold is the
# secondary and names a third category where one exists, never a granularity.
METHOD_NAMES = {"eap_ig": "EAP-IG", "relp": "RelP", "eap": "EAP"}
GRAN_NAMES = {"head_mlp": "attention heads and MLP blocks", "neuron": "MLP neurons"}
GRAN_SHORT = {"head_mlp": "component-level", "neuron": "neuron-level"}
TASK_NAMES = {"addition": "Addition", "arc_challenge": "ARC (Chal.)",
              "arc_easy": "ARC (Easy)", "boolean": "Boolean", "ioi": "IOI",
              "mcqa": "CopyColors MCQA"}
MODEL_NAMES = {"google/gemma-2-2b": "Gemma 2 2B",
               "google/gemma-2-2b-it": "Gemma 2 2B IT",
               "meta-llama/Llama-3.2-3B": "Llama 3.2 3B",
               "meta-llama/Llama-3.2-3B-Instruct": "Llama 3.2 3B Instruct",
               "qwen3-4b": "Qwen3 4B", "qwen3-8b": "Qwen3 8B",
               "allenai/OLMo-2-0425-1B": "OLMo 2 1B"}
CONFIG_ORDER = [("eap_ig", "head_mlp"), ("relp", "head_mlp"),
                ("eap_ig", "neuron"), ("relp", "neuron")]
# Sampled from Diebenkorn, Ocean Park #116: the dusty blue field against the
# ochre band, each with a lighter tone of itself for the second attribution
# method, and the mint band as the secondary.
GRAN_COLORS = {"head_mlp": "#5FA8D0", "neuron": "#E8B33C"}
GRAN_COLORS_LIGHT = {"head_mlp": "#AFD4E9", "neuron": "#F6DDA0"}
# Distinct hues for the five models, used wherever a figure is per-model.
MODEL_COLORS = ["#5FA8D0", "#E8B33C", "#66C79C", "#DD7F72", "#8E7CD0"]
METHOD_LINESTYLE = {"eap_ig": "-", "relp": (0, (4, 1.6))}
METHOD_MARKER = {"eap_ig": "o", "relp": "^"}
CONFIG_COLORS = {(m, g): (GRAN_COLORS if m == "eap_ig" else GRAN_COLORS_LIGHT)[g]
                 for m, g in CONFIG_ORDER}
# Sequential and diverging scales.
# along with the own/other pair used wherever a figure contrasts a task's own
# circuit against the mean over the others.
MISSING_COLOR = "#DFD8D0"
DROP_CMAP = "RdBu_r"
# Gold is the secondary: it names a third category where one exists, and it is
# the far end of the sequential ramp. It never denotes a granularity.
SECONDARY, SECONDARY_LIGHT = "#66C79C", "#B3E3CD"
# Own-against-other contrasts keep the panel's granularity hue for the task's own
# circuit and the secondary for foreign ones. Other-task circuits are a real
# experimental condition, not a control, so they take a colour; grey is reserved
# for controls.
OTHER_COLOR = SECONDARY


def _ramp():
    """Canvas-to-deep-blue sequential ramp, with its own colour for missing cells."""
    from matplotlib.colors import LinearSegmentedColormap
    cm = LinearSegmentedColormap.from_list(
        "ocean_park", ["#F8F6F1", "#D6E7F1", "#AFD4E9", "#5FA8D0", "#2E6E96"])
    return cm.with_extremes(bad=MISSING_COLOR)


LEGEND_KW = dict(frameon=True, fancybox=True, framealpha=1, edgecolor="0.2",
                 borderpad=.5, labelspacing=.4, handlelength=1.8)


def _legend(target, **kw):
    """Legend with a thin rounded box, so it stays readable over data."""
    leg = target.legend(**{**LEGEND_KW, **kw})
    leg.get_frame().set_linewidth(.7)
    return leg


def _color(key) -> str:
    return CONFIG_COLORS.get(tuple(key), GRAN_COLORS.get(key[1], "0.5"))


def _line_kw(key) -> dict:
    """Line, marker, and colour for one (method, granularity) configuration."""
    return {"color": _color(key), "linestyle": METHOD_LINESTYLE.get(key[0], "-"),
            "marker": METHOD_MARKER.get(key[0], "o")}


def _fill_kw(key) -> dict:
    """Face for a bar, box, or violin of one configuration."""
    return {"facecolor": _color(key), "edgecolor": "none"}


def _label(key) -> str:
    """Readable name for a (method, granularity) pair."""
    if isinstance(key, tuple) and len(key) == 2:
        return f"{METHOD_NAMES.get(key[0], key[0])}, {GRAN_SHORT.get(key[1], key[1])}"
    return "/".join(key) if isinstance(key, tuple) else str(key)


def _task_label(task: str) -> str:
    return TASK_NAMES.get(task, task)


def _model_label(model: str) -> str:
    return MODEL_NAMES.get(model, model.split("/")[-1])


def _ordered(groups):
    """Config keys in a fixed order, so colour and legend order never drift."""
    present = [k for k in CONFIG_ORDER if k in groups]
    return present + [k for k in groups if k not in present]


def _panel(ax, text: str):
    """Short identifying label for one panel of a multi-panel figure.

    Figures carry no titles: the claim and the provenance live in the caption,
    which is where a paper reader looks for them. Panel labels stay because a
    grid of panels still has to say which is which.
    """
    ax.set_title(text, loc="left", fontweight="bold", fontsize=10, pad=6)


def make_figures(extraction: pd.DataFrame, cross: pd.DataFrame, out: Path, point: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                         "axes.titlesize": 11, "axes.labelsize": 10,
                         "xtick.labelsize": 9, "ytick.labelsize": 9,
                         "legend.fontsize": 9, "figure.titlesize": 12})

    K, p = point.get("K"), point.get("p")
    n_models = extraction.model.nunique() if not extraction.empty else cross.model.nunique()
    n_tasks = extraction.task.nunique() if not extraction.empty else cross.target.nunique()

    def save(fig, name, tight=True):
        if tight:
            fig.tight_layout()
        fig.savefig(out / f"{name}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    def style(ax):
        ax.grid(alpha=.25, linewidth=.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    def line_vs_p(df, value, ylabel, name, symlog=False, note=""):
        sub = df[df.K == K]
        if sub.dropna(subset=[value]).empty:
            return
        fig, ax = plt.subplots(figsize=(7, 4.4))
        groups = dict(list(sub.groupby(CONFIG)))
        for key in _ordered(groups):
            series = groups[key].groupby("p")[value].mean()
            ax.plot(series.index, series.values, markersize=4.5, label=_label(key),
                    **_line_kw(key))
        ax.axvline(p, ls=":", c="grey", lw=1, zorder=0)
        ax.set(xlabel="Consensus threshold $P$ (%)", ylabel=ylabel)
        if symlog:
            # Sizes span two orders of magnitude and reach zero at strict P.
            ax.set_yscale("symlog", linthresh=1)
        style(ax)
        _legend(ax)
        save(fig, name)

    if not extraction.empty:
        # Consistency per task, both granularities. The band is the range over
        # models, so a line that separates by granularity can be told apart from
        # one that only separates because a single model is an outlier.
        at_p = extraction[extraction.p == p]
        tasks = sorted(at_p.task.unique())
        rows = [("reuse", "Reuse@$P$ (%)"), ("necessity_gap", "Necessity gap (pp)")]
        fig, axes = plt.subplots(len(rows), len(tasks), figsize=(2.35 * len(tasks), 5.4),
                                 sharex=True, squeeze=False)
        for r, (value, ylabel) in enumerate(rows):
            for c, task in enumerate(tasks):
                ax = axes[r][c]
                sub = at_p[at_p.task == task]
                groups = dict(list(sub.groupby(CONFIG)))
                for key in _ordered(groups):
                    per_model = groups[key].groupby(["K", "model"])[value].mean()
                    # The full range over five models is wide enough to swamp the
                    # lines, so the band is the interquartile range.
                    stats = per_model.groupby("K").agg(
                        mean="mean", lo=lambda s: s.quantile(.25),
                        hi=lambda s: s.quantile(.75))
                    ax.fill_between(stats.index, stats.lo, stats.hi,
                                    color=_color(key), alpha=.12, linewidth=0)
                    ax.plot(stats.index, stats["mean"], markersize=3.5, lw=1.8,
                            label=_label(key), **_line_kw(key))
                ax.set_xscale("log")
                ax.set_xticks([1, 5, 10, 30], ["1%", "5%", "10%", "30%"], minor=False)
                ax.minorticks_off()
                if r == 0:
                    ax.set_title(_task_label(task), fontsize=10)
                if r == len(rows) - 1:
                    ax.set_xlabel("Circuit size $K$ (%)")
                if c == 0:
                    ax.set_ylabel(ylabel)
                if value == "necessity_gap":
                    ax.axhline(0, c="0.3", lw=.8, zorder=0)
                style(ax)
            for ax in axes[r][1:]:
                ax.sharey(axes[r][0])
                ax.tick_params(labelleft=False)
        handles, labels = axes[0][0].get_legend_handles_labels()
        _legend(fig, handles=handles, labels=labels, ncol=len(labels), loc="lower center",
                   bbox_to_anchor=(.5, -.045))
        save(fig, "consistency_by_task")

        line_vs_p(extraction, "reuse", "Reuse@$P$ (%)", "within_task_reuse_vs_p")
        line_vs_p(extraction, "necessity_gap", "Necessity gap (pp)", "necessity_gap_vs_p")
        line_vs_p(extraction, "lift", "Normalized lift (%)", "normalized_lift_vs_p")
        line_vs_p(extraction, "circuit_size", "Mean circuit size", "circuit_size_vs_p",
                  symlog=True)

        # Coverage: fraction of non-empty shared circuits.
        cov = (extraction[extraction.K == K].groupby(CONFIG + ["p"])
               .circuit_size.agg(lambda x: (x > 0).mean()).reset_index())
        fig, ax = plt.subplots(figsize=(7, 4.4))
        groups = dict(list(cov.groupby(CONFIG)))
        for key in _ordered(groups):
            g = groups[key]
            ax.plot(g.p, g.circuit_size, markersize=4.5, label=_label(key),
                    **_line_kw(key))
        crit = point.get("min_coverage_criterion", .8)
        ax.axhline(crit, ls="--", c="k", lw=1, label=f"{crit:.0%} criterion")
        ax.axvline(p, ls=":", c="grey", lw=1, zorder=0)
        ax.set(xlabel="Consensus threshold $P$ (%)",
               ylabel="Fraction of tasks with non-empty $S_P$", ylim=(-.05, 1.05))
        style(ax)
        _legend(ax)
        save(fig, "nonempty_coverage_vs_p")

        # K sweep at the selected P.
        at_p = extraction[extraction.p == p]
        if not at_p.empty:
            metrics = [("reuse", "Reuse@$P$ (%)"), ("circuit_size", "Mean circuit size"),
                       ("necessity_gap", "Necessity gap (pp)"), ("lift", "Normalized lift (%)")]
            fig, axes = plt.subplots(1, len(metrics), figsize=(4.1 * len(metrics), 3.9))
            groups = dict(list(at_p.groupby(CONFIG)))
            for ax, (value, ylabel) in zip(axes, metrics):
                for key in _ordered(groups):
                    series = groups[key].groupby("K")[value].mean()
                    ax.plot(series.index, series.values, markersize=4.5,
                            label=_label(key), **_line_kw(key))
                ax.set(xlabel="Circuit size $K$ (%)", ylabel=ylabel)
                if value == "circuit_size":
                    ax.set_yscale("symlog", linthresh=1)
                style(ax)
            handles, labels = axes[0].get_legend_handles_labels()
            _legend(fig, handles=handles, labels=labels, ncol=len(labels),
                       loc="lower center", bbox_to_anchor=(.5, -.06))
            save(fig, "k_sweep_panels")

        # Circuit-size distributions at the operating point.
        at_point = extraction[(extraction.K == K) & (extraction.p == p)]
        if not at_point.empty:
            groups = _ordered(sorted(at_point.groupby(CONFIG).groups))
            data = [at_point[(at_point.method == m) & (at_point.granularity == g)]
                    .circuit_size.values for m, g in groups]
            fig, ax = plt.subplots(figsize=(7.2, 4.4))
            bp = ax.boxplot(data, tick_labels=[_label(k) for k in groups],
                            patch_artist=True, medianprops=dict(color="black"))
            for patch, key in zip(bp["boxes"], groups):
                patch.set(**_fill_kw(key))
            ax.set(ylabel="Shared circuit size (components)", yscale="log")
            style(ax)
            save(fig, "circuit_size_distributions")

    if cross.empty:
        return
    # Own, foreign, and specificity are one value per target, carried on the
    # diagonal. Averaging over all donor rows would weight targets by how many
    # donors happened to have a non-empty circuit.
    per_target = cross[cross.donor == cross.target]
    line_vs_p(per_target, "specificity_gap_pp", "Specificity gap (pp)",
              "specificity_gap_vs_p")
    line_vs_p(per_target, "normalized_specificity_gap",
              "Normalized specificity gap (%)", "normalized_specificity_gap_vs_p")
    line_vs_p(per_target, "own_drop_pp", "Own-task drop (pp)", "own_drop_vs_p")
    line_vs_p(per_target, "foreign_mean_drop_pp", "Other-task mean drop (pp)",
              "foreign_drop_vs_p")

    at_point = per_target[(per_target.K == K) & (per_target.p == p)]

    # Own vs other, aggregated per configuration.
    if not at_point.empty:
        groups = _ordered(sorted(at_point.groupby(CONFIG).groups))
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
        for ax, (own, foreign, ylabel) in zip(axes, [
                ("own_drop_pp", "foreign_mean_drop_pp", "Accuracy drop (pp)"),
                ("own_drop_rel", "foreign_mean_drop_rel", "Relative accuracy drop (%)")]):
            x = np.arange(len(groups))
            means = at_point.groupby(CONFIG)[[own, foreign]].mean().reindex(groups)
            ax.bar(x - .19, means[own].values, .38, label="Own circuit",
                   color=[_color(k) for k in groups])
            ax.bar(x + .19, means[foreign].values, .38, label="Other circuits (mean)",
                   color=OTHER_COLOR)
            for xi, (o, f) in enumerate(zip(means[own].values, means[foreign].values)):
                ax.annotate("", xy=(xi + .19, f), xytext=(xi - .19, o),
                            arrowprops=dict(arrowstyle="-", color="0.5", lw=.8, ls=":"))
            ax.set_xticks(x, [_label(k) for k in groups], rotation=12, ha="right")
            ax.set(ylabel=ylabel)
            style(ax)
        _legend(axes[0])
        save(fig, "own_vs_foreign_effects")

    # Every model-task cell as its own Other -> Own segment. The means alone
    # cannot show whether a granularity separates consistently or on average, and
    # this is the figure the specificity claim actually rests on.
    configs = _ordered(sorted(at_point.groupby(CONFIG).groups))
    fig, axes = plt.subplots(1, len(configs), figsize=(2.9 * len(configs), 4.4),
                             sharey=True, squeeze=False)
    for ax, key in zip(axes[0], configs):
        sub = at_point[(at_point.method == key[0]) & (at_point.granularity == key[1])]
        sub = sub.dropna(subset=["own_drop_pp", "foreign_mean_drop_pp"])
        color = _color(key)
        for r in sub.itertuples():
            rises = r.own_drop_pp > r.foreign_mean_drop_pp
            ax.plot([0, 1], [r.foreign_mean_drop_pp, r.own_drop_pp], lw=.9,
                    color=color if rises else "0.65", alpha=.5 if rises else .4,
                    marker="o", markersize=2.5, zorder=2)
        means = [sub.foreign_mean_drop_pp.mean(), sub.own_drop_pp.mean()]
        ax.plot([0, 1], means, lw=3, color=color, marker="o", markersize=7,
                markeredgecolor="white", markeredgewidth=1.2, zorder=4)
        gap = means[1] - means[0]
        ax.annotate(f"gap {gap:+.1f} pp", (1, means[1]), textcoords="offset points",
                    xytext=(-4, 12), ha="right", fontsize=9, fontweight="bold")
        ax.set_xlim(-.25, 1.25)
        ax.set_xticks([0, 1], ["Other\ncircuits", "Own\ncircuit"])
        ax.axhline(0, c="0.3", lw=.8, zorder=0)
        _panel(ax, _label(key))
        style(ax)
        ax.grid(alpha=.25, linewidth=.6, axis="y")
        ax.xaxis.grid(False)
    axes[0][0].set_ylabel("Accuracy drop (pp)")
    # The grey/coloured split is load-bearing, so it belongs in the legend rather
    # than only in the caption.
    handles = [plt.Line2D([], [], color="0.45", lw=1.6, label="own exceeds other"),
               plt.Line2D([], [], color="0.72", lw=1.6, label="other exceeds own"),
               plt.Line2D([], [], color="0.2", lw=3, marker="o", markersize=6,
                          label="configuration mean")]
    _legend(fig, handles=handles, ncol=3, loc="lower center",
               bbox_to_anchor=(.5, -.11))
    save(fig, "own_vs_other_paired")

    # Own versus other per task, split by granularity.
    tasks = sorted(at_point.target.unique())
    configs = _ordered(sorted(at_point.groupby(CONFIG).groups))
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), squeeze=False, sharey=True,
                             sharex=True)
    x = np.arange(len(tasks))
    for ax, key in zip(axes.ravel(), configs):
        sub = at_point[(at_point.method == key[0]) & (at_point.granularity == key[1])]
        own = [sub[sub.target == t].own_drop_pp.mean() for t in tasks]
        other = [sub[sub.target == t].foreign_mean_drop_pp.mean() for t in tasks]
        ax.bar(x - .19, own, .38, label="Own circuit", color=_color(key))
        ax.bar(x + .19, other, .38, label="Other circuits (mean)", color=OTHER_COLOR)
        ax.set_xticks(x, [_task_label(t) for t in tasks], rotation=30, ha="right")
        ax.axhline(0, c="0.3", lw=.8)
        _panel(ax, _label(key))
        style(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("Accuracy drop (pp)")
    _legend(axes[0][0])
    save(fig, "own_vs_other_by_granularity")

    # Consistency against specificity, one point per configuration and model.
    reuse = (extraction[(extraction.K == K) & (extraction.p == p)]
             .groupby(CONFIG + ["model"]).reuse.mean())
    gap = at_point.groupby(CONFIG + ["model"]).specificity_gap_pp.mean()
    joined = pd.concat([reuse, gap], axis=1).dropna().reset_index()
    if not joined.empty:
        fig, ax = plt.subplots(figsize=(7, 4.8))
        groups = dict(list(joined.groupby(CONFIG)))
        for key in _ordered(groups):
            g = groups[key]
            ax.scatter(g.reuse, g.specificity_gap_pp, s=64, alpha=.85,
                       color=_color(key), marker=METHOD_MARKER.get(key[0], "o"),
                       label=_label(key), edgecolor="white", linewidth=.8)
        ax.axhline(0, ls="--", c="grey", lw=1, zorder=0)
        ax.set(xlabel="Reuse@$P$ (%)  — consistency",
               ylabel="Specificity gap (pp)  — specificity")
        style(ax)
        _legend(ax)
        save(fig, "specificity_vs_reuse")

    # Per-model specificity gap, to show the effect is not carried by one model.
    per_model = at_point.groupby(CONFIG + ["model"]).specificity_gap_pp.mean().reset_index()
    if not per_model.empty:
        models = sorted(per_model.model.unique())
        configs = _ordered(sorted(per_model.groupby(CONFIG).groups))
        x = np.arange(len(models))
        width = .8 / len(configs)
        fig, ax = plt.subplots(figsize=(9.5, 4.4))
        for i, key in enumerate(configs):
            sub = per_model[(per_model.method == key[0]) & (per_model.granularity == key[1])]
            vals = [sub[sub.model == m].specificity_gap_pp.mean() for m in models]
            ax.bar(x + i * width - .4 + width / 2, vals, width, label=_label(key),
                   **_fill_kw(key))
        ax.axhline(0, c="0.3", lw=.9)
        ax.set_xticks(x, [_model_label(m) for m in models], rotation=18, ha="right")
        ax.set(ylabel="Specificity gap (pp)")
        style(ax)
        _legend(ax, ncol=2)
        save(fig, "specificity_gap_by_model")

    # One panel figure per configuration, models side by side.
    grid = cross[(cross.K == K) & (cross.p == p)]
    for value, tag, label_text in [("accuracy_drop_pp", "full", "Accuracy drop (pp)"),
                                   ("relative_drop_pct", "normalized", "Relative drop (%)")]:
        limit = float(np.nanmax(np.abs(grid[value].to_numpy()))) if len(grid) else 1.0
        for key, group in grid.groupby(CONFIG):
            models = sorted(group.model.unique())
            fig, axes = plt.subplots(1, len(models), figsize=(3.3 * len(models) + 1.4, 3.9),
                                     squeeze=False)
            for i, (ax, model) in enumerate(zip(axes[0], models)):
                pivot = (group[group.model == model]
                         .pivot_table(index="donor", columns="target", values=value))
                im = ax.imshow(pivot.values, cmap=DROP_CMAP, vmin=-limit, vmax=limit)
                ax.set_xticks(range(len(pivot.columns)),
                              [_task_label(c) for c in pivot.columns],
                              rotation=45, ha="right", fontsize=7.5)
                # Repeating y labels on every panel runs them into the panel to
                # the left, so only the first column carries them.
                if i == 0:
                    ax.set_yticks(range(len(pivot.index)),
                                  [_task_label(j) for j in pivot.index], fontsize=7.5)
                else:
                    ax.set_yticks(range(len(pivot.index)), [""] * len(pivot.index))
                ax.set_title(_model_label(model), fontsize=9.5)
            axes[0][0].set_ylabel("Donor task")
            fig.colorbar(im, ax=axes[0].tolist(), label=label_text, fraction=.025)
            fig.savefig(out / f"heatmaps_{tag}_{'_'.join(key)}_K{K}_p{p}.png", dpi=160,
                        bbox_inches="tight")
            plt.close(fig)


def write_artifacts(extraction: pd.DataFrame, cross: pd.DataFrame, out: Path,
                    plots: bool = False, resamples: int = 10000, seed: int = 42):
    out.mkdir(parents=True, exist_ok=True)
    extraction.to_csv(out / "extraction_tidy.csv", index=False)
    cross.to_csv(out / "cross_task_tidy.csv", index=False)
    point = select_operating_point(extraction)
    summary = {"operating_point": point,
               "excluded_models": list(EXCLUDED_MODELS),
               "excluded_tasks": list(EXCLUDED_TASKS),
               "models": sorted(extraction.model.unique()) if not extraction.empty else [],
               "tasks": sorted(extraction.task.unique()) if not extraction.empty else []}
    if not extraction.empty:
        summary["bootstrap"] = bootstrap_mean(extraction, "necessity_gap", n=resamples, seed=seed)
    (out / "operating_point.json").write_text(json.dumps(summary, indent=2, allow_nan=False))

    within, across = summarize(extraction, cross)
    if not within.empty:
        within.to_csv(out / "within_task_summary.csv", index=False)
    if not across.empty:
        across.to_csv(out / "cross_task_summary.csv", index=False)

    K, p = point.get("K"), point.get("p")
    tables = []
    if not extraction.empty:
        at_point = extraction[(extraction.K == K) & (extraction.p == p)]
        tables.append(bootstrap_table(at_point, EXTRACTION_METRICS, n=resamples, seed=seed))
    if not cross.empty:
        at_point = cross[(cross.K == K) & (cross.p == p) & (cross.donor == cross.target)]
        tables.append(bootstrap_table(at_point, CROSS_METRICS, n=resamples, seed=seed))
    if tables:
        pd.concat(tables, ignore_index=True).to_csv(out / "bootstrap_summary.csv", index=False)
    if plots:
        make_figures(extraction, cross, out, point)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", nargs="+",
                        help="raw extraction/cross-task JSON roots; omit with --replot")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--plots", action="store_true", help="enable Matplotlib figures")
    parser.add_argument("--replot", action="store_true",
                        help="regenerate figures from the tidy CSVs already in "
                             "--output-dir, without re-reading raw results")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    out = Path(args.output_dir)

    if args.replot:
        # The tidy CSVs carry every column the figures use, so plots can be
        # rebuilt long after the raw results are gone.
        extraction = pd.read_csv(out / "extraction_tidy.csv")
        cross = pd.read_csv(out / "cross_task_tidy.csv")
        point = json.loads((out / "operating_point.json").read_text())["operating_point"]
        make_figures(extraction, cross, out, point)
        print(f"replotted from CSVs: extraction_rows={len(extraction)} "
              f"cross_task_rows={len(cross)}")
        return

    if not args.results_root:
        parser.error("--results-root is required unless --replot is given")
    roots = [Path(r) for r in args.results_root]
    extraction = pd.concat([read_extraction(r) for r in roots], ignore_index=True)
    cross = add_cross_metrics(pd.concat([read_cross_task(r) for r in roots], ignore_index=True))
    write_artifacts(extraction, cross, out, plots=args.plots,
                    resamples=args.resamples, seed=args.seed)
    print(f"extraction_rows={len(extraction)} cross_task_rows={len(cross)}")


if __name__ == "__main__":
    main()
