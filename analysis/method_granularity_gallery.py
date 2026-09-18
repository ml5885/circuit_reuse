"""Plot every method-by-granularity cell on its own, with no cross-cell averaging.

One section per (method, granularity) cell, three families: lift and accuracy
drop, own-against-other-task drops with every donor shown separately, and reuse
statistics. Each family carries the full sweep over circuit size K and consensus
threshold P, as small-multiple grids, as bars at every point of that grid, and as
tables. The tidy CSVs written by :mod:`granularity_parity` are the only input, so
the gallery cannot drift from the audited artifact.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from granularity_parity import (GRAN_SHORT, METHOD_NAMES, MODEL_COLORS,
                                MODEL_NAMES, TASK_NAMES, _legend)

CELLS = [("relp", "neuron"), ("relp", "head_mlp"),
         ("eap_ig", "neuron"), ("eap_ig", "head_mlp")]
K_DEFAULT, P_DEFAULT = 10, 50
TASK_COLORS = ["#5FA8D0", "#E8B33C", "#66C79C", "#DD7F72", "#8E7CD0", "#3F7A5E"]
EMPTY_COLOR = "#DFD8D0"
OTHER_HATCH = "///"
DPI = 120


def cell_slug(cell) -> str:
    return f"{cell[0]}_{cell[1]}"


def cell_title(cell) -> str:
    return f"{METHOD_NAMES[cell[0]]}, {GRAN_SHORT[cell[1]]}"


def anchor(text: str) -> str:
    return text.lower().replace(",", "").replace(" ", "-")


def _setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                         "axes.titlesize": 11, "axes.labelsize": 10,
                         "xtick.labelsize": 9, "ytick.labelsize": 9,
                         "legend.fontsize": 9, "figure.titlesize": 12})
    return plt


def pooled_legend(fig, axes, order: list[str], extra: dict | None = None, **kw):
    """Legend pooled over panels, since a panel need not use every label.

    Matplotlib returns lines before patches, so the order is imposed here.
    """
    pooled = {}
    for ax in axes:
        pooled.update(dict(zip(*reversed(ax.get_legend_handles_labels()))))
    pooled.update(extra or {})
    labels = [l for l in order if l in pooled] + [l for l in pooled if l not in order]
    _legend(fig, handles=[pooled[l] for l in labels], labels=labels, **kw)


def style(ax):
    ax.grid(alpha=.25, linewidth=.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def sweep_grid(plt, sub: pd.DataFrame, value: str, ylabel: str, models: list[str],
               tasks: list[str], ks: list[int], zero_line: bool = False,
               symlog: bool = False):
    """One row per K, one column per task, one line per model, P on the x-axis."""
    fig, axes = plt.subplots(len(ks), len(tasks), figsize=(2.4 * len(tasks), 2.0 * len(ks)),
                             sharex=True, sharey=True, squeeze=False)
    for r, k in enumerate(ks):
        at_k = sub[sub.K == k]
        for c, task in enumerate(tasks):
            ax = axes[r][c]
            at_task = at_k[at_k.task == task]
            for model, color in zip(models, MODEL_COLORS):
                series = at_task[at_task.model == model].groupby("p")[value].mean()
                ax.plot(series.index, series.values, color=color, marker="o",
                        markersize=2.8, lw=1.4, label=MODEL_NAMES.get(model, model))
            if zero_line:
                ax.axhline(0, c="0.3", lw=.8, zorder=0)
            if r == 0:
                ax.set_title(TASK_NAMES.get(task, task), fontsize=10)
            if r == len(ks) - 1:
                ax.set_xlabel("Consensus threshold $P$ (%)")
            style(ax)
        axes[r][0].set_ylabel(f"$K={k}$%\n{ylabel}")
    if symlog:
        axes[0][0].set_yscale("symlog", linthresh=1)
    pooled_legend(fig, axes[0], [MODEL_NAMES.get(m, m) for m in models],
                  ncol=len(models), loc="lower center", bbox_to_anchor=(.5, -.04))
    fig.tight_layout()
    return fig


def metric_bars(plt, sub: pd.DataFrame, series: list[tuple[str, str, str]], ylabel: str,
                models: list[str], tasks: list[str], ks: list[int],
                empty_column: str = "circuit_size"):
    """One row per K, one column per model, tasks on the x-axis."""
    fig, axes = plt.subplots(len(ks), len(models),
                             figsize=(3.0 * len(models), 2.3 * len(ks)),
                             sharex=True, sharey="row", squeeze=False)
    x = np.arange(len(tasks))
    width = .8 / len(series)
    for r, k in enumerate(ks):
        at_k = sub[sub.K == k]
        for c, model in enumerate(models):
            ax = axes[r][c]
            rows = at_k[at_k.model == model].set_index("task")
            for i, (label, column, color) in enumerate(series):
                offset = (i - (len(series) - 1) / 2) * width
                ax.bar(x + offset, [rows[column].get(t, np.nan) for t in tasks],
                       width, color=color, label=label)
            empty = [xi for xi, t in zip(x, tasks) if rows[empty_column].get(t, 0) == 0]
            ax.plot(empty, np.zeros(len(empty)), linestyle="none", marker="x",
                    markersize=5, markeredgewidth=1.2, color="#8A8279", zorder=3,
                    label="Empty circuit")
            ax.axhline(0, c="0.3", lw=.8, zorder=0)
            if r == 0:
                ax.set_title(MODEL_NAMES.get(model, model), fontsize=10)
            if r == len(ks) - 1:
                ax.set_xticks(x, [TASK_NAMES.get(t, t) for t in tasks],
                              rotation=35, ha="right")
            style(ax)
        axes[r][0].set_ylabel(f"$K={k}$%\n{ylabel}")
    pooled_legend(fig, axes[0], [s[0] for s in series] + ["Empty circuit"],
                  ncol=len(series) + 1, loc="lower center", bbox_to_anchor=(.5, -.03))
    fig.tight_layout()
    return fig


def own_vs_other(plt, sub: pd.DataFrame, value: str, ylabel: str,
                 models: list[str], tasks: list[str]):
    """One panel per model, one bar per donor task, own donor marked."""
    fig, axes = plt.subplots(len(models), 1, figsize=(11, 2.5 * len(models)),
                             sharex=True, squeeze=False)
    width = .8 / len(tasks)
    x = np.arange(len(tasks))
    for ax, model in zip(axes[:, 0], models):
        at_model = sub[sub.model == model]
        for i, donor in enumerate(tasks):
            rows = at_model[at_model.donor == donor].set_index("target")
            offset = (i - (len(tasks) - 1) / 2) * width
            values = [rows[value].get(t, np.nan) for t in tasks]
            empty = [rows.donor_size.get(t, 0) == 0 for t in tasks]
            colors = [EMPTY_COLOR if e else TASK_COLORS[i] for e in empty]
            bars = ax.bar(x + offset, values, width, color=colors, edgecolor="white",
                          linewidth=0, label=TASK_NAMES.get(donor, donor))
            # A solid bar is the task's own circuit. Hatching separates the
            # foreign circuits without changing their colour.
            for bar, target in zip(bars, tasks):
                if target != donor:
                    bar.set_hatch(OTHER_HATCH)
            # An empty circuit gives a drop of exactly zero, which a bar chart
            # would show as a real null result. Mark it instead.
            marks = [xi + offset for xi, e in zip(x, empty) if e]
            ax.plot(marks, np.zeros(len(marks)), linestyle="none", marker="x",
                    markersize=5.5, markeredgewidth=1.2, color="#8A8279", zorder=3,
                    label="Empty circuit")
        ax.axhline(0, c="0.3", lw=.8, zorder=0)
        ax.set_ylabel(ylabel)
        ax.set_title(MODEL_NAMES.get(model, model), loc="left", fontweight="bold",
                     fontsize=10, pad=5)
        style(ax)
    axes[-1, 0].set_xticks(x, [TASK_NAMES.get(t, t) for t in tasks])
    axes[-1, 0].set_xlabel("Evaluated task")
    from matplotlib.patches import Patch
    proxies = {"Own circuit (solid)": Patch(facecolor="0.55", edgecolor="none"),
               "Other task's circuit (hatched)":
                   Patch(facecolor="0.55", edgecolor="white", linewidth=0,
                         hatch=OTHER_HATCH)}
    pooled_legend(fig, axes[:, 0],
                  [TASK_NAMES.get(t, t) for t in tasks]
                  + list(proxies) + ["Empty circuit"],
                  extra=proxies, ncol=len(tasks) + 1, loc="lower center",
                  bbox_to_anchor=(.5, -.03), title="Ablated circuit")
    return fig


def markdown_table(frame: pd.DataFrame, floats: int = 1) -> list[str]:
    """Render a frame as a GitHub table. Written out because tabulate is absent."""
    def fmt(v):
        if isinstance(v, float):
            return "" if not np.isfinite(v) else f"{v:.{floats}f}"
        return str(v)

    header = [str(c) for c in frame.columns]
    rows = [[fmt(v) for v in row] for row in frame.itertuples(index=False)]
    align = ["---" if frame[c].dtype.kind in "if" else ":--" for c in frame.columns]
    return (["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"]
            + ["| " + " | ".join(r) + " |" for r in rows] + [""])


def pivot_by_p(frame: pd.DataFrame, value: str, index: list[str]) -> pd.DataFrame:
    """Rows are the index columns, columns are the P grid."""
    wide = frame.pivot_table(index=index, columns="p", values=value, aggfunc="mean")
    wide.columns = [f"P={c}" for c in wide.columns]
    out = wide.reset_index()
    for column, names in [("model", MODEL_NAMES), ("task", TASK_NAMES),
                          ("donor", TASK_NAMES), ("target", TASK_NAMES)]:
        if column in out:
            out[column] = out[column].map(lambda v: names.get(v, v))
    return out.rename(columns={"model": "Model", "task": "Task", "donor": "Ablated",
                               "target": "Evaluated"})


def details(summary: str, body: list[str]) -> list[str]:
    return ["<details>", f"<summary>{summary}</summary>", ""] + body + ["</details>", ""]


# Column, table label, decimals, and the family whose section the table joins.
EXTRACTION_METRICS = [("reuse", "Reuse@P (%)", 1, "reuse"),
                      ("circuit_size", "Shared circuit size", 0, "reuse"),
                      ("lift", "Normalized lift (%)", 1, "lift"),
                      ("necessity_gap", "Necessity gap (pp)", 1, "lift"),
                      ("drop_ablation_pp", "Accuracy drop, circuit ablated (pp)", 1,
                       "lift"),
                      ("drop_control_pp", "Accuracy drop, random control (pp)", 1,
                       "lift")]


def build_cell(plt, ext: pd.DataFrame, cro: pd.DataFrame, out: Path, slug: str,
               models: list[str], tasks: list[str], ks: list[int], ps: list[int],
               K: int, p: int) -> dict:
    """Every figure and table for one (method, granularity) cell."""
    figures = {"lift": [], "own_vs_other": [], "reuse": []}

    def save(fig, name, family, caption):
        fig.savefig(out / f"{name}.png", dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        figures[family].append({"file": f"{name}.png", "caption": caption})

    # --- Lift and accuracy drop ---------------------------------------------
    save(sweep_grid(plt, ext, "lift", "Normalized lift (%)", models, tasks, ks,
                    zero_line=True),
         f"sweep_lift__{slug}", "lift",
         "Normalized lift for every circuit size and every consensus threshold.")
    save(sweep_grid(plt, ext, "necessity_gap", "Necessity gap (pp)", models, tasks, ks,
                    zero_line=True),
         f"sweep_necessity_gap__{slug}", "lift",
         "Necessity gap for every circuit size and every consensus threshold.")
    for pi in ps:
        save(metric_bars(plt, ext[ext.p == pi],
                         [("Circuit ablated", "drop_ablation_pp", "#5FA8D0"),
                          ("Random control", "drop_control_pp", "#B7B0A6")],
                         "Accuracy drop (pp)", models, tasks, ks),
             f"bars_drop_p{pi}__{slug}", "lift",
             f"Accuracy drop from ablating the circuit and from a size-matched "
             f"random control, at $P={pi}$%. A cross marks an empty circuit.")

    # --- Own against other tasks --------------------------------------------
    for pi in ps:
        for k in ks:
            save(own_vs_other(plt, cro[(cro.K == k) & (cro.p == pi)],
                              "accuracy_drop_pp", "Accuracy drop (pp)", models, tasks),
                 f"own_vs_other_k{k}_p{pi}__{slug}", "own_vs_other",
                 f"Accuracy drop on each evaluated task after ablating each task's "
                 f"circuit, at $K={k}$% and $P={pi}$%.")
    save(own_vs_other(plt, cro[(cro.K == K) & (cro.p == p)], "relative_drop_pct",
                      "Relative drop (%)", models, tasks),
         f"own_vs_other_relative_k{K}_p{p}__{slug}", "own_vs_other",
         f"Relative accuracy drop for the same comparison, at $K={K}$% and $P={p}$%. "
         f"The relative drop for every other grid point is in the tables and the CSV.")

    # --- Reuse statistics ----------------------------------------------------
    save(sweep_grid(plt, ext, "reuse", "Reuse@$P$ (%)", models, tasks, ks),
         f"sweep_reuse__{slug}", "reuse",
         "Reuse for every circuit size and every consensus threshold.")
    save(sweep_grid(plt, ext, "circuit_size", "Shared circuit size", models, tasks, ks,
                    symlog=True),
         f"sweep_circuit_size__{slug}", "reuse",
         "Shared circuit size for every circuit size and every consensus threshold. "
         "The vertical scale is symmetric-log.")
    for pi in ps:
        save(metric_bars(plt, ext[ext.p == pi],
                         [("Reuse@$P$", "reuse", "#E8B33C")], "Reuse@$P$ (%)",
                         models, tasks, ks),
             f"bars_reuse_p{pi}__{slug}", "reuse",
             f"Reuse at $P={pi}$%. A cross marks an empty circuit.")
        save(metric_bars(plt, ext[ext.p == pi],
                         [("Shared circuit size", "circuit_size", "#66C79C")],
                         "Shared circuit size", models, tasks, ks),
             f"bars_circuit_size_p{pi}__{slug}", "reuse",
             f"Shared circuit size at $P={pi}$%.")

    # --- Tables ---------------------------------------------------------------
    ext.to_csv(out / f"table_extraction__{slug}.csv", index=False)
    cro.to_csv(out / f"table_cross_task__{slug}.csv", index=False)
    tables = {"lift": [], "own_vs_other": [], "reuse": []}
    # Markdown and math are not processed inside a <summary> element, so the
    # summaries stay plain text.
    for value, label, digits, family in EXTRACTION_METRICS:
        for k in ks:
            frame = pivot_by_p(ext[ext.K == k], value, ["model", "task"])
            tables[family] += details(
                f"{label} at K={k}%, one row per model and task",
                markdown_table(frame, floats=digits))
    for value, label in [("accuracy_drop_pp", "Accuracy drop (pp)"),
                         ("relative_drop_pct", "Relative drop (%)")]:
        for k in ks:
            frame = pivot_by_p(cro[cro.K == k], value, ["model", "donor", "target"])
            tables["own_vs_other"] += details(
                f"{label} at K={k}%, one row per model, ablated circuit, and "
                f"evaluated task", markdown_table(frame))
    return {"figures": figures, "tables": tables}


# Every formula below is read off the code that produced the tidy CSVs:
# main_experiment.py for the extraction metrics, cross_task_experiment.py for the
# cross-task drops, and granularity_parity.py for the derived columns.
DEFINITIONS = r"""## Definitions

**Unit.** A unit is one attention head or one MLP block at component level. A unit
is one MLP neuron at neuron level. Write $N$ for the number of scored units.

**Per-example circuit.** The attribution method scores every unit on every example.
The per-example circuit $C_i$ is the top $K$% of units by score on example $i$:

$$m = |C_i| = \max\left(1,\ \left\lfloor N \cdot \frac{K}{100} \right\rfloor\right)$$

**Score sign.** The selection ranks units by the stored score in descending order.
At neuron level, both methods store a signed sum of attributions, so a unit with a
large negative attribution ranks low. At component level, RelP also stores a
signed sum. At component level, EAP-IG stores a sum of absolute edge
attributions, so its scores are never negative.

**Shared circuit.** A unit enters the shared circuit $S_P$ if it is in the
per-example circuit for at least $P$% of the $n$ scored examples:

$$S_P = \left\{\, u \ :\ \sum_{i=1}^{n} \mathbb{1}[\,u \in C_i\,] \ \geq\ \left\lceil \frac{P}{100}\, n \right\rceil \,\right\}$$

**Reuse.** Reuse is the fraction of one per-example circuit that survives the
consensus threshold. It is capped at 100%:

$$\text{Reuse}@P = \frac{\min(|S_P|,\ m)}{m} \times 100$$

**Ablation.** Ablation sets the output of every unit in a set to zero. Every
experiment in this gallery uses zero-ablation.

**Control.** The control set is a random set of units of the same size as $S_P$,
matched by unit type, and sampled without replacement from the units outside $S_P$.

**Accuracy drop, necessity gap, and lift.** Write $a_0$ for the clean accuracy,
$a_S$ for the accuracy after ablating $S_P$, and $a_C$ for the accuracy after
ablating the control set:

$$\text{drop, circuit ablated (pp)} = (a_0 - a_S) \times 100$$

$$\text{drop, random control (pp)} = (a_0 - a_C) \times 100$$

$$\text{necessity gap (pp)} = (a_0 - a_S) \times 100 - (a_0 - a_C) \times 100 = (a_C - a_S) \times 100$$

$$\text{normalized lift (\%)} = \frac{a_C - a_S}{a_0} \times 100$$

Lift and the necessity gap are the same quantity on two scales. Lift divides the
gap by the clean accuracy, so it stays comparable across tasks whose clean
accuracy differs. If $S_P$ is empty, the code sets $a_S = a_C = a_0$, and all four
quantities are then exactly zero.

**Cross-task drop.** The donor is the task whose shared circuit is ablated. The
target is the task that is evaluated. Write $b_0$ for the target's clean accuracy
and $b_D$ for its accuracy after ablating the donor's shared circuit:

$$\text{accuracy drop (pp)} = (b_0 - b_D) \times 100$$

$$\text{relative drop (\%)} = \frac{b_0 - b_D}{b_0} \times 100$$

The own drop is the case where the donor and the target are the same task. A
positive drop means the accuracy fell. A negative drop means the accuracy rose.

**Sample sizes.** Extraction uses 1000 examples per task, split 80% train and 20%
validation. CopyColors MCQA has 50 examples in total. Attribution and every
extraction figure use the train split, so $n$ is 800 for five tasks and 40 for
CopyColors MCQA. The cross-task evaluation uses 100 examples per task, and 50 for
CopyColors MCQA.
"""

FAMILY_TITLES = {"lift": "Lift and accuracy drop",
                 "own_vs_other": "Own circuit against other tasks' circuits",
                 "reuse": "Reuse statistics"}
FAMILY_NOTES = {
    "own_vs_other": "Each panel is one model. Each group of bars is one evaluated "
                    "task. Each bar is one ablated circuit. A solid bar is the "
                    "evaluated task's own circuit. A hatched bar is another task's "
                    "circuit. A grey bar with a cross means the ablated circuit is "
                    "empty, so the drop is zero by construction. No drop is averaged "
                    "over ablated circuits.",
    "lift": "Each grid row is one circuit size $K$. The sweep figures put the "
            "consensus threshold $P$ on the x-axis. The bar figures show one $P$ "
            "each, so between them they cover the whole grid.",
    "reuse": "Reuse is the fraction of the per-example circuit that survives the "
             "consensus threshold. It is capped at 100%.",
}


def write_markdown(built: dict, path: Path, K: int, p: int, models: list[str],
                   tasks: list[str], ks: list[int], ps: list[int]):
    lines = [
        "# Circuit plots by method and granularity",
        "",
        f"Every plot below covers {len(models)} models and {len(tasks)} tasks. "
        f"The models are {', '.join(MODEL_NAMES.get(m, m) for m in models)}. "
        f"The tasks are {', '.join(TASK_NAMES.get(t, t) for t in tasks)}. "
        "The source is the granularity-parity run, which is the only run that "
        "covers all four method-granularity cells with the same protocol. "
        "The analysis excludes Qwen3 8B and MMLU.",
        "",
        f"$K$ is the per-example circuit size, as a percent of all scored units at "
        f"the given granularity. $P$ is the consensus threshold: a unit enters the "
        f"shared circuit if it is in the per-example circuit for at least $P$% of "
        f"the examples. The grid is $K \\in \\{{{', '.join(map(str, ks))}\\}}$% and "
        f"$P \\in \\{{{', '.join(map(str, ps))}\\}}$%. Every figure family covers the "
        f"whole grid. The default operating point is $K={K}$% and $P={p}$%.",
        "",
        "Nothing here is averaged across methods, across granularities, or across "
        "donor tasks. Each section shows one method at one granularity. Tables give "
        "the numbers behind the figures. An extraction table has one row per model "
        "and task. A cross-task table has one row per model, ablated circuit, and "
        "evaluated task. Both put the $P$ grid in the columns. Each cell also has "
        "two CSV files that hold every column of the tidy data.",
        "",
    ]
    lines += ["- [Definitions](#definitions)"]
    lines += [f"- [{cell_title(c)}](#{anchor(cell_title(c))})" for c in CELLS]
    lines += ["", DEFINITIONS]
    for cell in CELLS:
        slug = cell_slug(cell)
        entry = built[slug]
        lines += [f"## {cell_title(cell)}", "",
                  f"Raw data: [`table_extraction__{slug}.csv`]"
                  f"(table_extraction__{slug}.csv) and "
                  f"[`table_cross_task__{slug}.csv`](table_cross_task__{slug}.csv).",
                  ""]
        for family, title in FAMILY_TITLES.items():
            lines += [f"### {title}", "", FAMILY_NOTES[family], ""]
            for figure in entry["figures"][family]:
                lines += [f"![{figure['file']}]({figure['file']})", "",
                          f"*{figure['caption']}*", ""]
            lines += ["#### Tables", ""] + entry["tables"][family]
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path,
                        default=Path("results/granularity_parity_analysis"))
    parser.add_argument("--out", type=Path,
                        default=Path("results/method_granularity_gallery"))
    parser.add_argument("--K", type=int, default=K_DEFAULT)
    parser.add_argument("--p", type=int, default=P_DEFAULT)
    args = parser.parse_args()

    extraction = pd.read_csv(args.analysis_dir / "extraction_tidy.csv")
    cross = pd.read_csv(args.analysis_dir / "cross_task_tidy.csv")
    # The drops the bar figures show, on the same scale as the necessity gap.
    extraction["drop_ablation_pp"] = (extraction.baseline_accuracy
                                      - extraction.ablation_accuracy) * 100
    extraction["drop_control_pp"] = (extraction.baseline_accuracy
                                     - extraction.control_accuracy) * 100

    models = sorted(extraction.model.unique())
    tasks = sorted(extraction.task.unique())
    ks = sorted(extraction.K.unique())
    ps = sorted(extraction.p.unique())

    plt = _setup()
    args.out.mkdir(parents=True, exist_ok=True)
    built = {}
    for cell in CELLS:
        slug = cell_slug(cell)
        ext = extraction[(extraction.method == cell[0])
                         & (extraction.granularity == cell[1])]
        cro = cross[(cross.method == cell[0]) & (cross.granularity == cell[1])]
        built[slug] = build_cell(plt, ext, cro, args.out, slug, models, tasks, ks, ps,
                                 args.K, args.p)
        print(f"[{slug}] "
              f"{sum(len(v) for v in built[slug]['figures'].values())} figures")

    write_markdown(built, args.out / "GALLERY.md", args.K, args.p, models, tasks,
                   ks, ps)
    total = sum(len(v) for cell in built.values() for v in cell["figures"].values())
    print(f"wrote {total} figures, 8 CSVs, and GALLERY.md to {args.out.resolve()}")


if __name__ == "__main__":
    main()
