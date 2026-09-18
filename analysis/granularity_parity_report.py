"""Render the consistency-and-specificity results as a standalone Markdown post.

Numbers for the method-by-granularity experiment are read from the tidy CSVs
written by :mod:`granularity_parity` and :mod:`circuit_overlap_chance`, so the
narrative cannot drift from the audited artifact. Figures from the earlier
six-model benchmark are referenced from the paper's figure directory, and the few
numbers quoted from that benchmark are collected in ``PRIOR`` below.
"""
from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path

import pandas as pd

CONFIG = ["method", "granularity"]
LABELS = {("eap_ig", "head_mlp"): "EAP-IG, heads + MLP layers",
          ("eap_ig", "neuron"): "EAP-IG, MLP neurons",
          ("relp", "head_mlp"): "RelP, heads + MLP layers",
          ("relp", "neuron"): "RelP, MLP neurons"}
ORDER = [("eap_ig", "head_mlp"), ("relp", "head_mlp"),
         ("eap_ig", "neuron"), ("relp", "neuron")]
METHOD_NAMES = {"eap_ig": "EAP-IG", "relp": "RelP", "eap": "EAP"}
GRAN_NAMES = {"head_mlp": "heads + MLP layers", "neuron": "MLP neurons"}
TASK_NAMES = {"addition": "Addition", "arc_challenge": "ARC (Chal.)",
              "arc_easy": "ARC (Easy)", "boolean": "Boolean", "ioi": "IOI",
              "mcqa": "CopyColors MCQA"}
PAPER_FIGS = ("../../paper/_COLM_2026__How_Much_Do_Circuits_Tell_Us__Measuring_"
              "the_Consistency_and_Specificity_of_Language_Model_Circuits/figures/")
CASE_FIGS = "../../new_plots/shared_head_attention/"

# Quoted from the six-model EAP benchmark reported in paper/integrated_results.md.
# That run predates the parity protocol and is not part of the audited artifact.
PRIOR = {
    "reuse_range": "40-70%",
    "llama_addition": "99%, and removing other tasks' circuits drops it by a mean of 99% as well",
    "llama_arc_c": "41% against 40%",
    "gemma_it_mcqa": "68% against 42%",
    "overlap_range": "0.46 to 0.89",
    "relp_spearman": "0.008 against EAP and 0.011 against EAP-IG",
}

# From the COLM rebuttal case study (analysis/shared_head_attention.py), on
# Llama-3.2-3B. Not part of the audited parity artifact -- a qualitative,
# single-model complement to it.
CASE_STUDY = {
    "intersection_k10": ("exactly 15 MLPs at layers 0 and 2-15, with no attention "
                         "heads, in both `Llama-3.2-3B` and `Llama-3.2-3B-Instruct`"),
    "l7h1": "50-69% of attention on `<|begin_of_text|>`, 11-22% on the last three tokens, "
            "and no more than 1% on any task-content token",
    "l8h17": "43-65% on `<|begin_of_text|>`, 19-38% on the final token itself, "
             "and no more than 2% on any task-content token",
}


def label(key) -> str:
    return LABELS.get(tuple(key), "/".join(key))


def reflow(text: str, width: int = 80) -> str:
    """Re-wrap prose paragraphs, leaving tables, code, and headings untouched."""
    out, buffer, in_code = [], [], False

    def flush():
        if buffer:
            out.extend(textwrap.wrap(" ".join(buffer), width=width,
                                     break_long_words=False, break_on_hyphens=False))
            buffer.clear()

    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("```"):
            flush()
            in_code = not in_code
            out.append(line)
        # Require a space after the bullet marker so a negative number opening a
        # line is not mistaken for a list item.
        elif (in_code or not stripped
              or stripped.startswith(("|", "#", "- ", "* ", "!", ">"))
              or re.match(r"\d+\. ", stripped)):
            flush()
            out.append(line)
        else:
            buffer.append(stripped)
    flush()
    return "\n".join(out)


def table(rows: list[list[str]], header: list[str], align: list[str] | None = None) -> str:
    align = align or ["---"] * len(header)
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    return f"| {' | '.join(header)} |\n| {' | '.join(align)} |\n{body}\n"


def fmt(x, nd=1) -> str:
    return "n/a" if pd.isna(x) else f"{x:,.{nd}f}"


def ci(row, scale: float = 1.0, nd: int = 2) -> str:
    return (f"{fmt(row['mean'] * scale, nd)} "
            f"`[{fmt(row['lo'] * scale, nd)}, {fmt(row['hi'] * scale, nd)}]`")


COLUMN_TITLES = {"p": "P", "K": "K", "model": "", "target": ""}


def pivot_table(df, value, index, columns, nd=1) -> str:
    piv = df.groupby(index + [columns])[value].mean().unstack(columns)
    piv = piv.reindex([k for k in ORDER if k in piv.index])
    prefix = COLUMN_TITLES.get(columns, columns)
    def head(c):
        if columns == "model":
            return str(c).split("/")[-1]
        return f"{prefix}={c}%" if prefix else str(c)
    header = ["Method and granularity"] + [head(c) for c in piv.columns]
    rows = [[label(k)] + [fmt(v, nd) for v in piv.loc[k]] for k in piv.index]
    return table(rows, header, ["---"] + ["---:"] * len(piv.columns))


def main_table(analysis: Path, ext_pt, diag_pt, boot_pt) -> tuple[str, str]:
    """The single parity table, as Markdown and as LaTeX for the paper."""
    overlap = None
    if (analysis / "overlap_summary.csv").exists():
        overlap = pd.read_csv(analysis / "overlap_summary.csv").set_index(CONFIG)
    header = ["Method", "Granularity", "Circuit size", "Reuse@P (%)", "Coverage",
              "Own (pp)", "Other (pp)", "Specificity gap (pp)", "Necessity"]
    if overlap is not None:
        header += ["Jaccard", "Jaccard / chance"]
    rows = []
    for key in ORDER:
        e = ext_pt[(ext_pt.method == key[0]) & (ext_pt.granularity == key[1])]
        d = diag_pt[(diag_pt.method == key[0]) & (diag_pt.granularity == key[1])]
        row = [METHOD_NAMES[key[0]], GRAN_NAMES[key[1]],
               fmt(e.circuit_size.mean()), fmt(e.reuse.mean()),
               f"{(e.circuit_size > 0).mean():.0%}",
               fmt(d.own_drop_pp.mean()), fmt(d.foreign_mean_drop_pp.mean()),
               ci(boot_pt.loc[(*key, "specificity_gap_pp")]),
               # Necessity is the pp-scale gap normalized by clean accuracy, the
               # same 0-1 quantity defined in the methodology and quoted in prose.
               ci(boot_pt.loc[(*key, "lift")], scale=0.01)]
        if overlap is not None:
            row += [fmt(overlap.loc[key, "observed_jaccard"], 3),
                    f"{fmt(overlap.loc[key, 'ratio_to_chance'], 0)}x"]
        rows.append(row)
    md = table(rows, header,
               ["---", "---", "---:", "---:", "---:", "---:", "---:", "---", "---"]
               + (["---:", "---:"] if overlap is not None else []))

    tex_names = {"eap_ig": r"\textsc{EAP-IG}", "relp": r"\textsc{RelP}"}
    tex_gran = {"head_mlp": r"\texttt{head\_mlp}", "neuron": r"neuron"}
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\setlength{\tabcolsep}{4pt}",
             r"\renewcommand{\arraystretch}{1.08}",
             r"\begin{tabular}{@{}llrrrrrcc@{}}", r"\toprule",
             r"Method & Granularity & Size & Reuse & Own & Other & Spec.\ gap & Jaccard & $\times$chance \\",
             r"\midrule"]
    for key in ORDER:
        e = ext_pt[(ext_pt.method == key[0]) & (ext_pt.granularity == key[1])]
        d = diag_pt[(diag_pt.method == key[0]) & (diag_pt.granularity == key[1])]
        b = boot_pt.loc[(*key, "specificity_gap_pp")]
        cells = [tex_names.get(key[0], key[0]), tex_gran.get(key[1], key[1]),
                 fmt(e.circuit_size.mean(), 0), fmt(e.reuse.mean()),
                 fmt(d.own_drop_pp.mean()), fmt(d.foreign_mean_drop_pp.mean()),
                 f"{fmt(b['mean'])} [{fmt(b['lo'])}, {fmt(b['hi'])}]"]
        if overlap is not None:
            cells += [fmt(overlap.loc[key, "observed_jaccard"], 3),
                      fmt(overlap.loc[key, "ratio_to_chance"], 0)]
        lines.append(" & ".join(cells) + r" \\")
        if key == ORDER[1]:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{\textbf{Attribution method crossed with component granularity.} "
              r"Reported at $K$=10\%, $p$=50\%, the only threshold at which every "
              r"configuration retains $\geq$80\% non-empty circuits. Specificity gap "
              r"is the own-task minus mean foreign-task accuracy drop, with 95\% "
              r"hierarchical bootstrap intervals. Jaccard is mean pairwise overlap "
              r"between different tasks' circuits, and $\times$chance divides it by "
              r"the overlap expected from independent draws of the same sizes.}",
              r"\label{tab:granularity_parity}", r"\end{table}"]
    return md, "\n".join(lines) + "\n"


def build(analysis: Path, figs: str, paper_figs: str) -> str:
    extraction = pd.read_csv(analysis / "extraction_tidy.csv")
    cross = pd.read_csv(analysis / "cross_task_tidy.csv")
    boot = pd.read_csv(analysis / "bootstrap_summary.csv")
    point = json.loads((analysis / "operating_point.json").read_text())["operating_point"]
    K, p = point["K"], point["p"]

    diag = cross[cross.donor == cross.target]
    ext_pt = extraction[(extraction.K == K) & (extraction.p == p)]
    diag_pt = diag[(diag.K == K) & (diag.p == p)]
    boot_pt = boot.set_index(CONFIG + ["metric"])
    md_table, tex_table = main_table(analysis, ext_pt, diag_pt, boot_pt)
    (analysis / "table_granularity_parity.tex").write_text(tex_table)

    ov = pd.read_csv(analysis / "overlap_summary.csv").set_index(CONFIG) \
        if (analysis / "overlap_summary.csv").exists() else None
    pairs = pd.read_csv(analysis / "overlap_pairs.csv") \
        if (analysis / "overlap_pairs.csv").exists() else None

    def E(key, col):
        return ext_pt[(ext_pt.method == key[0]) & (ext_pt.granularity == key[1])][col].mean()

    def D(key, col):
        return diag_pt[(diag_pt.method == key[0]) & (diag_pt.granularity == key[1])][col].mean()

    def G(key, col):
        return ov.loc[key, col] if ov is not None else float("nan")

    # Structural quantities read off the shared component sets themselves.
    def optional(name):
        path = analysis / name
        return pd.read_csv(path) if path.exists() else None

    sf, comp = optional("shared_fraction_pairs.csv"), optional("circuit_composition.csv")
    sel = optional("selective_ablation_tidy.csv")
    partition = {}
    if sel is not None:
        sel = sel.assign(selectivity=sel.target - sel.non_target)
        partition = {c: g for c, g in sel.groupby("condition")}
        wide = sel.pivot_table(index=["model", "task_a", "task_b"], columns="condition",
                               values="selectivity").dropna(
                                   subset=["residual_a", "random_control"])
        delta = wide.residual_a - wide.random_control
        boot = [delta.sample(len(delta), replace=True, random_state=s).mean()
                for s in range(2000)]
        partition["_delta"] = (delta.mean(), float(pd.Series(boot).quantile(.025)),
                               float(pd.Series(boot).quantile(.975)))
        by_task = sel[sel.condition == "residual_a"].groupby("task_a").selectivity.mean()
        partition["_best_task"] = (by_task.idxmax(), by_task.max())

    def P(condition, col):
        return partition[condition][col].mean() if condition in partition else float("nan")

    def S(key):
        """Mean fraction of a task's circuit that another task also uses."""
        if sf is None:
            return float("nan")
        return sf[(sf.method == key[0]) & (sf.granularity == key[1])].shared_fraction.mean()

    def M(key):
        """MLP share of a heads-and-MLP shared circuit."""
        if comp is None:
            return float("nan")
        return comp[(comp.method == key[0]) & (comp.granularity == key[1])].mlp_fraction.mean()

    def B(key, metric):
        return ci(boot_pt.loc[(*key, metric)])

    DUP, UNREL = ("arc_challenge", "arc_easy"), ("addition", "ioi")

    def pair_stats(key) -> dict:
        if pairs is None:
            return {}
        sub = pairs[(pairs.method == key[0]) & (pairs.granularity == key[1])]
        is_dup = (sub.task_a == DUP[0]) & (sub.task_b == DUP[1])
        is_un = (sub.task_a == UNREL[0]) & (sub.task_b == UNREL[1])
        seps = []
        for _, ms in sub.groupby("model"):
            d = ms[(ms.task_a == DUP[0]) & (ms.task_b == DUP[1])].observed.mean()
            o = ms[~((ms.task_a == DUP[0]) & (ms.task_b == DUP[1]))].observed.median()
            if o:
                seps.append(d / o)
        return {"dup": sub[is_dup].observed.mean(), "other": sub[~is_dup].observed.mean(),
                "unrel": sub[is_un].observed.mean(),
                "lo": min(seps) if seps else float("nan"),
                "hi": max(seps) if seps else float("nan")}

    hm, nu, nu_r = (pair_stats(("eap_ig", "head_mlp")), pair_stats(("eap_ig", "neuron")),
                    pair_stats(("relp", "neuron")))
    EIH, EIN = ("eap_ig", "head_mlp"), ("eap_ig", "neuron")
    RPH, RPN = ("relp", "head_mlp"), ("relp", "neuron")

    def fig(stem, caption, prefix=None):
        prefix = figs if prefix is None else prefix
        if prefix is figs and not (analysis / f"{stem}.png").exists():
            return ""
        return f"\n![{caption}]({prefix}{stem}.png)\n\n*{caption}.*\n"

    gap_by_model = diag_pt.groupby(CONFIG + ["model"]).specificity_gap_pp.mean()
    nu_lo, nu_hi = gap_by_model.loc[EIN].min(), gap_by_model.loc[EIN].max()
    hm_lo, hm_hi = gap_by_model.loc[EIH].min(), gap_by_model.loc[EIH].max()
    worst = (extraction[extraction.K == K].groupby(["model"] + CONFIG + ["p"])
             .circuit_size.agg(lambda x: (x > 0).mean()).groupby("p").min())
    necessity = {k: E(k, 'lift') / 100 for k in ORDER}
    necessity = {k: E(k, 'lift') / 100 for k in ORDER}
    out = [f"""# How Much Do Circuits Tell Us? Measuring the Consistency and Specificity of Language Model Circuits

*Michael Li, Nishant Subramani -- Language Technologies Institute, Carnegie
Mellon University*

**Abstract.** The circuits framework in mechanistic interpretability aims to
identify causally important sparse subgraphs of model components, typically
evaluated by measuring *necessity* and *sufficiency*. We study two further
properties. *Consistency* asks whether the same components recur across
instances of a task. *Specificity* asks whether a task's circuit is
distinguishable from the circuits of other tasks. We measure both across six
tasks and seven models, using two levels of granularity -- attention heads and
MLP layers, and individual MLP neurons. Circuits are highly consistent over
attention heads and MLP layers, where the shared components account for
{fmt(E(EIH, 'reuse'))}% of an average per-example circuit. Over individual
neurons, circuits are far less consistent, and reuse falls to
{fmt(E(EIN, 'reuse'))}%. We observe the
opposite pattern for specificity. Over heads and MLP layers, ablating one
task's circuit damages other tasks almost as much as its own. Conversely,
specificity is much stronger over individual neurons, where the same ablation
procedure separates own-task from other-task damage by roughly
{fmt(D(EIN, 'specificity_gap_pp'), 0)} percentage points. This granularity
effect traces back to how circuits overlap. Overlap between heads-and-MLP
circuits is uniformly high across every task pair, whether or not the tasks
are related. Overlap between neuron circuits instead tracks task similarity.
Once we correct for the size of each component pool, neuron circuits turn out
to be more shared than heads-and-MLP circuits, not less. The difference
between the two granularities lies in what is shared, not in how much is
shared. A case study of the two attention heads present in every task's
shared circuit in `Llama-3.2-3B` confirms this mechanism directly. Both are
generic attention-sink heads rather than task-specific machinery. Together,
these results show that neither consistency nor specificity is a fixed
property of a model. Both depend on the granularity at which a circuit is
described.

## 1. Introduction

Neural networks are infamously black box; even when we can elicit strong
performance on a task, it is unclear which internal computations are
responsible. The field of mechanistic interpretability seeks to
reverse-engineer the internal computations of neural networks by identifying
*circuits*: sparse subgraphs of model components that are causally responsible
for a particular behavior. A growing body of work has developed methods for
extracting such circuits and evaluating their *necessity* (removing the
circuit should degrade performance) and *sufficiency* (the circuit alone
should reproduce the behavior).

We argue that there are two additional properties that are crucial to
consider. First, circuits should be *consistent*: if a circuit truly captures
how a model solves a task, the same components should recur for different
instances of that task. Second, circuits should be *specific*: a task's
circuit should be meaningfully different from the circuits of unrelated tasks.
Without consistency, a circuit is an artifact of a particular input rather
than a description of the model's algorithm. Without specificity, a circuit is
not task-specific, limiting its utility for understanding or intervention.

Both properties presuppose an answer to a prior question, namely what counts as
a component. That choice is really a choice of the level of abstraction at which
we describe a model, and circuit discovery has largely settled on one level,
attention heads and MLP layers, without testing whether the properties we want
hold there. We test consistency and specificity at scale, and we test them at
two levels of abstraction together, so that a finding about either property can
be separated from a finding about the representation it was measured in.
We extract per-example circuits for \\(n\\)=1000 examples across six tasks
spanning algorithmic reasoning (`Addition`, `Boolean Logic`), information
retrieval from context (`IOI`, `CopyColors MCQA`), and knowledge-intensive
benchmarks (`ARC Easy`, `ARC Challenge`); seven models from four architecture
families (`Gemma 2`, `Llama 3.2`, `Qwen3`, `OLMo-2`); and two granularities,
attention heads together with MLP layers and individual MLP neurons. We find
the following:

1. **Circuits are consistent at both granularities, far more so over heads and
MLP layers.** A substantial fraction of each per-example circuit is drawn from
a shared set of components, and ablating that shared set causes large
accuracy loss compared to a capacity-matched random ablation -- the shared
components are causally important, not merely high-scoring artifacts of the
attribution method. Reuse is {fmt(E(EIH, 'reuse'))}% over heads and MLP layers
against {fmt(E(EIN, 'reuse'))}% over individual neurons.
2. **Whether a task's circuit is specific to it depends on the granularity at
which we look.** Over heads and MLP layers, removing task \\(A\\)'s circuit
damages task \\(B\\) almost as much as it damages \\(A\\); the own-task and
other-task drops differ by only {fmt(D(EIH, 'specificity_gap_pp'), 0)} points.
Over individual neurons, the same procedure separates own from other by
roughly {fmt(D(EIN, 'specificity_gap_pp'), 0)} points.
3. **This is explained by how circuits overlap, and the two granularities
differ in kind, not just in degree.** Overlap between heads-and-MLP circuits
is uniformly high across every task pair, related or not. Overlap between
neuron circuits tracks task similarity, so near-duplicate tasks share much
more than unrelated ones. Relative to a chance baseline matched to each component
pool, neuron circuits are in fact the more shared of the two
({fmt(G(EIN, 'ratio_to_chance'), 0)}x chance against
{fmt(G(EIH, 'ratio_to_chance'), 0)}x), so the granularity effect is not a
matter of neurons being less shared -- it is that heads-and-MLP overlap is
uninformative about task relatedness while neuron overlap is not.
4. **A case study grounds this mechanism in an actual component.** The two
attention heads present in every task's shared circuit in `Llama-3.2-3B` are
both generic attention-sink heads. One directs the large majority of its
attention to the beginning-of-text token, and the other splits attention
between that token and the current position. Neither attends meaningfully to
any task-specific content. This is a concrete instance of the uniform,
task-invariant sharing that the overlap statistics describe in aggregate.

These findings suggest that circuit discovery over attention heads and MLP
layers primarily identifies general-purpose model infrastructure rather than
task-specific mechanisms, and that this is a property of that granularity
rather than of the models, since the same models, described at the level of
individual neurons, show overlap that tracks task structure. Neither level we
study gives both properties at once, which complicates the question of what
level of abstraction circuit analysis should target. We discuss
implications for applications that assume circuit-level modularity, including
model editing and safety interventions, while noting important limitations of
our analysis for these settings.

## 2. Methodology

### 2.1 Tasks, Models, and Component Granularities

We evaluate on six tasks spanning algorithmic reasoning (`Addition`, `Boolean
Logic`), information retrieval from context (`IOI`, `CopyColors MCQA`), and
knowledge-intensive benchmarks (`ARC Easy`, `ARC Challenge`). We study seven
models from four architecture families: `Gemma 2` (2B, 2B IT), `Llama 3.2`
(3B, 3B Instruct), `Qwen3` (4B, 8B), and `OLMo-2-1B`, which is used for
pretraining dynamics analysis.

Nodes in the computation graph are the model's computational units. We measure
consistency and specificity at two choices of node. The head-and-MLP
granularity treats each attention head and each MLP layer as one component.
The neuron granularity treats each MLP neuron -- the pre-down-projection
hidden activation within an MLP block -- as a component, following Arora et
al., who show that this granularity supports circuits as sparse and faithful
as those built from SAE features; it scores MLP neurons only, so it cannot
describe attention.

We use three attribution methods. Edge Attribution Patching (EAP) approximates
causal effects using gradient information and defines the circuit from the
top-\\(K\\) components by absolute attribution score. EAP-IG averages
gradients over interpolation points along the path to the corrupted input, and
RelP replaces the backward pass with LRP-modified rules. EAP scores edges over
a graph defined only for heads and MLP layers and cannot be evaluated at the
neuron granularity, so that granularity is covered by EAP-IG and RelP.

### 2.2 Extracting and Evaluating Shared Circuits

For each task \\(T\\) we use a dataset of \\(n\\)=1000 examples and a disjoint
held-out evaluation set. Let \\(\\mathcal{{C}}=(\\mathcal{{V}},
\\mathcal{{E}})\\) denote the model's computation graph. For each example we
extract a per-input circuit \\(\\mathcal{{C}}_i \\subseteq \\mathcal{{C}}\\),
defined as the subgraph spanned by the top-\\(K\\)% of components by absolute
attribution score, and sweep \\(K \\in \\{{1,5,10,20,30\\}}\\). Given the
per-input circuits, the **shared component set** \\(S_P\\) contains all
components appearing in at least \\(P\\) of them. We define **reuse@\\(P\\)**
as the mean fraction of a per-input circuit overlapping with the shared set.

To test whether the shared components are causally important, we ablate (zero
out) \\(S_P\\) and measure accuracy, compared against a **capacity-conserved
control** (C\\(^3\\)), a uniformly random subset of \\(\\mathcal{{C}}
\\setminus S_P\\) matching \\(S_P\\) in size. Since both ablations are
capacity-matched, any additional degradation from ablating \\(S_P\\) can
largely be attributed to the functional role of those components rather than
their number. **Necessity** is this difference, normalized by clean accuracy.

Because the neuron granularity leaves \\(S_P\\) empty at strict consensus
thresholds, we sweep \\(P \\in \\{{{','.join(str(v) for v in sorted(extraction.p.unique()))}\\}}\\)% and report
**coverage**, the fraction of tasks with a non-empty \\(S_P\\), alongside
every comparison.

### 2.3 Cross-Task Experiments

Necessity tells us whether the shared components matter for a given task, not
whether they are specific to it -- a component could matter for task
\\(A\\) simply because it matters for every task. We define
\\(\\Delta_A^B\\) as the accuracy drop on task \\(A\\) from ablating task
\\(B\\)'s shared circuit, and compare \\(\\Delta_A^A\\) ("Own") against the
mean of \\(\\Delta_A^B\\) over \\(B \\neq A\\) ("Other"). To localize where
task-specific signal resides, we also partition the union
\\(\\mathcal{{C}}_A \\cup \\mathcal{{C}}_B\\) for a pair of tasks into the
**shared core**, the **\\(A\\)-only** set, and the **\\(B\\)-only** set, and
ablate each independently.

We measure overlap between different tasks' shared circuits with Jaccard
similarity. Because the two granularities offer component pools that differ
by three orders of magnitude, raw overlap is not comparable between them. We
therefore report it alongside the value expected from two independent uniform
draws of the observed circuit sizes, and their ratio.

The cross-granularity comparison holds models, datasets, thresholds, and the
100 target examples per evaluation fixed. It covers five of the seven models,
excluding `Qwen3-8B` because its extraction logs contain OOM-skipped examples
and `OLMo-2-1B` because we use it only for pretraining dynamics. This gives
{len(cross) // 36:,} cross-task matrices, each a complete 6x6 donor-by-target
grid, for {len(cross):,} cells with no missing drops. We summarize at \\(K\\)={K}%,
\\(P\\)={p}%, the only setting at which every model, method, and granularity
retains at least {point['min_coverage_criterion']:.0%} non-empty circuits
([§8](#sec-limitations) reports the full sweep). Intervals are 95%
hierarchical paired bootstrap intervals, resampling models and then tasks
within models, with 10,000 resamples at seed 42.

## 3. Consistency Is Strong Over Heads and MLP Layers, Weak Over Neurons

Circuits recur strongly over heads and MLP layers, and only weakly over
individual neurons. Over heads and MLP layers, most task-model combinations show
{PRIOR['reuse_range']} reuse@97% at \\(K\\)=10%, meaning roughly half or more
of any individual example's circuit is drawn from a set of components shared
by nearly every example; `CopyColors MCQA` and `Boolean` tend toward the
higher end, `ARC` and `IOI` more moderate. At the operating point of the
cross-granularity comparison, reuse@{p}% is {fmt(E(EIH, 'reuse'))}% for EAP-IG
and {fmt(E(RPH, 'reuse'))}% for RelP over heads and MLP layers, against only
{fmt(E(EIN, 'reuse'))}% and {fmt(E(RPN, 'reuse'))}% over neurons -- a neuron
shared set at this threshold is roughly thirty to sixty times larger in
component count and far more weakly recurring.

{fig('consistency_by_task', f'Within-task consistency and necessity at both granularities, one panel column per task. Top: reuse@{p}% against circuit size K. Bottom: the necessity gap, the excess accuracy drop from ablating the shared circuit over a capacity-matched random ablation. Lines are means over the five models of the cross-granularity comparison and bands their interquartile range. The two granularities separate on reuse in every task and at every K, and do not separate on necessity')}
Reuse alone does not establish that the shared components matter -- they
could simply be high-activation components that get ranked highly without
playing a functional role. Necessity is positive at both granularities,
however. It reaches {fmt(necessity[EIH], 2)} for EAP-IG and
{fmt(necessity[RPH], 2)} for RelP over heads and MLP layers, and
{fmt(necessity[EIN], 2)} and {fmt(necessity[RPN], 2)} respectively over
neurons. Neither granularity is recovering components that do not matter. They
differ in how often the same components recur, not in whether they are doing
work.

{fig('within_task_reuse_vs_p', f'Reuse@P against the consensus threshold P at K=10%, for each attribution method and granularity. Each point averages over the five models and six tasks of the cross-granularity comparison')}
## 4. Specificity Depends on the Granularity of Analysis

If circuits are task-specific, removing task \\(A\\)'s circuit should damage
\\(A\\) far more than task \\(B\\). Whether this holds depends entirely on
which granularity we ask about.

Over heads and MLP layers, it mostly does not. In `Llama-3.2-3B`, removing the
`Addition` circuit drops `Addition` by {PRIOR['llama_addition']}. For `ARC
(Challenge)`, the own-circuit and other-circuit drops are
{PRIOR['llama_arc_c']}. The `Gemma` family differentiates a little more --
`CopyColors MCQA` in `Gemma 2 2B IT` has {PRIOR['gemma_it_mcqa']} -- but the
general pattern is that own and other are close. Aggregating over models, the
own drop is {fmt(D(EIH, 'own_drop_pp'))} points against
{fmt(D(EIH, 'foreign_mean_drop_pp'))} for other circuits, a gap of
{B(EIH, 'specificity_gap_pp')} points whose interval covers zero.

{fig('own_vs_other_paired', f'Own-circuit vs. other-circuit accuracy drop at K={K}%, P={p}%, with one segment per model-task pair and one panel per attribution method and granularity. Own is the drop from removing the circuit belonging to that task; Other is the mean drop from removing every other task circuit. Segments are coloured where own exceeds other and grey otherwise. The bold line is the mean. Segments are flat over heads and MLP layers and rise steeply over MLP neurons')}
Over individual neurons, running the identical experiment -- same models,
data, thresholds, and sample counts -- own and other separate, with a gap of
{B(EIN, 'specificity_gap_pp')} points. The separation opens from one side. The
mean drop from other tasks' circuits falls from
{fmt(D(EIH, 'foreign_mean_drop_pp'))} to {fmt(D(EIN, 'foreign_mean_drop_pp'))}
points, while the own drop falls only from {fmt(D(EIH, 'own_drop_pp'))} to
{fmt(D(EIN, 'own_drop_pp'))}. Heads-and-MLP circuits are large enough relative
to the pool of available components that ablating any one of them damages
nearly any task; neuron circuits are not.

{md_table}
*Table 1. Consistency and specificity at both granularities, for each
attribution method, at \\(K\\)={K}%, \\(P\\)={p}%. All values average over the
five models and six tasks of the cross-granularity comparison. Circuit size is
the number of components in \\(S_P\\), and coverage is the fraction of
model-task pairs for which \\(S_P\\) is non-empty. Own and Other are the
accuracy drops from ablating a task's own circuit and the mean over the other
five tasks' circuits. Necessity is the accuracy drop from ablating \\(S_P\\)
minus the drop from a capacity-matched random control, normalized by clean
accuracy. Jaccard is mean pairwise overlap between different tasks' shared
circuits, and the last column divides it by the overlap expected from
independent draws of the same sizes. Bracketed ranges are 95% hierarchical
bootstrap intervals.*

{fig('own_vs_other_by_granularity', f'Own-circuit vs. other-circuit accuracy drop per task at K={K}%, P={p}%, one panel per attribution method and granularity. Bars average over the five models of the cross-granularity comparison. Own and other track each other over heads and MLP layers and separate over MLP neurons')}
The separation holds in every model rather than a subset. Both neuron
configurations exceed both heads-and-MLP configurations across all five models
of the cross-granularity comparison, with EAP-IG neuron gaps from {fmt(nu_lo)}
to {fmt(nu_hi)} points against {fmt(hm_lo)} to {fmt(hm_hi)} over heads and MLP
layers. It also holds under both attribution methods. RelP correlates only
weakly with the gradient-based methods on raw scores (mean cellwise Spearman
{PRIOR['relp_spearman']}) but agrees with EAP-IG on specificity at both
granularities. Method does change the reliability of the recovered circuit --
over heads and MLP layers, RelP's reuse ({fmt(E(RPH, 'reuse'))}%) and
necessity ({fmt(necessity[RPH], 2)}) both trail EAP-IG's
({fmt(E(EIH, 'reuse'))}%, {fmt(necessity[EIH], 2)}) -- but not the specificity
verdict.

{fig('specificity_gap_by_model', f'Specificity gap per model at K={K}%, P={p}%, averaged over the six tasks. Both neuron configurations exceed both heads-and-MLP configurations in all five models')}
## 5. Why Granularity Changes the Overlap

The specificity result is a downstream consequence of overlap. Different
tasks' heads-and-MLP circuits are built from largely the same components. Under EAP in
`Llama-3.2-3B` and `Qwen3-4B`, pairwise overlap at \\(K\\)=10% typically ranges
from {PRIOR['overlap_range']},
an order of magnitude above the roughly 5% expected of two random circuits of
that size. Ablating any task's circuit strips away much the same machinery,
which is why own and other are close.

Raw overlap is not comparable across granularities, though, and taking it at
face value would make the specificity gap look like simple arithmetic. Heads
and MLP layers offer only about {fmt(G(EIH, 'pool'), 0)} components per model,
neurons about {fmt(G(EIN, 'pool'), 0)}, so two unrelated circuits collide far
less often at the neuron granularity by chance alone. Correcting for this --
comparing observed overlap to the value expected from independent draws of
the same sizes -- reverses the naive reading. Heads-and-MLP circuits sit
{fmt(G(EIH, 'ratio_to_chance'), 0)}x above chance and neuron circuits
{fmt(G(EIN, 'ratio_to_chance'), 0)}x above chance, in every model. Neuron
circuits are the more shared of the two; the gap between them is not about how
much is shared, but about what is shared.

{fig('overlap_vs_chance', f'Cross-task circuit overlap at K={K}%, P={p}%. Left: observed Jaccard overlap between different tasks\u2019 shared circuits against the chance baseline matched to each component pool, on a log scale. Right: their ratio. Both average over the five models and all 15 task pairs')}
Specifically, heads-and-MLP overlap does not track task similarity, and
neuron overlap does. `ARC Easy` and `ARC Challenge` -- near-duplicate tasks --
share {fmt(hm['dup'], 3)} of their heads-and-MLP circuits under EAP-IG, and
the average unrelated pair shares {fmt(hm['other'], 3)}: `Addition` and `IOI`,
which have little in common, still share {fmt(hm['unrel'], 3)}. Per-model
separation between the near-duplicate pair and the median pair is only
{fmt(hm['lo'])} to {fmt(hm['hi'])}x. Over neurons the same comparison
separates cleanly: the near-duplicate pair shares {fmt(nu['dup'], 3)} against
{fmt(nu['other'], 3)} for the average pair, a factor of roughly
{fmt(nu['dup'] / nu['other'], 0)}, with per-model separation from
{fmt(nu['lo'], 0)} to {fmt(nu['hi'], 0)}x under EAP-IG and
{fmt(nu_r['lo'], 0)} to {fmt(nu_r['hi'], 0)}x under RelP. `Addition` and `IOI`
fall to {fmt(nu['unrel'], 3)}, indistinguishable from chance.

{fig('overlap_task_pairs', f'Pairwise Jaccard overlap between task circuits at K={K}%, P={p}%, averaged over the five models, with one panel per attribution method and granularity on a shared colour scale. Overlap over heads and MLP layers is uniformly high whether or not tasks are related; over MLP neurons it is near zero except between ARC Easy and ARC Challenge')}
This is why the two granularities give opposite specificity verdicts despite
neuron circuits being the more shared. Ablation-based specificity responds to
the *fraction* of a circuit that a foreign ablation removes, not to symmetric
overlap, and that fraction is what separates the two granularities. Ablating
another task's circuit takes {fmt(S(EIH) * 100, 0)}% of a heads-and-MLP circuit
under EAP-IG and {fmt(S(RPH) * 100, 0)}% under RelP, against
{fmt(S(EIN) * 100, 0)}% and {fmt(S(RPN) * 100, 0)}% over neurons. Heads-and-MLP
circuits are small enough that a uniformly-shared component set is most of the
circuit, while neuron circuits are large enough that a task-tracking pattern of
overlap still leaves most of the circuit unshared with any given other task.
Across the twenty configuration-model points, the shared fraction and the
specificity gap are strongly anti-correlated, which is the mechanism the
ablation result rests on.

{fig('shared_fraction_by_granularity', f'How much of a task circuit a foreign ablation removes, at K={K}%, P={p}%. Left: the fraction |C_A n C_B| / |C_A| over every ordered task pair and model, with the configuration mean marked. Right: the same quantity averaged per model against the measured specificity gap, one point per attribution method, granularity, and model')}
## 6. Case Study: The Shared Core Is Generic Attention-Sink Machinery

The overlap statistics describe what is shared in aggregate; here we name an
actual instance of it. Intersecting the per-task shared circuits over heads
and MLP layers at \\(K\\)=10%, \\(P\\)=100% gives {CASE_STUDY['intersection_k10']}.
At \\(K\\)=30%, two attention heads enter this
six-way intersection in `Llama-3.2-3B`: `L7H1` and `L8H17`.

To test whether these two heads implement the same function across tasks or
are simply co-listed, we captured their attention weights on 16 prompts per
task and aggregated attention from the final query position by template role
(beginning-of-text, recent positions, the final token, and task-specific
roles such as digit, name, label, or question). Both heads behave identically
across all six tasks. `L7H1` is a beginning-of-text sink head, placing
{CASE_STUDY['l7h1']}. `L8H17` splits attention between the beginning-of-text
token and the current position, placing {CASE_STUDY['l8h17']}.

{fig('summary_L7H1', 'Mean attention from the final query position for head L7H1 in Llama-3.2-3B, by template role (columns) and task (rows), over 16 prompts per task. Blank cells are roles that do not occur in that task. Nearly all attention lands on the beginning-of-text token rather than on any task content', CASE_FIGS)}
{fig('summary_L8H17', 'Mean attention from the final query position for head L8H17 in Llama-3.2-3B, by template role (columns) and task (rows), over 16 prompts per task. Attention splits between the beginning-of-text token and the final token itself, again with no task-content attention', CASE_FIGS)}
Neither head implements a task-specific algorithm -- there is no name-moving
on `IOI`, no digit-copying on `Addition`, no choice-selection on `CopyColors
MCQA`. Both are general-purpose attention-sink or position-anchor machinery
that every task reuses at the head-and-MLP granularity. This is exactly the
kind of component the cross-task overlap statistics in §5 are dominated by,
made concrete. What is shared is not a compressed reasoning routine but generic
bookkeeping, present because every task's forward pass needs it rather than
because it does anything about `Addition` or `IOI` in particular. This
analysis is necessarily specific to the head-and-MLP granularity -- the
neuron granularity, as defined here, does not include attention at all, so it
cannot describe this kind of component directly.

## 7. The Task-Only Remainder Does Not Recover Specificity

If the shared core is what makes heads-and-MLP circuits look unspecific, the
natural repair is to subtract it and ask whether the remainder is
task-specific. It is not. Partitioning \\(\\mathcal{{C}}_A \\cup
\\mathcal{{C}}_B\\) into the shared core, the \\(A\\)-only set, and the
\\(B\\)-only set, and ablating each in turn over all six models, the shared
core carries most of the damage: it drops the target task by
{fmt(P('shared_core', 'target'))}% and other tasks by
{fmt(P('shared_core', 'non_target'))}% relative to clean accuracy, from
{fmt(P('shared_core', 'size'), 0)} components against
{fmt(P('residual_a', 'size'), 0)} in the \\(A\\)-only set. Those components are
genuinely important rather than merely numerous, and they are just as important
to every other task.

The \\(A\\)-only remainder is where any task-specific signal would have to
live, and it does not survive its own control. Read naively it looks selective
on some tasks -- ablating it costs `{TASK_NAMES.get(partition['_best_task'][0], partition['_best_task'][0])}`
{fmt(partition['_best_task'][1])} points more than it costs other tasks, the
largest such margin of any task -- but that comparison is confounded by how fragile a task
is, since a size-matched random ablation looks just as selective on the same
tasks. The experiment samples exactly that control, matched to
\\(|\\mathcal{{C}}_A \\setminus \\mathcal{{C}}_B|\\). Against it, the
\\(A\\)-only set is selective by {fmt(partition['_delta'][0])} points
`[{fmt(partition['_delta'][1])}, {fmt(partition['_delta'][2])}]`, an interval
covering zero with a point estimate on the wrong side, and no model is
positive. Removing the shared core does not leave task-specific machinery
behind at this granularity; it leaves a set that behaves like a random set of
the same size. This is the same conclusion §4 reaches by ablating whole
circuits, reached instead by dissecting one.

{fig('selective_ablation_summary', f'Selective ablation of the circuit partition at K={K}%, P=100% over heads and MLP layers, under EAP, over all six models. Left: relative accuracy drop from ablating each group, with solid bars the task the circuit came from and hatched bars the mean over other tasks; dots are per-model means. Right: selectivity, the target minus other-task drop, for the A-only set against the random control the experiment size-matches to it, with one point per model and the paired difference labelled')}
This analysis exists only at the head-and-MLP granularity. The neuron
granularity would need a fresh ablation sweep over the same partition, which we
have not run, so §7 alone does not carry the cross-granularity comparison that
the rest of the report does.

## 8. Limitations
<a id="sec-limitations"></a>

**Coverage bounds where the two granularities can be compared.** The neuron
granularity leaves \\(S_P\\) empty as \\(P\\) tightens, unevenly across
configurations.

{pivot_table(extraction[extraction.K == K].assign(nonempty=lambda d: (d.circuit_size > 0).astype(float)), 'nonempty', CONFIG, 'p', 2)}
*Fraction of the 30 model-task pairs with a non-empty shared circuit at
\\(K\\)={K}%, by consensus threshold.*

Taking the worst model-method-granularity cell at each threshold, coverage is
{worst.loc[p]:.2f} at \\(P\\)={p}% and {worst.loc[75]:.2f} at \\(P\\)=75%. We
summarize at \\(P\\)={p}% because it is the only setting where every
configuration retains at least {point['min_coverage_criterion']:.0%}
non-empty circuits; above it, the neuron specificity gap decays toward the
heads-and-MLP range, but the number of contributing model-task pairs falls
from 30 to between 4 and 15 by \\(P\\)=98%, on the tasks whose attributions
happen to be most stable. We read this as underpowered, not as evidence that
the granularity effect reverses.

{fig('nonempty_coverage_vs_p', f'Fraction of tasks with a non-empty shared circuit against the consensus threshold P at K={K}%, averaged over the five models and six tasks. The dashed line marks the 80% coverage criterion and the dotted line marks the operating point P={p}%')}
**The result is stable in circuit size above the sparsest setting.** Both
neuron configurations and EAP-IG over heads and MLP layers are stable across
\\(K\\). RelP over heads and MLP layers is the exception, reaching the neuron
range at \\(K\\)=1% and decaying steadily as \\(K\\) grows, so the separation
by granularity is clean for \\(K \\geq 5\\)% but not at \\(K\\)=1%.

{fig('k_sweep_panels', f'Reuse@P, mean circuit size, necessity gap in percentage points, and normalized lift against circuit size K at P={p}%, averaged over the five models and six tasks, for each attribution method and granularity')}
**Consistency and specificity trade off, so neither granularity is simply
better.** Across the twenty configuration-model points the two are strongly
anti-correlated (Spearman -0.75, Pearson -0.86). Neuron shared sets recur in
under 10% of per-input circuits, so their specificity should not be read as
evidence that they are the more faithful object -- no configuration we study
has both properties at once.

{fig('specificity_vs_reuse', f'Specificity gap against reuse@P at K={K}%, P={p}%. Each point is one attribution method, granularity, and model, averaged over the six tasks, for 20 points in total')}
**The neuron granularity drops attention entirely,** following Arora et al.,
so the two granularities differ in which parts of the network they can
describe as well as in resolution -- a confound we cannot remove without a
neuron-level treatment of attention, and the reason the case study in §6 is
heads-and-MLP-only by construction.

## 9. Discussion

**Why do circuits overlap?** Shared circuits over heads and MLP layers draw
disproportionately on MLP layers -- {fmt(M(EIH) * 100, 0)}% of the shared set
under EAP-IG and {fmt(M(RPH) * 100, 0)}% under RelP, in models where MLP layers
are a few percent of the available components -- so small circuits are
constrained to draw from a small shared pool. Beyond pool size, these layers
plausibly perform
general-purpose operations -- storing parametric knowledge, mapping
tokens into a useful representational space, adjusting positional information
-- that every task's forward pass depends on. The chance-corrected results add
a constraint on this account. Sharing is not merely an artifact of having few
components to choose from, since it survives at
{fmt(G(EIN, 'ratio_to_chance'), 0)}x chance over a pool of
{fmt(G(EIN, 'pool'), 0)} neurons. The case study in §6 also gives the
explanation a name, in two heads that do positional bookkeeping every task
needs rather than reasoning any task is defined by.

**Polysemanticity and superposition.** Individual heads and MLP layers
inevitably serve multiple roles, since networks represent more features than
they have dimensions. At the granularity of an entire layer these roles
cannot be disentangled, so circuits for different tasks overlap even where
the underlying feature-level computations are distinct. Our neuron-granularity
results are consistent with this account and put a number on it, since the
same procedure at finer resolution recovers overlap that tracks task similarity,
which is what one would expect if heads-and-MLP components were mixing
distinct feature-level roles.

**What this means for circuit-level analysis.** The value of per-task circuit
discovery relies on specificity. If task \\(A\\)'s circuit largely matches
task \\(B\\)'s, finding \\(A\\)'s circuit reveals more about what the model
needs to function at all than about what it does for \\(A\\) specifically.
That is the accurate description over heads and MLP layers, and the limit is
one of resolution rather than of the models, since the same models show
similarity-tracking overlap at the neuron granularity. Recovering task-specific
structure reliably may therefore require finer-grained units of analysis --
the neuron granularity, sparse feature circuits, or attribution methods that
score components by attribution to one task *relative to* others rather than
in absolute terms.

**What level of abstraction should circuit analysis target?** The two levels we
study each satisfy one of the properties we set out to measure and fail the
other. Over attention heads and MLP layers, circuits are consistent, with the
shared set covering {fmt(E(EIH, 'reuse'))}% of an average per-example circuit,
but not specific, with a gap of {fmt(D(EIH, 'specificity_gap_pp'))} points whose
interval covers zero. Over MLP neurons, circuits are specific, at
{fmt(D(EIN, 'specificity_gap_pp'))} points, but barely recur at all, with the
shared set covering {fmt(E(EIN, 'reuse'))}%. Necessity and
sufficiency, the properties the field already checks, are satisfied at both
levels and so cannot adjudicate between them. On the two properties that can,
neither level is the right answer, and the honest conclusion is that the level
of abstraction for reasoning about model behavior remains unsettled.

One reply is that neither level is the interesting one, and that circuits should
instead be written over learned feature bases such as sparse autoencoder
features or cross-layer transcoders, which are designed to disentangle the
superposition that makes coarse components polysemantic. Our results bear on
this more directly than they might appear to. Arora et al. show that circuits
traced over MLP neurons are as sparse and as faithful as those obtained from
learned feature bases, and replicate a cross-layer transcoder case study in the
neuron basis. If the neuron basis is competitive with learned features on
sparsity and faithfulness, then the consistency cost we measure at that level is
a cost a disentangled basis would plausibly also pay, and it is not obviously
resolved by moving to features. We would not treat that as settled either, and
measuring consistency and specificity directly over a learned feature basis is
the natural next test.

**Reporting practice.** A raw overlap number cannot distinguish a granularity
that finds uniform sharing from one that finds structured sharing, and a
specificity gap measured where most shared sets are empty is not comparable to
one measured where none are. Both of our cross-granularity results depend on
controls of this kind. We suggest reporting specificity alongside a chance
baseline and a coverage figure.

**Reuse as a feature, not a bug.** Treating non-specificity as a limitation
implicitly assumes task circuits *ought* to be disjoint. High reuse may
instead be a desirable property. Models plausibly develop small, reusable
computational motifs, the attention-sink heads in §6 among them, that
function as general-purpose neural machinery, and finding these shared
primitives is itself a valuable interpretability goal. Reuse is also plausibly
one driver of generalization, since in-context learning likely succeeds
precisely because models apply the same retrieval and binding operations
across novel tasks without dedicated machinery for each. The chance-corrected results
strengthen this reading, since sharing survives at both granularities and is
not an artifact of coarse binning.

## 10. Conclusion

We set out to test whether circuits are consistent and specific, at the level
of description the field usually uses and at a finer one. Circuits over
attention heads and MLP layers are highly consistent and causally necessary,
but not specific. A case study of the two heads shared by every task shows
why, since what recurs is generic attention-sink machinery rather than
task-bound computation. Circuits over individual MLP neurons are far less
consistent, but far more specific, and not because they are less shared --
against chance they are more shared -- but because what they share tracks
task similarity instead of being uniform across every pair. Neither property,
consistency or specificity, is a fixed fact about a model. Both are properties
of the resolution at which a circuit is described, and claims about either
should say which resolution they were measured at.

That leaves the motivating question open in a specific way. Necessity and
sufficiency are satisfied at both levels we study, so they cannot tell us which
level to reason at. Consistency and specificity can, and they disagree, each
picking out a different level. Until some basis satisfies both at once, there is
no single level of abstraction that circuit analysis can be said to have earned,
and the choice remains one that has to be argued for rather than assumed.

## Appendix

### Specificity gap against threshold at \\(K\\)={K}%

"""]

    gap_p = (diag[diag.K == K].groupby(CONFIG + ["p"])
             .specificity_gap_pp.agg(["mean", "count"]))
    piv_m = gap_p["mean"].unstack("p").reindex(ORDER)
    piv_n = gap_p["count"].unstack("p").reindex(ORDER)
    out.append(table(
        [[label(k)] + [f"{fmt(piv_m.loc[k, c])} ({int(piv_n.loc[k, c])})" for c in piv_m.columns]
         for k in piv_m.index],
        ["Method and granularity"] + [f"P={c}%" for c in piv_m.columns],
        ["---"] + ["---:"] * len(piv_m.columns)))
    out.append("Mean specificity gap in percentage points, with the number of "
               "contributing model-task pairs in parentheses out of a possible 30. "
               "Support falls as P tightens because shared circuits go empty.\n")

    out.append(f"### Specificity gap against circuit size at \\\\(P\\\\)={p}%\n\n"
               + pivot_table(diag[diag.p == p], "specificity_gap_pp", CONFIG, "K", 1)
               + "\nMean specificity gap in percentage points, averaged over the "
                 "five models and six tasks.\n")

    out.append(f"### Donor-by-target matrices at \\\\(K\\\\)={K}%, \\\\(P\\\\)={p}%\n")
    for key in ORDER:
        stem = f"heatmaps_full_{key[0]}_{key[1]}_K{K}_p{p}"
        if (analysis / f"{stem}.png").exists():
            out.append(f"![{label(key)}]({figs}{stem}.png)\n\n*{label(key)}. Accuracy "
                       f"drop in percentage points, one panel per model. Rows are the "
                       f"donor task whose circuit is ablated, columns the target task "
                       f"evaluated. All four panels share one colour scale.*\n")

    out.append(f"""### Additional figures

{fig('circuit_size_distributions', f'Distribution of shared circuit size at K={K}%, P={p}% on a symmetric log scale. Each box covers the 30 model-task pairs of the cross-granularity comparison')}
{fig('circuit_composition_by_granularity', f'What the shared circuit is built from at K={K}%, P={p}%. Left: the split between MLP layers and attention heads at the head-and-MLP granularity, which the neuron granularity does not have by construction. Right: the share of the shared circuit at each relative depth, one line per attribution method and granularity, with the band the range over the six tasks and the dashed line a uniform spread over depth')}
{fig('pretraining/pretraining_combined', 'Reuse@P and necessity across OLMo-2-1B pretraining checkpoints at K=10% over heads and MLP layers, under EAP, sweeping the consensus threshold P. Checkpoints span the full stage-1 run plus two stage-2 anneal checkpoints', paper_figs)}
## Reproducing

```bash
python analysis/granularity_parity.py \\
  --results-root results/granularity_parity \\
  --output-dir results/granularity_parity_analysis \\
  --resamples 10000 --seed 42 --plots
python -m analysis.circuit_overlap_chance \\
  --results-root results/granularity_parity/granularity_parity_extraction \\
  --output-dir results/granularity_parity_analysis --K {K} --p {p} --plots
python -m analysis.circuit_structure_by_granularity \\
  --results-root results/granularity_parity/granularity_parity_extraction \\
  --analysis-dir results/granularity_parity_analysis --K {K} --p {p}
python -m analysis.selective_ablation_summary \\
  --results-root results/selective_ablation_k{K} \\
  --output-dir results/granularity_parity_analysis --K {K}
python analysis/granularity_parity_report.py \\
  --analysis-dir results/granularity_parity_analysis
```

Table 1 is also written as `table_granularity_parity.tex` for direct inclusion in
the paper. Numbers quoted from the earlier six-model EAP benchmark are collected
in `PRIOR` at the top of the report generator and sourced from
`paper/integrated_results.md`.
""")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--figure-prefix", default="", help="prefix for parity image paths")
    ap.add_argument("--paper-figure-prefix", default=PAPER_FIGS,
                    help="prefix for figures from the paper's figure directory")
    args = ap.parse_args()
    analysis = Path(args.analysis_dir)
    out = Path(args.output) if args.output else analysis / "REPORT.md"
    out.write_text(reflow(build(analysis, args.figure_prefix, args.paper_figure_prefix)))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
