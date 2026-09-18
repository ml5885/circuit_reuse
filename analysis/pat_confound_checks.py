"""Checks on existing attribution data for two PAT review claims.

Claim A: component-level EAP-IG scores a node as the sum of |edge| scores
(no cancellation, grows with fan-out), while RelP scores nodes with a signed
sum. The review says this inflates early-layer / MLP components and makes
EAP-IG component circuits denser and less specific.

Claim B: top-K% over a ~300x larger neuron basis buries a sparse circuit in
attribution noise, forcing Reuse@P toward 0.

Inputs: cache/*.jsonl (per-example head_mlp scores for eap_ig / relp) and
results/granularity_parity_analysis/extraction_tidy.csv.
"""

import argparse
import itertools
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.granularity_parity import CONFIG_COLORS, CONFIG_ORDER
from circuit_reuse.dataset import get_model_display_name

matplotlib.use("Agg")
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                     "axes.titlesize": 11, "axes.labelsize": 10,
                     "xtick.labelsize": 9, "ytick.labelsize": 9,
                     "legend.fontsize": 8.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": .25, "grid.linewidth": .6})

MODELS = ["google/gemma-2-2b", "google/gemma-2-2b-it", "meta-llama/Llama-3.2-3B",
          "meta-llama/Llama-3.2-3B-Instruct", "qwen3-4b"]
TASKS = ["addition", "arc_challenge", "arc_easy", "boolean", "ioi", "mcqa"]
KS = [1, 5, 10, 20]
P = 50

KV_HEADS = {"gemma": 4, "llama": 8, "qwen3": 8}

# (method, variant label, transform of the per-example score matrix)
VARIANTS = [
    ("eap_ig", "EAP-IG, sum of |edge| (paper)"),
    ("eap_ig", r"EAP-IG, sum of |edge| / $\sqrt{d}$"),
    ("eap_ig", "EAP-IG, sum of |edge| / d"),
    ("relp", "RelP, signed score (paper)"),
    ("relp", "RelP, |score|"),
]
VARIANT_COLORS = ["#2E6E96", "#5FA8D0", "#AFD4E9", "#C98A1B", "#E8B33C"]


def cache_path(model, task, method):
    stem = model.replace("/", "_")
    d = "d3" if task == "addition" else "dna"
    m = "eap_ig__ig5" if method == "eap_ig" else method
    return Path("cache") / f"{stem}__none__{task}__{m}__n1000__{d}__s42.jsonl"


def load_scores(path):
    """Return (components, [n_ex, n_comp] score matrix); components are (layer, kind, index)."""
    rows = [json.loads(line)["components"] for line in path.open()]
    comps = sorted({(c["layer"], c["kind"], c["index"]) for c in rows[0]})
    idx = {c: i for i, c in enumerate(comps)}
    S = np.zeros((len(rows), len(comps)), dtype=np.float32)
    for i, row in enumerate(rows):
        for c in row:
            S[i, idx[(c["layer"], c["kind"], c["index"])]] = c["score"]
    return comps, S


def out_degree(comps, model):
    """Number of outgoing edges of each node in the EAP graph (what sum|e| sums over)."""
    n_layers = max(c[0] for c in comps) + 1
    n_heads = sum(1 for c in comps if c[0] == 0 and c[1] == "head")
    n_kv = next(v for k, v in KV_HEADS.items() if k in model.lower())
    slots = n_heads + 2 * n_kv + 1
    d = np.array([(n_layers - 1 - l) * slots + 1 + (kind == "head") for l, kind, _ in comps],
                 dtype=np.float32)
    return d


def topk_sets(S, k):
    take = max(1, int(S.shape[1] * k / 100))
    order = np.argsort(-S, axis=1, kind="stable")[:, :take]
    M = np.zeros(S.shape, dtype=bool)
    np.put_along_axis(M, order, True, axis=1)
    return M


def shared(M, p=P):
    need = int(np.ceil(p / 100 * M.shape[0]))
    return M.sum(0) >= need


def reuse(M, p=P):
    take = M[0].sum()
    return min(shared(M, p).sum(), take) / take * 100


def jaccard(a, b):
    return (a & b).sum() / max(1, (a | b).sum())


def variant_scores(method, S, d):
    yield S
    if method == "eap_ig":
        yield S / np.sqrt(d)
        yield S / d
    else:
        yield np.abs(S)


def run_claim_a(out):
    rows, memberships = [], {}
    for model in MODELS:
        for task in TASKS:
            for method in ("eap_ig", "relp"):
                comps, S = load_scores(cache_path(model, task, method))
                d = out_degree(comps, model)
                depth = np.array([c[0] for c in comps]) / (max(c[0] for c in comps))
                is_mlp = np.array([c[1] == "mlp" for c in comps])
                labels = [v for m, v in VARIANTS if m == method]
                base = None
                for label, Sv in zip(labels, variant_scores(method, S, d)):
                    for k in KS:
                        M = topk_sets(Sv, k)
                        if k == 10:
                            memberships[(model, task, label)] = shared(M)
                        if label.endswith("(paper)"):
                            base = base or {}
                            base[k] = M
                        r = dict(model=model, task=task, method=method, variant=label, K=k,
                                 reuse=reuse(M),
                                 shared_size=int(shared(M).sum()),
                                 mean_depth=float(np.broadcast_to(depth, M.shape)[M].mean()),
                                 early_frac=float((np.broadcast_to(depth, M.shape)[M] < 1 / 3).mean()),
                                 mlp_frac=float(np.broadcast_to(is_mlp, M.shape)[M].mean()),
                                 neg_frac=float((S[M] < 0).mean()),
                                 jaccard_vs_paper=float(np.mean([jaccard(a, b) for a, b in zip(M, base[k])])))
                        rows.append(r)
                print(f"[A] {model} {task} {method}")
    df = pd.DataFrame(rows)
    df.to_csv(out / "claim_a_variants.csv", index=False)

    # cross-task structural overlap of S_50 at K=10 within each model
    ov = []
    for model in MODELS:
        for _, label in VARIANTS:
            js = [jaccard(memberships[(model, a, label)], memberships[(model, b, label)])
                  for a, b in itertools.combinations(TASKS, 2)]
            allshared = np.logical_and.reduce([memberships[(model, t, label)] for t in TASKS]).sum()
            ov.append(dict(model=model, variant=label, mean_pair_jaccard=float(np.mean(js)),
                           shared_all_tasks=int(allshared)))
    ov = pd.DataFrame(ov)
    ov.to_csv(out / "claim_a_cross_task_overlap.csv", index=False)
    return df, ov


def plot_claim_a(df, ov, out):
    d10 = df[df.K == 10]
    labels = [v for _, v in VARIANTS]
    x = np.arange(len(MODELS))
    w = 0.8 / len(labels)
    metrics = [("reuse", "Reuse@50 at K=10% (%)"),
               ("mean_depth", "Mean depth of selected nodes (0 = first layer, 1 = last)"),
               ("early_frac", "Share of circuit in the first third of layers"),
               ("mlp_frac", "Share of circuit that is MLP blocks")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.6))
    for ax, (col, title) in zip(axes.ravel(), metrics):
        for j, (label, color) in enumerate(zip(labels, VARIANT_COLORS)):
            sub = d10[d10.variant == label].groupby("model")[col].agg(["mean", "std"]).reindex(MODELS)
            ax.bar(x + (j - 2) * w, sub["mean"], w, yerr=sub["std"], color=color,
                   edgecolor="white", linewidth=.5, error_kw=dict(lw=.7, capsize=2), label=label)
        ax.set_xticks(x)
        ax.set_xticklabels([get_model_display_name(m) for m in MODELS], rotation=15, ha="right")
        ax.set_title(title)
    fig.suptitle("Component-level circuits under different score rules (error bars: s.d. across 6 tasks)")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), ncol=5, frameon=False, loc="lower center")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out / "claim_a_variants.png", dpi=200)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, (col, title) in zip(axes, [("mean_pair_jaccard", "Overlap of shared circuits between task pairs (Jaccard, K=10%)"),
                                       ("shared_all_tasks", "Components in the shared circuit of all six tasks (K=10%)")]):
        for j, (label, color) in enumerate(zip(labels, VARIANT_COLORS)):
            sub = ov[ov.variant == label].set_index("model").reindex(MODELS)
            ax.bar(x + (j - 2) * w, sub[col], w, color=color, edgecolor="white", linewidth=.5, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels([get_model_display_name(m) for m in MODELS], rotation=15, ha="right")
        ax.set_title(title)
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=5, frameon=False, loc="lower center")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / "claim_a_cross_task_overlap.png", dpi=200)


def plot_depth_profiles(out):
    labels = [v for _, v in VARIANTS]
    bins = np.linspace(0, 1, 11)
    fig, axes = plt.subplots(1, len(MODELS), figsize=(14, 3.4), sharey=True)
    for ax, model in zip(axes, MODELS):
        hist = {label: np.zeros(10) for label in labels}
        for task in TASKS:
            for method in ("eap_ig", "relp"):
                comps, S = load_scores(cache_path(model, task, method))
                d = out_degree(comps, model)
                depth = np.array([c[0] for c in comps]) / max(c[0] for c in comps)
                for label, Sv in zip([v for m, v in VARIANTS if m == method], variant_scores(method, S, d)):
                    M = topk_sets(Sv, 10)
                    h, _ = np.histogram(np.repeat(depth, M.sum(0)), bins=bins)
                    hist[label] += h / M.sum()
        for label, color in zip(labels, VARIANT_COLORS):
            ax.plot((bins[:-1] + bins[1:]) / 2, hist[label] / len(TASKS), "-o", ms=3, color=color, label=label)
        ax.set_title(get_model_display_name(model))
        ax.set_xlabel("depth (0 = first layer, 1 = last)")
    axes[0].set_ylabel("share of circuit")
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=5, frameon=False, loc="lower center")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out / "claim_a_depth_profiles.png", dpi=200)


def run_claim_b(out):
    df = pd.read_csv("results/granularity_parity_analysis/extraction_tidy.csv")
    pools = json.load(open("results/granularity_parity_analysis/component_pools.json"))
    df = df[(df.p == P) & df.model.isin(MODELS) & df.task.isin(TASKS)].copy()
    df["basis"] = [pools[m][g] for m, g in zip(df.model, df.granularity)]
    df["per_example_size"] = (df.basis * df.K / 100).astype(int).clip(lower=1)
    df["config"] = list(zip(df.method, df.granularity))
    summ = df.groupby(["method", "granularity", "K"]).agg(
        reuse_mean=("reuse", "mean"), reuse_sd=("reuse", "std"),
        S50_mean=("circuit_size", "mean"), size_mean=("per_example_size", "mean")).reset_index()
    summ.to_csv(out / "claim_b_reuse_vs_k.csv", index=False)

    names = {("eap_ig", "head_mlp"): "EAP-IG component", ("relp", "head_mlp"): "RelP component",
             ("eap_ig", "neuron"): "EAP-IG neuron", ("relp", "neuron"): "RelP neuron"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    ax = axes[0]
    for cfg in CONFIG_ORDER:
        s = summ[(summ.method == cfg[0]) & (summ.granularity == cfg[1])]
        ax.errorbar(s.K, s.reuse_mean, yerr=s.reuse_sd, marker="o", ms=4, capsize=2, lw=1.2,
                    color=CONFIG_COLORS[cfg], label=names[cfg])
    for cfg in CONFIG_ORDER[2:]:
        s = summ[(summ.method == cfg[0]) & (summ.granularity == cfg[1])]
        r30 = s[s.K == 30].reuse_mean.item()
        ks = np.array([1, 5, 10, 20, 30])
        ax.plot(ks, np.minimum(100, r30 * 30 / ks), ":", color=CONFIG_COLORS[cfg], lw=1,
                label="review's picture (small stable set + noise): 1/K" if cfg == CONFIG_ORDER[2] else None)
    ax.set_xlabel("K (%)")
    ax.set_ylabel("Reuse@50 (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Reuse vs K (dotted: what the review's picture predicts)")

    ax = axes[1]
    for cfg in CONFIG_ORDER:
        s = df[df.config == cfg]
        ax.scatter(s.per_example_size, s.reuse.clip(lower=0.05), s=10, alpha=.6, color=CONFIG_COLORS[cfg], label=names[cfg])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"per-example circuit size $|C_i|$")
    ax.set_ylabel("Reuse@50 (%)")
    ax.set_title("Reuse against circuit size in units\n(one point per model, task, K)")

    ax = axes[2]
    for cfg in CONFIG_ORDER:
        s = df[df.config == cfg]
        ax.scatter(s.per_example_size, s.circuit_size.clip(lower=0.5), s=10, alpha=.6, color=CONFIG_COLORS[cfg], label=names[cfg])
    lim = np.array([1, 1e5])
    ax.plot(lim, lim, "-", color="0.5", lw=.8, label="$|S_{50}| = |C_i|$")
    ax.plot(lim, 0.1 * lim, "--", color="0.5", lw=.8, label=r"$|S_{50}| = 0.1\,|C_i|$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"per-example circuit size $|C_i|$")
    ax.set_ylabel(r"shared circuit size $|S_{50}|$")
    ax.set_title("Shared circuit grows with K\n(a small stable set would give a flat line)")
    h0, l0 = axes[0].get_legend_handles_labels()
    h2, l2 = axes[2].get_legend_handles_labels()
    order = [i for i, l in enumerate(l0) if l in names.values()] + [i for i, l in enumerate(l0) if l not in names.values()]
    fig.legend([h0[i] for i in order] + h2[-2:], [l0[i] for i in order] + l2[-2:], ncol=7, frameon=False, loc="lower center")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / "claim_b_reuse_vs_k.png", dpi=200)
    plot_topn_bound(out)
    return summ


def plot_topn_bound(out):
    """Upper bound on neuron Reuse@P under a top-N rule with N matched to the component
    circuit size at K=10%. A unit in the top-N of >= P% of examples is also in the top-1%
    of those examples, so |S_P(top-N)| <= |S_P(K=1%)| and Reuse@P <= |S_P(K=1%)| / N."""
    df = pd.read_csv("results/granularity_parity_analysis/extraction_tidy.csv")
    pools = json.load(open("results/granularity_parity_analysis/component_pools.json"))
    df = df[df.model.isin(MODELS) & df.task.isin(TASKS)]
    rows = []
    for (model, task, method), g in df.groupby(["model", "task", "method"]):
        n = max(1, int(pools[model]["head_mlp"] * 10 / 100))
        comp = g[(g.granularity == "head_mlp") & (g.K == 10)].set_index("p").circuit_size
        neur = g[(g.granularity == "neuron") & (g.K == 1)].set_index("p").circuit_size
        for p in comp.index:
            rows.append(dict(model=model, task=task, method=method, p=p, N=n,
                             component_reuse=min(comp[p], n) / n * 100,
                             neuron_bound=min(neur[p], n) / n * 100))
    b = pd.DataFrame(rows)
    b.to_csv(out / "claim_b_topn_bound.csv", index=False)
    summ = b.groupby(["method", "p"])[["component_reuse", "neuron_bound"]].mean().unstack("method")
    print(summ.round(1).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, method in zip(axes, ["eap_ig", "relp"]):
        s = b[b.method == method].groupby("p")[["component_reuse", "neuron_bound"]].agg(["mean", "std"])
        ax.errorbar(s.index, s[("component_reuse", "mean")], yerr=s[("component_reuse", "std")], marker="o",
                    ms=4, capsize=2, color=CONFIG_COLORS[(method, "head_mlp")], label="component, actual (K=10%)")
        ax.errorbar(s.index, s[("neuron_bound", "mean")], yerr=s[("neuron_bound", "std")], marker="s",
                    ms=4, capsize=2, color=CONFIG_COLORS[(method, "neuron")], label="neuron, largest possible top-N value")
        ax.set_title({"eap_ig": "EAP-IG", "relp": "RelP"}[method] + ": neuron circuit cut to the component circuit size")
        ax.set_xlabel("P: required % of examples that select the unit")
    axes[0].set_ylabel("Reuse@P (%)")
    axes[0].set_ylim(0, 105)
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=2, frameon=False, loc="lower center")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out / "claim_b_topn_bound.png", dpi=200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/pat_confound_checks")
    ap.add_argument("--skip-a", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)

    summ = run_claim_b(out)
    print(summ.round(1).to_string())
    if args.skip_a:
        return
    df, ov = run_claim_a(out)
    plot_claim_a(df, ov, out)
    plot_depth_profiles(out)
    d10 = df[df.K == 10]
    print(d10.groupby("variant")[["reuse", "shared_size", "mean_depth", "early_frac", "mlp_frac", "neg_frac", "jaccard_vs_paper"]].mean().round(3).to_string())
    print(ov.groupby("variant")[["mean_pair_jaccard", "shared_all_tasks"]].mean().round(3).to_string())
    print(df.groupby(["variant", "K"])["reuse"].mean().unstack("K").round(1).to_string())


if __name__ == "__main__":
    main()
