"""Check the paper's claim that P=50% is the only consensus threshold at which all four
method x granularity configurations have a non-empty shared circuit in most model-task pairs.

Counts, for every configuration, K and P, the share of the 30 model-task pairs (5 main
models x 6 tasks) whose shared circuit S_P is non-empty.
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.granularity_parity import CONFIG_ORDER

matplotlib.use("Agg")
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})

NAMES = {("eap_ig", "head_mlp"): "EAP-IG component", ("relp", "head_mlp"): "RelP component",
         ("eap_ig", "neuron"): "EAP-IG neuron", ("relp", "neuron"): "RelP neuron"}
OUT = Path("results/pat_confound_checks")


def md_table(df):
    df = df.reset_index()
    cols = [str(c) for c in df.columns]
    rows = [" | ".join(cols), " | ".join("---" for _ in cols)]
    rows += [" | ".join(str(v) for v in r) for r in df.itertuples(index=False)]
    return "\n".join(f"| {r} |" for r in rows)


def main():
    df = pd.read_csv("results/granularity_parity_analysis/extraction_tidy.csv")
    pools = json.load(open("results/granularity_parity_analysis/component_pools.json"))
    df = df[df.model.isin(pools) & (df.task != "mmlu")]
    share = (df.assign(nonempty=df.circuit_size > 0)
               .groupby(["method", "granularity", "K", "p"]).nonempty.mean().mul(100))
    table = share.unstack("p").round(0).astype(int)
    table.to_csv(OUT / "nonempty_shared_circuits.csv")

    wide = share.unstack(["method", "granularity"])
    most = (wide >= 50).all(axis=1).unstack("p")
    every = (wide == 100).all(axis=1).unstack("p")

    lines = ["# Share of model-task pairs (out of 30) with a non-empty shared circuit S_P", "",
             "Rows: configuration and K (%). Columns: P (%).", "",
             md_table(table), "",
             "## All four configurations non-empty in at least half of the pairs", "",
             md_table(most.map(lambda b: "yes" if b else "no")), "",
             "## All four configurations non-empty in every pair", "",
             md_table(every.map(lambda b: "yes" if b else "no")), ""]
    (OUT / "nonempty_shared_circuits.md").write_text("\n".join(lines))
    print("\n".join(lines))

    ps = sorted(df.p.unique())
    ks = sorted(df.K.unique())
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2), sharey=True)
    for ax, cfg in zip(axes, CONFIG_ORDER):
        m = table.loc[cfg].reindex(index=ks, columns=ps).values
        im = ax.imshow(m, vmin=0, vmax=100, cmap="Blues", aspect="auto")
        for i in range(len(ks)):
            for j in range(len(ps)):
                ax.text(j, i, f"{m[i, j]:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if m[i, j] > 60 else "black")
        ax.set_xticks(range(len(ps)))
        ax.set_xticklabels(ps, fontsize=8)
        ax.set_yticks(range(len(ks)))
        ax.set_yticklabels(ks)
        ax.set_xlabel("P (%)")
        ax.set_title(NAMES[cfg], fontsize=10)
    axes[0].set_ylabel("K (%)")
    fig.colorbar(im, ax=axes, fraction=.02, pad=.01, label="% of pairs non-empty")
    fig.suptitle("Share of the 30 model-task pairs whose shared circuit is non-empty", fontsize=11)
    fig.savefig(OUT / "nonempty_shared_circuits.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
