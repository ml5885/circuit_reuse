#!/usr/bin/env python3
"""Compare RelP results under the default rule set (LN, Identity, AH, Half;
Arora et al. 2026) with those under another rule set, by default ``legacy``,
our pre-2026-09-21 set without the Identity-rule.

The default runs live in the standard directories
(granularity_parity_relp_<gran>, relp_<gran>_l40s); the other rule set's runs
carry a ``__lrp<tag>`` suffix (for the legacy results, rename the old
directories; see docs/relp_identity_rule_rerun.md). For every model, task and
granularity present in both we report, at K=10%:

  jaccard    overlap of the two consensus circuits at P=50
  reuse      within-task reuse@50 under each rule set
  necessity  necessity gap (pp) under each rule set
  gap        specificity gap (pp) under each rule set, from the cross-task matrices

and the per-model means. Writes CSVs and the LaTeX table
tables/lrp_variants_<variant>.tex.

    python analysis/compare_lrp_variants.py --variant legacy
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from analysis.granularity_parity import (GRAN_SHORT, MODEL_NAMES, TASK_NAMES, add_cross_metrics,
                                         read_cross_task, read_extraction)
from circuit_reuse.lrp_patch import PUBLISHED_LRP_RULES, lrp_rules_tag

RESULTS = REPO_ROOT / "results" / "granularity_parity"
PAPER_TABLES = next(REPO_ROOT.glob("paper2/_ICLR27*/tables"), None)
VARIANT_NAMES = {"legacy": "the same rules without the Identity-rule (our earlier configuration)",
                 "jafari": r"the RelP repository default \{LN, Identity, Half\}",
                 "arora": r"\citet{arora2026language}"}
KEYS = ["model", "task", "granularity"]


def consensus_sets(root: Path, variant: str | None, K: int, p: int) -> pd.DataFrame:
    rows = []
    for path in sorted(root.rglob("metrics.json")):
        is_variant = "__lrp" in str(path)
        if is_variant != (variant is not None) or (variant and f"__lrp{variant}" not in str(path)):
            continue
        d = json.loads(path.read_text())
        if d.get("method") != "relp":
            continue
        cell = d["by_k"].get(str(K), {}).get("thresholds", {}).get(str(p))
        if cell is None:
            continue
        rows.append({"model": d["model_name"], "task": d["task"], "granularity": d["granularity"],
                     "circuit": frozenset(cell["shared_components"])})
    return pd.DataFrame(rows)


def jaccard(a: frozenset, b: frozenset) -> float:
    return len(a & b) / len(a | b) if a | b else float("nan")


def paired(default: pd.DataFrame, variant: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    d = default[KEYS + cols].rename(columns={c: f"{c}_ours" for c in cols})
    v = variant[KEYS + cols].rename(columns={c: f"{c}_variant" for c in cols})
    return d.merge(v, on=KEYS, how="inner")


def build(results: Path, tag: str, K: int, p: int) -> pd.DataFrame:
    EXTRACTION = results / "granularity_parity_extraction"
    CROSS = results / "granularity_parity_cross_task"
    ext_ours = read_extraction(EXTRACTION)
    ext_var = read_extraction(EXTRACTION, variant=tag)
    sel = lambda df: df[(df.method == "relp") & (df.K == K) & (df.p == p)]
    table = paired(sel(ext_ours), sel(ext_var), ["reuse", "necessity_gap", "circuit_size"])

    sets = consensus_sets(EXTRACTION, None, K, p).merge(
        consensus_sets(EXTRACTION, tag, K, p), on=KEYS, suffixes=("_ours", "_variant"))
    sets["jaccard"] = [jaccard(a, b) for a, b in zip(sets.circuit_ours, sets.circuit_variant)]
    table = table.merge(sets[KEYS + ["jaccard"]], on=KEYS, how="left")

    cross_ours = add_cross_metrics(read_cross_task(CROSS))
    cross_var = add_cross_metrics(read_cross_task(CROSS, variant=tag))
    diag = lambda df: (df[(df.method == "relp") & (df.K == K) & (df.p == p) & (df.donor == df.target)]
                       .rename(columns={"target": "task"}))
    if not cross_ours.empty and not cross_var.empty:
        table = table.merge(paired(diag(cross_ours), diag(cross_var), ["specificity_gap_pp"]),
                            on=KEYS, how="left")
    return table.sort_values(["granularity", "model", "task"]).reset_index(drop=True)


def per_model(table: pd.DataFrame) -> pd.DataFrame:
    value_cols = [c for c in table.columns if c not in KEYS]
    return table.groupby(["granularity", "model"], as_index=False)[value_cols].mean()


def latex(summary: pd.DataFrame, variant: str, K: int, p: int) -> str:
    def pair(row, col, fmt="{:.1f}"):
        return f"{fmt.format(row[f'{col}_ours'])} / {fmt.format(row[f'{col}_variant'])}"
    has_gap = "specificity_gap_pp_ours" in summary.columns
    lines = [r"\begin{table}[h]", r"  \centering", r"  \small",
             r"  \begin{tabular}{ll" + "c" * (4 if has_gap else 3) + "}", r"    \toprule",
             r"    Granularity & Model & Jaccard & \reuseat{" + str(p) + r"} & Necessity gap"
             + (r" & Specificity gap" if has_gap else "") + r" \\", r"    \midrule"]
    for gran, g in summary.groupby("granularity", sort=False):
        for i, row in enumerate(g.itertuples(index=False)):
            r = row._asdict()
            cells = [GRAN_SHORT[gran].capitalize() if i == 0 else "", MODEL_NAMES.get(r["model"], r["model"]),
                     f"{r['jaccard']:.2f}", pair(r, "reuse"), pair(r, "necessity_gap")]
            if has_gap:
                cells.append(pair(r, "specificity_gap_pp"))
            lines.append("    " + " & ".join(cells) + r" \\")
        lines.append(r"    \midrule")
    lines[-1] = r"    \bottomrule"
    lines += [r"  \end{tabular}", r"  \vskip 1em",
              r"  \caption{\textbf{RelP under our rule set against " + VARIANT_NAMES[variant]
              + r".} At $K$=" + str(K) + r"\% and $P$=" + str(p) + r"\%, per model and averaged over tasks: "
              r"Jaccard overlap of the two consensus circuits, and within-task \reuseat{" + str(p) + r"}, "
              r"necessity gap and specificity gap (percentage points) under our rules / under the other rules.}",
              r"  \label{tab:lrp_variants_" + variant + "}", r"\end{table}", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="legacy", choices=list(PUBLISHED_LRP_RULES))
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--P", type=int, default=50)
    ap.add_argument("--results-root", type=Path, default=RESULTS)
    ap.add_argument("--out", default="results2/lrp_variants")
    ap.add_argument("--no-paper", action="store_true", help="do not write the table into the paper directory")
    args = ap.parse_args()

    tag = lrp_rules_tag(PUBLISHED_LRP_RULES[args.variant])
    table = build(args.results_root, tag, args.K, args.P)
    if table.empty:
        raise SystemExit(f"no variant runs found under {args.results_root} for tag {tag}")
    summary = per_model(table)
    out = Path(args.out) / args.variant
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "per_task.csv", index=False)
    summary.to_csv(out / "per_model.csv", index=False)
    tex = latex(summary, args.variant, args.K, args.P)
    (out / "lrp_variants.tex").write_text(tex)
    if PAPER_TABLES and not args.no_paper:
        (PAPER_TABLES / f"lrp_variants_{args.variant}.tex").write_text(tex)
    pd.set_option("display.width", 200)
    print(f"variant {args.variant} = {PUBLISHED_LRP_RULES[args.variant]} (tag {tag})")
    print(table.assign(task=table.task.map(TASK_NAMES)).to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print()
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
