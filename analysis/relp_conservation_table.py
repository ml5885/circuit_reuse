#!/usr/bin/env python3
"""LaTeX table of the relevance-conservation ratios written by
analysis/relp_conservation_check.py, one column per model.

    python analysis/relp_conservation_table.py --root results2/relp_conservation
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from analysis.granularity_parity import MODEL_NAMES

PAPER_TABLES = next(REPO_ROOT.glob("paper2/_ICLR27*/tables"), None)
RULE_SET_NAMES = {
    "ours": r"Ours \{LN, Identity, AH, Half\} (\citealp{arora2026language})",
    "legacy": r"Without the Identity-rule",
    "jafari": r"RelP repository default \{LN, Identity, Half\}",
}
CHECK_NAMES = {"norm": "RMSNorm", "attention": "Attention", "gate": "Gate product",
               "act_fn": "Activation function"}
MODEL_ORDER = ["google/gemma-2-2b", "google/gemma-2-2b-it", "meta-llama/Llama-3.2-3B",
               "meta-llama/Llama-3.2-3B-Instruct", "qwen3-4b"]


def cell(row: pd.Series) -> str:
    if all(abs(row[c] - 1) < 5e-4 for c in ("median", "q05", "q95")):
        return "1.000"
    return f"{row['median']:.2f} [{row.q05:.1f}, {row.q95:.1f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=REPO_ROOT / "results2" / "relp_conservation")
    ap.add_argument("--no-paper", action="store_true")
    args = ap.parse_args()

    runs = {}
    for cfg_path in sorted(args.root.glob("*/config.json")):
        cfg = json.loads(cfg_path.read_text())
        runs[cfg["model"]] = (cfg, pd.read_csv(cfg_path.with_name("summary.csv")))
    if not runs:
        raise SystemExit(f"no summaries under {args.root}")
    models = [m for m in MODEL_ORDER if m in runs] + sorted(m for m in runs if m not in MODEL_ORDER)
    n_ex = {cfg["num_examples"] for cfg, _ in runs.values()}
    dtypes = {cfg["dtype"] for cfg, _ in runs.values()}

    lines = [r"\begin{table}[h]", r"  \centering", r"  \small",
             r"  \begin{tabular}{ll" + "c" * len(models) + "}", r"    \toprule",
             r"    Rule set & Operation & " + " & ".join(MODEL_NAMES.get(m, m) for m in models) + r" \\",
             r"    \midrule"]
    for rules, rules_name in RULE_SET_NAMES.items():
        for i, (check, check_name) in enumerate(CHECK_NAMES.items()):
            cells = []
            for m in models:
                s = runs[m][1]
                r = s[(s.rules == rules) & (s.check == check)]
                cells.append(cell(r.iloc[0]) if len(r) else "--")
            lines.append("    " + " & ".join([rules_name if i == 0 else "", check_name] + cells) + r" \\")
        lines.append(r"    \midrule")
    lines[-1] = r"    \bottomrule"
    lines += [r"  \end{tabular}", r"  \vskip 1em",
              r"  \caption{\textbf{Numerical check of relevance conservation.} For each rule set and each "
              r"linearized operation, the ratio of the total gradient$\times$activation relevance entering the "
              r"operation to that leaving it, recorded in every layer during the RelP backward pass on "
              f"{max(n_ex)} IOI examples in {'/'.join(sorted(dtypes)).replace('float32', 'float32')} precision. "
              r"A conserved operation gives exactly $1$; otherwise we report the median and the 5--95\% range "
              r"of the per-layer ratios. The norm is conserved under the LN-rule, attention under the AH-rule, "
              r"the gate product under the half-rule (the ratio is exactly $2$ without it), and the activation "
              r"function only under the Identity-rule.}",
              r"  \label{tab:relp_conservation}", r"\end{table}", ""]
    tex = "\n".join(lines)
    out = args.root / "relp_conservation.tex"
    out.write_text(tex)
    if PAPER_TABLES and not args.no_paper:
        (PAPER_TABLES / "relp_conservation.tex").write_text(tex)
    print(tex)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
