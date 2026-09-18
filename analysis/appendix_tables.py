"""Appendix tables for one attribution method and granularity.

The paper's appendix tables were originally produced for EAP alone. This
regenerates all of them from the granularity-parity runs, which cover both
attribution methods at both granularities, so every table can be reported for
the configuration it belongs to.

Run: python -m analysis.appendix_tables --method eap_ig --granularity head_mlp
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from circuit_reuse.dataset import get_model_display_name, get_task_display_name

REPO = Path(__file__).resolve().parent.parent
EXTRACTION = REPO / "results" / "granularity_parity" / "granularity_parity_extraction"
CROSS_TASK = REPO / "results" / "granularity_parity" / "granularity_parity_cross_task"

TASKS = ["addition", "arc_challenge", "arc_easy", "boolean", "ioi", "mcqa"]
MODELS = ["google/gemma-2-2b", "google/gemma-2-2b-it", "meta-llama/Llama-3.2-3B",
          "meta-llama/Llama-3.2-3B-Instruct", "qwen3-4b", "qwen3-8b"]
KS = [1, 5, 10, 20, 30]

METHOD_LABEL = {"eap_ig": "EAP-IG", "relp": "RelP"}
GRAN_LABEL = {"head_mlp": "attention heads and MLP blocks", "neuron": "MLP neurons"}


def load_runs(method: str, granularity: str) -> dict[tuple[str, str], dict]:
    """Map (model, task) to its metrics.json for this method and granularity."""
    runs = {}
    for path in sorted((EXTRACTION / f"granularity_parity_{method}_{granularity}").glob("*/metrics.json")):
        metrics = json.loads(path.read_text())
        runs[(metrics["model_name"], metrics["task"])] = metrics
    return runs


def load_cross_task(method: str, granularity: str, k: int, p: int) -> dict[str, dict]:
    """Map model to its donor-by-target accuracy drops at this K and P."""
    matrices = {}
    for path in (CROSS_TASK / f"{method}_{granularity}_l40s").glob(f"*_K{k}_p{p}.json"):
        data = json.loads(path.read_text())
        matrices[data["model_name"]] = data["cells"]
    return matrices


def at(metrics: dict, k: int, p: int) -> dict | None:
    return metrics["by_k"].get(str(k), {}).get("thresholds", {}).get(str(p))


def table(rows: list[str], caption: str, label: str, col_spec: str) -> str:
    header = " & ".join(["$K$", "Model"] + [get_task_display_name(t) for t in TASKS])
    return "\n".join([
        r"\begin{table*}[htbp]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        rf"\begin{{tabular}}{{cl{col_spec}}}", r"\toprule",
        header + r" \\", r"\midrule", *rows, r"\bottomrule",
        r"\end{tabular}", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        r"\end{table*}", "",
    ])


def k_blocks(cell: callable) -> list[str]:
    """One block of model rows per K, with the K label on the first row.

    Models with no data in any cell are dropped rather than printed as a row of
    dashes; the cross-task runs cover five of the six models.
    """
    models = [m for m in MODELS
              if any(cell(m, task, k) != "--" for task in TASKS for k in KS)]
    rows = []
    for i, k in enumerate(KS):
        for j, model in enumerate(models):
            k_label = rf"{k}\%" if j == 0 else ""
            values = [cell(model, task, k) for task in TASKS]
            rows.append(f"{k_label} & \\texttt{{{get_model_display_name(model)}}} & "
                        + " & ".join(values) + r" \\")
        if i < len(KS) - 1:
            rows.append(r"\midrule")
    return rows


def reuse_table(runs, p, method, granularity):
    def cell(model, task, k):
        entry = at(runs[(model, task)], k, p) if (model, task) in runs else None
        return f"{entry['reuse_percent']:.0f}" if entry else "--"

    return table(k_blocks(cell),
                 rf"\reuseat{{{p}}} (\%) over {GRAN_LABEL[granularity]} under {METHOD_LABEL[method]}, "
                 r"for every model, task, and circuit size $K$.",
                 f"tab:reuse_all_{method}_{granularity}", "r" * len(TASKS))


def necessity_table(runs, p, method, granularity):
    def cell(model, task, k):
        metrics = runs.get((model, task))
        entry = at(metrics, k, p) if metrics else None
        if not entry or not metrics["baseline_val_accuracy"]:
            return "--"
        val = entry["val"]
        return f"{(val['control_accuracy'] - val['ablation_accuracy']) / metrics['baseline_val_accuracy']:.2f}"

    return table(k_blocks(cell),
                 rf"\Causalmetric~of the shared set over {GRAN_LABEL[granularity]} under "
                 rf"{METHOD_LABEL[method]}, at $P$={p}\%. Positive values mean ablating the shared "
                 r"set costs more accuracy than a capacity-matched random ablation.",
                 f"tab:lift_all_{method}_{granularity}", "r" * len(TASKS))


def composition_table(runs, p, method, granularity):
    def cell(model, task, k):
        entry = at(runs[(model, task)], k, p) if (model, task) in runs else None
        if not entry or not entry["shared_components"]:
            return "--"
        heads = sum(c.startswith("head") for c in entry["shared_components"])
        total = len(entry["shared_components"])
        return f"{100 * heads / total:.0f}/{100 * (total - heads) / total:.0f}"

    return table(k_blocks(cell),
                 rf"Attention head / MLP block percentage of the shared set under "
                 rf"{METHOD_LABEL[method]}, at $P$={p}\%. A dash marks an empty shared set.",
                 f"tab:composition_all_{method}_{granularity}", "c" * len(TASKS))


def decomposition_table(runs, p, method, granularity):
    def cell(model, task, k):
        entry = at(runs[(model, task)], k, p) if (model, task) in runs else None
        if not entry:
            return "--"
        own = set(entry["shared_components"])
        partners = [at(runs[(model, other)], k, p) for other in TASKS
                    if other != task and (model, other) in runs]
        partners = [set(x["shared_components"]) for x in partners if x]
        if not partners:
            return "--"
        core = sum(len(own & other) for other in partners) / len(partners)
        only = sum(len(own - other) for other in partners) / len(partners)
        complement = sum(len(other - own) for other in partners) / len(partners)
        return rf"{core:.0f}\,/\,{only:.0f}\,/\,{complement:.0f}"

    return table(k_blocks(cell),
                 rf"Mean shared core / task-only / partner-only sizes over {GRAN_LABEL[granularity]} "
                 rf"under {METHOD_LABEL[method]}, at $P$={p}\%, averaged over the five partner tasks.",
                 f"tab:decomposition_all_{method}_{granularity}", "c" * len(TASKS))


def diag_offdiag_table(method, granularity, p):
    matrices = {k: load_cross_task(method, granularity, k, p) for k in KS}

    def cell(model, task, k):
        cells = matrices[k].get(model)
        if not cells or task not in cells:
            return "--"
        own = cells[task][task]["accuracy_drop_pp"]
        other = [cells[donor][task]["accuracy_drop_pp"] for donor in cells if donor != task]
        return rf"{own:.0f}\,/\,{sum(other) / len(other):.0f}" if other else "--"

    return table(k_blocks(cell),
                 rf"Own-circuit / mean other-circuit accuracy drop in percentage points over "
                 rf"{GRAN_LABEL[granularity]} under {METHOD_LABEL[method]}, at $P$={p}\%.",
                 f"tab:diag_offdiag_all_{method}_{granularity}", "c" * len(TASKS))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", default="eap_ig", choices=["eap_ig", "relp"])
    p.add_argument("--granularity", default="head_mlp", choices=["head_mlp", "neuron"])
    p.add_argument("--threshold", type=int, default=50)
    p.add_argument("--output-dir", type=str, default="results2/tables")
    args = p.parse_args()

    runs = load_runs(args.method, args.granularity)
    print(f"[LOAD] {len(runs)} runs for {args.method}/{args.granularity}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "reuse_all": reuse_table(runs, args.threshold, args.method, args.granularity),
        "lift_all": necessity_table(runs, args.threshold, args.method, args.granularity),
        "decomposition_all": decomposition_table(runs, args.threshold, args.method, args.granularity),
        "diag_offdiag_all": diag_offdiag_table(args.method, args.granularity, args.threshold),
    }
    if args.granularity == "head_mlp":
        tables["composition_all"] = composition_table(runs, args.threshold, args.method, args.granularity)

    for name, body in tables.items():
        path = out_dir / f"{name}_{args.method}_{args.granularity}.tex"
        path.write_text(body)
        print(f"[SAVED] {path}")


if __name__ == "__main__":
    main()
