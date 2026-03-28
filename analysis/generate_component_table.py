"""Generate LaTeX tables of attention head vs MLP breakdown across top-K thresholds.

For each (model, task, K%), computes the mean number of attention heads and MLPs
in the top-K% circuit across all examples. Outputs one table per task.

Requires: booktabs
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from circuit_reuse.dataset import get_model_display_name, get_task_display_name


def parse_attrib_filename(path: Path) -> dict:
    stem = path.stem
    parts = stem.split("__")
    return {
        "model_name": parts[0].replace("_", "/", 1) if "_" in parts[0] else parts[0],
        "task": parts[2],
    }


def compute_attn_fractions(
    examples: list[list[dict]], k_percents: list[float]
) -> dict[float, dict[str, float]]:
    n_total = len(examples[0])
    n_examples = len(examples)
    results = {}
    for k_pct in k_percents:
        top_n = max(1, math.ceil(n_total * k_pct / 100))
        head_sum = 0
        mlp_sum = 0
        for ex in examples:
            for c in ex[:top_n]:
                if c["kind"] == "head":
                    head_sum += 1
                else:
                    mlp_sum += 1
        heads = head_sum / n_examples
        mlps = mlp_sum / n_examples
        results[k_pct] = {"heads": heads, "mlps": mlps}
    return results


def build_table(
    tasks: list[str],
    k_percents: list[float],
    model: str,
    data: dict[str, dict[str, dict]],
) -> str:
    model_disp = get_model_display_name(model)
    safe_model = model.replace("/", "-").replace(" ", "-").lower()
    n_k = len(k_percents)
    col_spec = "l" + "r" * n_k

    k_headers = " & ".join(f"${int(k)}\\%$" for k in k_percents)
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        rf"\caption{{{model_disp}: mean attention heads\,/\,MLPs in the top-$K\%$ circuit.}}",
        rf"\label{{tab:comp-{safe_model}}}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        rf"\textbf{{Task}} & {k_headers} \\",
        r"\midrule",
    ]

    for task in tasks:
        task_disp = get_task_display_name(task)
        cells = [task_disp]
        counts = data[model].get(task)
        for k in k_percents:
            if counts is None:
                cells.append("--")
            else:
                c = counts[k]
                cells.append(f"{c['heads']:.1f}\\,/\\,{c['mlps']:.1f}")
        lines.append(" & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=str, default="cache")
    p.add_argument("--output", type=str, default="results2/tables/component_breakdown.tex")
    p.add_argument("--k-percents", type=str, default="1,10,20,30")
    p.add_argument("--models", type=str, default=None)
    p.add_argument("--tasks", type=str, default=None)
    p.add_argument("--exclude-tasks", type=str, default="")
    args = p.parse_args()

    cache_dir = Path(args.cache_dir)
    k_percents = [float(x) for x in args.k_percents.split(",")]
    model_filter = set(args.models.split(",")) if args.models else None
    task_filter = set(args.tasks.split(",")) if args.tasks else None
    exclude_tasks = set(args.exclude_tasks.split(",")) if args.exclude_tasks else set()

    data: dict[str, dict[str, dict]] = defaultdict(dict)
    all_tasks: set[str] = set()

    for path in sorted(cache_dir.glob("*.jsonl")):
        info = parse_attrib_filename(path)
        model_name, task = info["model_name"], info["task"]
        if model_filter and model_name not in model_filter:
            continue
        if task_filter and task not in task_filter:
            continue
        if task in exclude_tasks:
            continue

        print(f"[LOAD] {model_name} / {task}")
        examples = []
        with path.open() as f:
            for line in f:
                examples.append(json.loads(line)["components"])
        if not examples:
            continue
        data[model_name][task] = compute_attn_fractions(examples, k_percents)
        all_tasks.add(task)

    tasks = sorted(all_tasks)
    models = sorted(data.keys())

    tables = []
    for model in models:
        tables.append(build_table(tasks, k_percents, model, data))

    tex = "\n".join(tables)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(tex)
    print(f"\n[SAVED] {out_path}")
    print("\n" + tex)


if __name__ == "__main__":
    main()
