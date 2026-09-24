"""Selective ablation for every ordered task pair and K of one model, in one process.

Same decomposition and output files as selective_ablation_experiment.py (one JSON per pair
and K): for tasks A and B, the shared core C_A & C_B, the A-only set C_A - C_B, the B-only set
C_B - C_A, and a random control matched to the A-only set, each ablated and evaluated on every
task. Evaluations are cached by the set of units removed, since the same set recurs across
pairs.

    python -m followups.selective_batch --model google/gemma-2-2b --results-dir <extraction root> --output-root <dir>
"""
import argparse
import json
import random
from pathlib import Path

from circuit_reuse.evaluate import evaluate_accuracy, evaluate_accuracy_with_ablation
from cross_task_experiment import find_metrics_file, load_shared_components
from followups.common import TASKS, eval_datasets, load_model
from main_experiment import _enumerate_all_components, _sample_control_components


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--method", default="eap_ig")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--K", default="1,5,10,20,30")
    parser.add_argument("--threshold", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    model = load_model(args.model)
    datasets = eval_datasets(args.model)
    units = _enumerate_all_components(model)
    baseline = {t: (lambda c: c[0] / c[1])(evaluate_accuracy(model, ds, task=t)) for t, ds in datasets.items()}
    cache = {}

    def accuracy(task, removed):
        key = (task, frozenset(removed))
        if key not in cache:
            c, n = evaluate_accuracy_with_ablation(model, datasets[task], task=task, removed=list(removed))
            cache[key] = c / n
        return cache[key]

    slug = args.model.replace("/", "_")
    for K in (int(k) for k in args.K.split(",")):
        circuits = {t: set(load_shared_components(
            find_metrics_file(args.results_dir, args.model, None, t, args.threshold, method=args.method,
                              granularity="head_mlp"), K, args.threshold)) for t in TASKS}
        out_dir = Path(args.output_root) / f"selective_ablation_k{K}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for a in TASKS:
            for b in TASKS:
                if a == b:
                    continue
                path = out_dir / f"selective_{slug}_K{K}_t{args.threshold}_{a}_vs_{b}.json"
                if path.exists():
                    continue
                c_a, c_b = circuits[a], circuits[b]
                conditions = {"shared_core": c_a & c_b, "residual_a": c_a - c_b, "residual_b": c_b - c_a}
                conditions["random_control"] = set(_sample_control_components(
                    sorted(conditions["residual_a"], key=lambda c: (c.layer, c.kind, c.index)), units,
                    random.Random(args.seed)))
                results = {}
                for name, removed in conditions.items():
                    results[name] = {"size": len(removed)}
                    for t in TASKS:
                        acc = accuracy(t, removed) if removed else baseline[t]
                        drop = baseline[t] - acc
                        results[name][t] = {"accuracy": acc, "drop_pp": 100 * drop,
                                            "relative_drop_pct": 100 * drop / baseline[t] if baseline[t] > 0 else 0.0}
                target = results["residual_a"][a]["drop_pp"]
                mean_other = sum(results["residual_a"][t]["drop_pp"] for t in TASKS if t != a) / (len(TASKS) - 1)
                path.write_text(json.dumps({
                    "model_name": args.model, "hf_revision": None, "method": args.method,
                    "task_a": a, "task_b": b, "K": K, "threshold": args.threshold,
                    "num_examples": 100, "seed": args.seed,
                    "circuit_sizes": {"c_a": len(c_a), "c_b": len(c_b), **{k: len(v) for k, v in conditions.items()}},
                    "baseline_accuracy": baseline, "conditions": results,
                    "selectivity": {"target_task": a, "target_drop_pp": target, "mean_nontarget_drop_pp": mean_other,
                                    "selectivity_ratio": target / mean_other if abs(mean_other) > 1e-6 else float("inf")},
                }, indent=2))
        print(f"K{K} done ({len(cache)} evaluations cached)", flush=True)


if __name__ == "__main__":
    main()
