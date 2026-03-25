"""Selective ablation experiment: shared core vs. task residuals.

For a pair of tasks (A, B), decomposes task A's circuit into:
  G_AB = C_A ∩ C_B   (shared core)
  R_A  = C_A \ C_B   (target residual)
  R_B  = C_B \ C_A   (cross residual)

Ablates each (plus a size-matched random control) and evaluates on all tasks.
"""

import argparse
import json
import random
from pathlib import Path

import torch

from models.olmo_adapter import load_model_any
from circuit_reuse.dataset import get_dataset
from circuit_reuse.evaluate import evaluate_accuracy_with_ablation, evaluate_accuracy
from circuit_reuse.circuit_extraction import Component
from cross_task_experiment import find_metrics_file, load_shared_components
from main_experiment import _sample_control_components, _enumerate_all_components


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Root dir with per-task metrics.json files.")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--hf_revision", type=str, default=None)
    parser.add_argument("--task-a", type=str, required=True, help="Source/target task A.")
    parser.add_argument("--task-b", type=str, required=True, help="Source task B for decomposition.")
    parser.add_argument("--eval-tasks", type=str, required=True,
                        help="Comma-separated tasks to evaluate on.")
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--threshold", type=int, default=100)
    parser.add_argument("--num-examples", type=int, default=100)
    parser.add_argument("--digits", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    eval_tasks = [t.strip() for t in args.eval_tasks.split(",") if t.strip()]
    results_dir = Path(args.results_dir)
    random.seed(args.seed)

    # Load circuits for tasks A and B
    metrics_a = find_metrics_file(results_dir, args.model_name, args.hf_revision, args.task_a)
    metrics_b = find_metrics_file(results_dir, args.model_name, args.hf_revision, args.task_b)
    c_a = set(load_shared_components(metrics_a, args.K, args.threshold))
    c_b = set(load_shared_components(metrics_b, args.K, args.threshold))

    # Decompose
    g_ab = c_a & c_b
    r_a = c_a - c_b
    r_b = c_b - c_a

    print(f"[DECOMPOSE] {args.task_a} vs {args.task_b} @ K={args.K}, threshold={args.threshold}")
    print(f"  |C_A|={len(c_a)}  |C_B|={len(c_b)}  |G|={len(g_ab)}  |R_A|={len(r_a)}  |R_B|={len(r_b)}")

    # Load model
    model = load_model_any(args.model_name, device=args.device, revision=args.hf_revision)
    model.eval()

    # Sample random control matched to |R_A|
    all_components = _enumerate_all_components(model)
    rng = random.Random(args.seed)
    random_control = _sample_control_components(list(r_a), all_components, rng)

    # Prepare datasets
    datasets = {}
    for task in eval_tasks:
        digits = args.digits if (args.digits and task == "addition") else 2
        datasets[task] = list(get_dataset(task, num_examples=args.num_examples, digits=digits))

    # Evaluate baselines
    baseline_acc = {}
    for task in eval_tasks:
        correct, total = evaluate_accuracy(model, datasets[task], task=task)
        baseline_acc[task] = correct / total if total > 0 else 0.0
        print(f"[BASELINE] {task}: {baseline_acc[task]*100:.1f}%")

    # Define ablation conditions
    conditions = {
        "shared_core": list(g_ab),
        "residual_a": list(r_a),
        "residual_b": list(r_b),
        "random_control": random_control,
    }

    # Run ablations
    results = {}
    for cond_name, components in conditions.items():
        results[cond_name] = {"size": len(components)}
        if not components:
            for task in eval_tasks:
                results[cond_name][task] = {
                    "accuracy": baseline_acc[task],
                    "drop_pp": 0.0,
                    "relative_drop_pct": 0.0,
                }
            print(f"[{cond_name}] empty set, skipping ablation")
            continue

        for task in eval_tasks:
            correct, total = evaluate_accuracy_with_ablation(
                model, datasets[task], task=task, removed=components
            )
            acc = correct / total if total > 0 else 0.0
            drop = baseline_acc[task] - acc
            rel_drop = (drop / baseline_acc[task] * 100.0) if baseline_acc[task] > 0 else 0.0
            results[cond_name][task] = {
                "accuracy": acc,
                "drop_pp": drop * 100.0,
                "relative_drop_pct": rel_drop,
            }
            print(f"  [{cond_name}] {task}: {acc*100:.1f}% (drop={drop*100:.1f}pp)")

    # Compute selectivity for R_A: target drop on A vs mean drop on others
    r_a_results = results["residual_a"]
    if args.task_a in eval_tasks:
        target_drop = r_a_results[args.task_a]["drop_pp"]
        other_drops = [r_a_results[t]["drop_pp"] for t in eval_tasks if t != args.task_a]
        mean_other = sum(other_drops) / len(other_drops) if other_drops else 0.0
        selectivity = target_drop / mean_other if abs(mean_other) > 1e-6 else float("inf")
    else:
        target_drop = mean_other = selectivity = float("nan")

    # Save
    output = {
        "model_name": args.model_name,
        "hf_revision": args.hf_revision,
        "task_a": args.task_a,
        "task_b": args.task_b,
        "K": args.K,
        "threshold": args.threshold,
        "num_examples": args.num_examples,
        "seed": args.seed,
        "circuit_sizes": {
            "c_a": len(c_a), "c_b": len(c_b),
            "shared_core": len(g_ab), "residual_a": len(r_a), "residual_b": len(r_b),
            "random_control": len(random_control),
        },
        "baseline_accuracy": baseline_acc,
        "conditions": results,
        "selectivity": {
            "target_task": args.task_a,
            "target_drop_pp": target_drop,
            "mean_nontarget_drop_pp": mean_other,
            "selectivity_ratio": selectivity,
        },
    }

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        model_slug = args.model_name.replace("/", "_")
        fname = f"selective_{model_slug}_K{args.K}_t{args.threshold}_{args.task_a}_vs_{args.task_b}.json"
        out_path = out_dir / fname
        with out_path.open("w") as f:
            json.dump(output, f, indent=2)
        print(f"\n[SAVED] {out_path}")
    else:
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
