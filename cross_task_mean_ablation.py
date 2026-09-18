"""Cross-task ablation matrix using **mean ablation** instead of zero ablation.

Mirrors cross_task_experiment.py end-to-end. For each (model, target task A,
source task B), we ablate task B's shared circuit while evaluating on task A,
replacing each ablated component with its mean activation computed over the
corrupted version of task A's dataset (following Wang et al. 2022, Miller et
al. 2024).

We do NOT re-run attribution. Shared components are loaded from the same
metrics.json files used by the zero-ablation cross-task experiment.
"""

import argparse
import json
import os
import random
from pathlib import Path

import torch

from models.olmo_adapter import load_model_any
from circuit_reuse.dataset import get_dataset, apply_few_shot_prefix
from circuit_reuse.evaluate import (
    evaluate_accuracy,
    evaluate_accuracy_with_mean_ablation,
    compute_corrupted_means,
)
from circuit_reuse.circuit_extraction import Component
from cross_task_experiment import (
    parse_component_str,
    find_metrics_file,
    load_shared_components,
)


def main():
    parser = argparse.ArgumentParser(description="Mean-ablation cross-task confusion matrix.")
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--hf_revision", type=str, default=None)
    parser.add_argument("--tasks", type=str, required=True)
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--threshold", type=int, default=100)
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--score-filter", type=float, default=None, dest="score_filter")
    parser.add_argument("--num-examples", type=int, default=100)
    parser.add_argument("--digits", type=int, default=None)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    results_dir = Path(args.results_dir)

    random.seed(args.seed)

    model = load_model_any(args.model_name, device=args.device, revision=args.hf_revision)
    model.eval()

    baseline_acc, baseline_correct, baseline_total = {}, {}, {}
    datasets = {}
    for task in tasks:
        digits = args.digits if (args.digits and task == "addition") else 2
        ds = get_dataset(task, num_examples=args.num_examples, digits=digits)
        datasets[task] = apply_few_shot_prefix(list(ds), task, args.model_name)
        bc, bt = evaluate_accuracy(model, datasets[task], task=task)
        baseline_correct[task] = bc
        baseline_total[task] = bt
        baseline_acc[task] = bc / bt if bt > 0 else 0.0
        print(f"[BASELINE] {task}: {bc}/{bt} = {baseline_acc[task]*100:.1f}%")

    # Pre-compute corrupted-distribution means once per target task.
    print("\n[MEANS] Computing corrupted-distribution means per target task ...")
    means_by_task = {}
    for tgt_task in tasks:
        print(f"  [MEANS] {tgt_task}")
        means_by_task[tgt_task] = compute_corrupted_means(model, datasets[tgt_task])

    matrix_drop = {t: {} for t in tasks}
    matrix_norm = {t: {} for t in tasks}
    matrix_ablated = {t: {} for t in tasks}
    circuit_sizes = {}
    skipped_tasks = set()

    for src_task in tasks:
        try:
            metrics_path = find_metrics_file(
                results_dir, args.model_name, args.hf_revision, src_task,
                threshold=args.threshold, score_threshold=args.score_threshold,
                score_filter=args.score_filter,
            )
        except FileNotFoundError:
            print(f"[{src_task}] WARNING: no metrics found, skipping as source task")
            skipped_tasks.add(src_task)
            continue
        shared_components = load_shared_components(
            metrics_path, args.K, args.threshold,
            score_threshold=args.score_threshold,
            score_filter=args.score_filter,
        )
        circuit_sizes[src_task] = len(shared_components)
        print(f"[{src_task}] Loaded {len(shared_components)} shared components")

        for tgt_task in tasks:
            if not shared_components:
                acc = baseline_acc[tgt_task]
            else:
                correct, total = evaluate_accuracy_with_mean_ablation(
                    model, datasets[tgt_task], task=tgt_task,
                    removed=shared_components, means=means_by_task[tgt_task],
                )
                acc = correct / total if total > 0 else 0.0
            drop = baseline_acc[tgt_task] - acc
            matrix_drop[src_task][tgt_task] = drop * 100.0
            matrix_norm[src_task][tgt_task] = (
                drop / baseline_acc[tgt_task] * 100.0
                if baseline_acc[tgt_task] > 0 else 0.0
            )
            matrix_ablated[src_task][tgt_task] = acc * 100.0
            print(f"  -> {tgt_task}: baseline={baseline_acc[tgt_task]*100:.1f}% "
                  f"ablated={acc*100:.1f}% drop={drop*100:.1f}pp")

    active_tasks = [t for t in tasks if t not in skipped_tasks]
    csv_lines = []
    header = ["source_task"] + tasks
    csv_lines.append(",".join(header))
    for src_task in active_tasks:
        row = [src_task] + [f"{matrix_drop[src_task][t]:.3f}" for t in tasks]
        csv_lines.append(",".join(row))
    csv_text = "\n".join(csv_lines) + "\n"

    if skipped_tasks:
        print(f"\n[WARNING] Skipped source tasks (no metrics): {', '.join(sorted(skipped_tasks))}")
    print("\n=== Accuracy Drop (percentage points, mean ablation) ===")
    print(csv_text)

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        model_slug = args.model_name.replace("/", "_")
        rev_suffix = f"_{args.hf_revision}" if args.hf_revision else ""
        prefix = f"cross_task_{model_slug}{rev_suffix}_K{args.K}_t{args.threshold}_meanabl"

        csv_path = out_dir / f"{prefix}.csv"
        csv_path.write_text(csv_text)
        print(f"\n[SAVED] CSV: {csv_path}")

        output = {
            "model_name": args.model_name,
            "hf_revision": args.hf_revision,
            "K": args.K,
            "threshold": args.threshold,
            "ablation": "mean",
            "mean_source": "corrupted_prompt of target task",
            "num_examples": args.num_examples,
            "seed": args.seed,
            "tasks": tasks,
            "baseline_accuracy": {t: baseline_acc[t] for t in tasks},
            "circuit_sizes": circuit_sizes,
            "skipped_tasks": sorted(skipped_tasks),
            "accuracy_drop_pp": {src: {tgt: matrix_drop[src][tgt] for tgt in tasks}
                                  for src in active_tasks},
            "relative_drop_pct": {src: {tgt: matrix_norm[src][tgt] for tgt in tasks}
                                  for src in active_tasks},
            "ablated_accuracy_pct": {src: {tgt: matrix_ablated[src][tgt] for tgt in tasks}
                                      for src in active_tasks},
        }
        json_path = out_dir / f"{prefix}.json"
        with json_path.open("w") as f:
            json.dump(output, f, indent=2)
        print(f"[SAVED] JSON: {json_path}")


if __name__ == "__main__":
    main()
