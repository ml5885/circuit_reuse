"""
Computes a confusion matrix of accuracy drops from ablating shared circuits across tasks.
"""

import argparse
import json
import os
import random
from pathlib import Path

import torch

from models.olmo_adapter import load_model_any
from circuit_reuse.dataset import get_dataset
from circuit_reuse.evaluate import evaluate_accuracy_with_ablation, evaluate_accuracy
from circuit_reuse.circuit_extraction import Component


def parse_component_str(s):
    """Parse a string representation of a component into a Component object.

    The expected format is either ``"head[layer=3, index=2]"`` or
    ``"mlp[layer=4, index=0]"``. Whitespace around commas is ignored.
    """
    kind, rest = s.split("[", 1)
    rest = rest.rstrip("]")
    parts = rest.split(",")
    layer = None
    index = None
    for part in parts:
        k, v = part.split("=")
        k = k.strip()
        v = v.strip()
        if k == "layer":
            layer = int(v)
        elif k == "index":
            index = int(v)
    if layer is None or index is None:
        raise ValueError(f"Unable to parse component string: {s}")
    return Component(layer=layer, kind=kind.strip(), index=index)


def find_metrics_file(results_dir, model_name, hf_revision, task):
    """Search results_dir recursively for a metrics file matching the given settings.

    Returns the path to the first matching metrics.json file.  Raises an
    exception if no match is found.
    """
    for root, _, files in os.walk(results_dir):
        if "metrics.json" in files:
            path = Path(root) / "metrics.json"
            with path.open() as f:
                data = json.load(f)
            if (
                data["model_name"] == model_name
                and str(data["hf_revision"] or "none") == (hf_revision or "none")
                and data["task"] == task
            ):
                return path

    raise FileNotFoundError(
        f"No metrics.json found in {results_dir} for model={model_name}, revision={hf_revision}, task={task}."
    )


def load_shared_components(metrics_path, K, threshold):
    """Load the list of shared components for a given K and threshold from a metrics file."""
    with metrics_path.open() as f:
        data = json.load(f)
    by_k = data["by_k"]
    entry = by_k[str(K)]
    thresholds = entry["thresholds"]
    thr_entry = thresholds[str(threshold)]
    
    return [parse_component_str(c) for c in thr_entry["shared_components"]]


def main():
    parser = argparse.ArgumentParser(description="Compute cross-task ablation confusion matrix.")
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Root directory containing run subdirectories with metrics.json files.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name used in the experiments.",
    )
    parser.add_argument(
        "--hf_revision",
        type=str,
        default=None,
        help="Revision/tag of the model used in the experiments.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        required=True,
        help="Comma-separated list of task names to include in the matrix.",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=10,
        help="Top-K percentage used when constructing shared circuits.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=100,
        help="Reuse threshold percentage (p) used to define the shared circuit.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=100,
        help="Number of examples to use for each task.",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=None,
        help="Number of digits for addition task (only used for addition).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device on which to run evaluations.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset generation (default: 42).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save output CSV and JSON. Prints CSV to stdout if not set.",
    )
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    results_dir = Path(args.results_dir)

    random.seed(args.seed)

    # Load model once
    model = load_model_any(args.model_name, device=args.device, revision=args.hf_revision)
    model.eval()

    # Baseline accuracies per task (to avoid recomputation)
    baseline_acc = {}
    baseline_correct = {}
    baseline_total = {}
    # Datasets per task
    datasets = {}
    for task in tasks:
        digits = args.digits if (args.digits and task == "addition") else 2
        ds = get_dataset(task, num_examples=args.num_examples, digits=digits)
        datasets[task] = list(ds)
        bc, bt = evaluate_accuracy(model, datasets[task], task=task)
        baseline_correct[task] = bc
        baseline_total[task] = bt
        baseline_acc[task] = bc / bt if bt > 0 else 0.0

    # For each source task, load shared components and compute drops on all tasks
    matrix_drop = {t: {} for t in tasks}       # raw accuracy drop (pp)
    matrix_norm = {t: {} for t in tasks}       # drop / baseline (relative)
    matrix_ablated = {t: {} for t in tasks}    # ablated accuracy
    circuit_sizes = {}
    for src_task in tasks:
        metrics_path = find_metrics_file(results_dir, args.model_name, args.hf_revision, src_task)
        shared_components = load_shared_components(metrics_path, args.K, args.threshold)
        circuit_sizes[src_task] = len(shared_components)
        print(f"[{src_task}] Loaded {len(shared_components)} shared components")
        for tgt_task in tasks:
            correct, total = evaluate_accuracy_with_ablation(
                model, datasets[tgt_task], task=tgt_task, removed=shared_components
            )
            acc = correct / total if total > 0 else 0.0
            drop = baseline_acc[tgt_task] - acc
            matrix_drop[src_task][tgt_task] = drop * 100.0
            matrix_norm[src_task][tgt_task] = (drop / baseline_acc[tgt_task] * 100.0) if baseline_acc[tgt_task] > 0 else 0.0
            matrix_ablated[src_task][tgt_task] = acc * 100.0
            print(f"  -> {tgt_task}: baseline={baseline_acc[tgt_task]*100:.1f}% ablated={acc*100:.1f}% drop={drop*100:.1f}pp")

    # Build CSV lines
    csv_lines = []
    header = ["source_task"] + tasks
    csv_lines.append(",".join(header))
    for src_task in tasks:
        row = [src_task] + [f"{matrix_drop[src_task][t]:.3f}" for t in tasks]
        csv_lines.append(",".join(row))
    csv_text = "\n".join(csv_lines) + "\n"

    # Always print CSV to stdout
    print("\n=== Accuracy Drop (percentage points) ===")
    print(csv_text)

    print("=== Relative Drop (% of baseline) ===")
    norm_header = ["source_task"] + tasks
    print(",".join(norm_header))
    for src_task in tasks:
        row = [src_task] + [f"{matrix_norm[src_task][t]:.3f}" for t in tasks]
        print(",".join(row))

    # Save to output directory if specified
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        model_slug = args.model_name.replace("/", "_")
        prefix = f"cross_task_{model_slug}_K{args.K}_t{args.threshold}"

        # Save CSV
        csv_path = out_dir / f"{prefix}.csv"
        csv_path.write_text(csv_text)
        print(f"\n[SAVED] CSV: {csv_path}")

        # Save structured JSON
        output = {
            "model_name": args.model_name,
            "hf_revision": args.hf_revision,
            "K": args.K,
            "threshold": args.threshold,
            "num_examples": args.num_examples,
            "seed": args.seed,
            "tasks": tasks,
            "baseline_accuracy": {t: baseline_acc[t] for t in tasks},
            "circuit_sizes": circuit_sizes,
            "accuracy_drop_pp": {src: {tgt: matrix_drop[src][tgt] for tgt in tasks} for src in tasks},
            "relative_drop_pct": {src: {tgt: matrix_norm[src][tgt] for tgt in tasks} for src in tasks},
            "ablated_accuracy_pct": {src: {tgt: matrix_ablated[src][tgt] for tgt in tasks} for src in tasks},
        }
        json_path = out_dir / f"{prefix}.json"
        with json_path.open("w") as f:
            json.dump(output, f, indent=2)
        print(f"[SAVED] JSON: {json_path}")


if __name__ == "__main__":
    main()
