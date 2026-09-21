"""Efficient cross-task ablation sweep.

One invocation loads one model and one dataset per task, then evaluates the
complete ``K x p`` grid.  Outputs are one resumable JSON matrix per setting.
The old scalar CLI remains valid (``--K 10 --threshold 100``).

``--ablation`` selects one or more ablations, computed in one process on the
same evaluation examples: ``zero``; ``mean``, which replaces each removed
component with its activation averaged over positions and over the target
task's corrupted prompts (Wang et al. 2022, Miller et al. 2024); and
``mean_pos``, which averages per token position. Outputs carry the suffixes
``_meanabl`` and ``_meanabl_pos``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Iterable

import torch

from models.olmo_adapter import load_model_any
from circuit_reuse.dataset import get_dataset, apply_few_shot_prefix
from circuit_reuse.evaluate import (
    compute_corrupted_means,
    evaluate_accuracy,
    evaluate_accuracy_with_ablation,
    evaluate_accuracy_with_mean_ablation,
)
from circuit_reuse.circuit_extraction import Component


def parse_component_str(s: str) -> Component:
    kind, rest = s.split("[", 1)
    values = {}
    for part in rest.rstrip("]").split(","):
        key, value = part.split("=", 1)
        values[key.strip()] = int(value.strip())
    return Component(layer=values["layer"], kind=kind.strip(), index=values["index"])


def parse_int_list(value: str | int | Iterable[int]) -> list[int]:
    if isinstance(value, int):
        return [value]
    if not isinstance(value, str):
        return [int(x) for x in value]
    values = [int(x.strip()) for x in value.replace(";", ",").split(",") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return sorted(set(values))


def _matches(data: dict, model_name: str, hf_revision: str | None, task: str,
             method: str | None = None, granularity: str | None = None) -> bool:
    actual_granularity = data.get("granularity")
    if actual_granularity is None and granularity is not None:
        # Older parity extraction metrics predate the explicit field. Infer
        # the basis from one stored shared component.
        sample = ""
        for by_k in data.get("by_k", {}).values():
            for entry in by_k.get("thresholds", {}).values():
                components = entry.get("shared_components", [])
                if components:
                    sample = str(components[0])
                    break
            if sample:
                break
        actual_granularity = "neuron" if sample.startswith("neuron[") else "head_mlp"
    return (
        data.get("model_name") == model_name
        and str(data.get("hf_revision") or "none") == str(hf_revision or "none")
        and data.get("task") == task
        and (method is None or data.get("method", "eap") == method)
        and (granularity is None or actual_granularity == granularity)
    )


def find_metrics_file(results_dir, model_name, hf_revision, task, threshold=None,
                      score_threshold=None, score_filter=None, method=None,
                      granularity=None) -> Path:
    candidates = []
    for path in Path(results_dir).rglob("metrics.json"):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if _matches(data, model_name, hf_revision, task, method, granularity):
            candidates.append((path, data))
    if not candidates:
        raise FileNotFoundError(f"No matching metrics.json for {model_name}/{task}")

    def has_setting(data):
        if score_filter is not None:
            return str(score_filter) in data.get("by_score_filter", {})
        if score_threshold is not None:
            return str(score_threshold) in data.get("by_threshold", {})
        return any(str(threshold) in entry.get("thresholds", {})
                   for entry in data.get("by_k", {}).values())
    matching = [item for item in candidates if has_setting(item[1])]
    return sorted(matching or candidates, key=lambda x: str(x[0]))[0][0]


def load_shared_components(metrics_path: Path, K: int, threshold: int,
                           score_threshold=None, score_filter=None) -> list[Component]:
    data = json.loads(Path(metrics_path).read_text())
    if score_filter is not None:
        entry = data["by_score_filter"][str(score_filter)]["thresholds"][str(threshold)]
    elif score_threshold is not None:
        entry = data["by_threshold"][str(score_threshold)]["thresholds"][str(threshold)]
    else:
        entry = data["by_k"][str(K)]["thresholds"][str(threshold)]
    return [parse_component_str(item) for item in entry.get("shared_components", [])]


def output_path(output_dir: Path, model_name: str, method: str, granularity: str,
                K: int, threshold: int, score_threshold=None, score_filter=None,
                ablation: str = "zero") -> Path:
    model_slug = model_name.replace("/", "_")
    if score_filter is not None:
        stem = f"cross_task_{model_slug}_{method}_{granularity}_sf{score_filter}_p{threshold}"
    elif score_threshold is not None:
        stem = f"cross_task_{model_slug}_{method}_{granularity}_tau{score_threshold}_p{threshold}"
    else:
        stem = f"cross_task_{model_slug}_{method}_{granularity}_K{K}_p{threshold}"
    stem += {"zero": "", "mean": "_meanabl", "mean_pos": "_meanabl_pos"}[ablation]
    return output_dir / f"{stem}.json"


def valid_output(path: Path, tasks: list[str]) -> bool:
    try:
        data = json.loads(path.read_text())
        matrix = data["cells"]
        return (data.get("schema_version", 0) >= 2 and data.get("tasks") == tasks
                and len(matrix) == len(tasks)
                and all(len(matrix[src]) == len(tasks) for src in tasks)
                and not data.get("skipped_donors") and not data.get("skipped_targets"))
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return False


def run_sweep(args):
    tasks = [x.strip() for x in args.tasks.split(",") if x.strip()]
    refresh = {x.strip() for x in (args.refresh_tasks or "").split(",") if x.strip()}
    ablations = [x.strip() for x in args.ablation.split(",") if x.strip()]
    Ks = parse_int_list(args.K)
    thresholds = parse_int_list(args.threshold)
    out_dir = Path(args.output_dir) if args.output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    # The generated tasks (addition, boolean) draw their examples from the global RNG.
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = load_model_any(args.model_name, device=args.device, revision=args.hf_revision)
    model.eval()
    datasets, baseline = {}, {}
    for task in tasks:
        digits = args.digits if args.digits is not None and task == "addition" else 2
        ds = apply_few_shot_prefix(list(get_dataset(task, num_examples=args.num_examples,
                                                     digits=digits)), task, args.model_name)
        datasets[task] = ds
        correct, total = evaluate_accuracy(model, ds, task=task)
        baseline[task] = {"correct": int(correct), "total": int(total),
                          "accuracy": correct / total if total else float("nan")}
    means = {ab: {task: compute_corrupted_means(model, datasets[task], per_position=ab == "mean_pos")
                  for task in tasks}
             for ab in ablations if ab != "zero"}

    component_cache: dict[tuple[str, int, int], tuple[Component, ...]] = {}
    metric_cache = {}
    completed = 0
    for K, threshold, ablation in ((K, t, ab) for K in Ks for t in thresholds for ab in ablations):
        path = output_path(out_dir, args.model_name, args.method, args.granularity,
                           K, threshold, args.score_threshold, args.score_filter,
                           ablation) if out_dir else None
        existing = None
        if path and valid_output(path, tasks):
            if not refresh:
                completed += 1
                continue
            existing = json.loads(path.read_text())
            if refresh <= set(existing.get("refreshed", [])):
                completed += 1
                continue
        skipped_donors, skipped_targets = [], []
        for donor in tasks:
            try:
                if donor not in metric_cache:
                    metric_cache[donor] = find_metrics_file(
                        args.results_dir, args.model_name, args.hf_revision, donor,
                        threshold, args.score_threshold, args.score_filter,
                        args.method, args.granularity)
                key = (donor, K, threshold)
                if key not in component_cache:
                    component_cache[key] = tuple(load_shared_components(
                        metric_cache[donor], K, threshold, args.score_threshold, args.score_filter))
            except (FileNotFoundError, KeyError):
                skipped_donors.append(donor)

        cells = {donor: {} for donor in tasks if donor not in skipped_donors}
        for donor in tasks:
            if donor in skipped_donors:
                continue
            removed = component_cache[(donor, K, threshold)]
            for target in tasks:
                if target in skipped_targets:
                    continue
                if existing is not None and donor not in refresh and target not in refresh:
                    cells[donor][target] = existing["cells"][donor][target]
                    continue
                base = baseline[target]
                if not removed:  # empty circuit: ablating nothing is the baseline
                    correct, total = base["correct"], base["total"]
                elif ablation == "zero":
                    correct, total = evaluate_accuracy_with_ablation(
                        model, datasets[target], task=target, removed=removed)
                else:
                    correct, total = evaluate_accuracy_with_mean_ablation(
                        model, datasets[target], task=target, removed=removed,
                        means=means[ablation][target])
                acc = correct / total if total else float("nan")
                cells[donor][target] = {
                    "baseline_correct": base["correct"], "baseline_total": base["total"],
                    "baseline_accuracy": base["accuracy"],
                    "ablated_correct": int(correct), "ablated_total": int(total),
                    "ablated_accuracy": acc,
                    "accuracy_drop_pp": (base["accuracy"] - acc) * 100 if total else float("nan"),
                    "relative_drop_pct": ((base["accuracy"] - acc) / base["accuracy"] * 100
                                          if base["accuracy"] else float("nan")),
                    "evaluated_samples": int(total),
                }
        result = {
            "schema_version": 2, "model_name": args.model_name,
            "hf_revision": args.hf_revision, "method": args.method,
            "granularity": args.granularity, "ablation": ablation,
            "K": K, "threshold": threshold, "num_examples": args.num_examples, "seed": args.seed, "tasks": tasks,
            "baseline": baseline, "circuit_sizes": {
                donor: len(component_cache[(donor, K, threshold)])
                for donor in tasks if donor not in skipped_donors},
            "skipped_donors": skipped_donors, "skipped_targets": skipped_targets,
            "cells": cells,
        }
        if refresh:
            result["refreshed"] = sorted(refresh | set(existing.get("refreshed", []) if existing else []))
        if path:
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(result, indent=2, allow_nan=False))
            os.replace(tmp, path)
        else:
            print(json.dumps(result, indent=2, allow_nan=False))
        completed += 1
    print(f"[DONE] {completed}/{len(Ks) * len(thresholds) * len(ablations)} matrices complete")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--hf_revision", default=None)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--K", default="10", help="integer or comma-separated list")
    parser.add_argument("--threshold", default="100", help="integer or comma-separated list")
    parser.add_argument("--method", default="eap")
    parser.add_argument("--granularity", default="head_mlp")
    parser.add_argument("--ablation", default="zero",
                        help="comma-separated subset of zero, mean, mean_pos; all are evaluated "
                             "in one process on the same examples")
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--score-filter", type=float, default=None, dest="score_filter")
    parser.add_argument("--num-examples", type=int, default=100)
    parser.add_argument("--digits", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--refresh-tasks", default=None,
                        help="Comma-separated tasks whose cells (as donor or target) are recomputed; "
                             "other cells are copied from the existing output file.")
    run_sweep(parser.parse_args())


if __name__ == "__main__":
    main()
