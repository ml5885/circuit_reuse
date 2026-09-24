"""Re-evaluate every IOI accuracy the paper reports, with the fixed label scoring.

The old scorer let both names tie on the newline or colon token, and ties went to the
gold name, so ablations that destroyed the model's preference still counted as correct.
Circuits are unchanged; only the IOI evaluations are redone.

``necessity``: for every K and P of the paper's IOI extraction, the accuracy on the 800
training examples with the task circuit ablated and with the capacity-conserved control
ablated. The control is redrawn with the stored seed; for neuron circuits this is the
same control, for component circuits it may differ in which heads and MLP blocks it holds
(the draw order of the two kinds depends on Python's string hashing), not in how many.

``cross``: the IOI column of the cross-task matrices (every donor's circuit ablated, IOI
evaluated), zero ablation at every K and P, and per-position mean ablation at K=10.

``selective``: the IOI evaluations of the selective-ablation experiment (EAP component
circuits at P=100), for every task pair that includes IOI as target or evaluation task.

    python -m followups.reeval_ioi necessity --model google/gemma-2-2b --method eap_ig --granularity head_mlp
"""
import argparse
import json
import random
from pathlib import Path

from circuit_reuse.evaluate import (_build_ablation_hooks, _build_mean_ablation_hooks, compute_corrupted_means,
                                    evaluate_graded)
from cross_task_experiment import find_metrics_file, load_shared_components, parse_component_str
from followups.common import EXTRACT, OUT, TASKS, eval_datasets, load_model, train_examples, write_json
from main_experiment import _enumerate_all_components, _sample_control_components


class Evaluator:
    """IOI accuracy (and gold log-prob) under an ablation, cached by the set of units removed."""

    def __init__(self, model, examples, means=None):
        self.model, self.examples, self.means, self.cache = model, examples, means, {}

    def __call__(self, removed, ablation="zero"):
        key = (ablation, frozenset(removed))
        if key not in self.cache:
            hooks = [] if not removed else (_build_ablation_hooks(removed) if ablation == "zero"
                                            else _build_mean_ablation_hooks(removed, self.means))
            rows = evaluate_graded(self.model, self.examples, "ioi", hooks)
            self.cache[key] = {"accuracy": sum(r["correct"] for r in rows) / len(rows),
                               "logit_diff": sum(r["logit_diff"] for r in rows) / len(rows)}
        return self.cache[key]


def necessity(model, name, method, gran):
    path = find_metrics_file(EXTRACT / f"granularity_parity_{method}_{gran}", name, None, "ioi",
                             method=method, granularity=gran)
    m = json.loads(path.read_text())
    ev = Evaluator(model, train_examples("ioi", name))
    units = _enumerate_all_components(model, granularity=gran, method=method)
    base = ev(())
    out = {"baseline": base, "cells": {}}
    for K, by_k in m["by_k"].items():
        for P, e in by_k["thresholds"].items():
            shared = [parse_component_str(c) for c in e["shared_components"]]
            if not shared:
                continue
            control = _sample_control_components(shared, units, random.Random(e["rng_seed"]))
            abl, ctrl = ev(shared), ev(control)
            t = e["train"]
            out["cells"][f"K{K}_P{P}"] = {
                "size": len(shared), "ablated": abl, "control": ctrl,
                "necessity": (ctrl["accuracy"] - abl["accuracy"]) / base["accuracy"],
                "old": {"ablated": t["ablation_accuracy"], "control": t["control_accuracy"],
                        "baseline": m["baseline_train_accuracy"],
                        "necessity": (t["control_accuracy"] - t["ablation_accuracy"]) / m["baseline_train_accuracy"]},
            }
            print(f"K{K} P{P} |S|={len(shared)} necessity {out['cells'][f'K{K}_P{P}']['old']['necessity']:+.2f} -> "
                  f"{out['cells'][f'K{K}_P{P}']['necessity']:+.2f}", flush=True)
    return out


def cross(model, name, method, gran):
    root = EXTRACT / f"granularity_parity_{method}_{gran}"
    ds = eval_datasets(name)["ioi"]
    kinds = ("neuron",) if gran == "neuron" else ("head", "mlp")
    zero = Evaluator(model, ds)
    mean = Evaluator(model, ds, compute_corrupted_means(model, ds, per_position=True, kinds=kinds))
    metrics = {t: json.loads(find_metrics_file(root, name, None, t, method=method, granularity=gran).read_text())
               for t in TASKS}
    out = {"baseline": zero(()), "cells": {}}
    for K, by_k in metrics["ioi"]["by_k"].items():
        for P in by_k["thresholds"]:
            cell = {}
            for donor, m in metrics.items():
                removed = [parse_component_str(c) for c in m["by_k"][K]["thresholds"].get(P, {}).get("shared_components", [])]
                cell[donor] = {"zero": zero(removed)}
                if K == "10" and P in ("50", "75", "85", "100"):
                    cell[donor]["mean_pos"] = mean(removed, "mean_pos")
            out["cells"][f"K{K}_P{P}"] = cell
        print(f"K{K} done", flush=True)
    return out


def selective(model, name, results_dir: Path, K_list=(1, 5, 10, 20, 30), P=100):
    ds = eval_datasets(name)["ioi"]
    ev = Evaluator(model, ds)
    units = _enumerate_all_components(model)
    out = {"baseline": ev(()), "pairs": {}}
    for K in K_list:
        circuits = {t: set(load_shared_components(find_metrics_file(results_dir, name, None, t), K, P)) for t in TASKS}
        for a in TASKS:
            for b in TASKS:
                if a == b:
                    continue
                g, ra, rb = circuits[a] & circuits[b], circuits[a] - circuits[b], circuits[b] - circuits[a]
                control = _sample_control_components(list(ra), units, random.Random(42))
                out["pairs"][f"K{K}_{a}_vs_{b}"] = {
                    "shared_core": ev(list(g)), "residual_a": ev(list(ra)),
                    "residual_b": ev(list(rb)), "random_control": ev(control)}
        print(f"K{K} done", flush=True)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["necessity", "cross", "selective"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--method", default="eap_ig")
    parser.add_argument("--granularity", default="head_mlp")
    parser.add_argument("--selective-results", default="results/cross_task")
    args = parser.parse_args()
    model = load_model(args.model)
    slug = args.model.replace("/", "_")
    if args.mode == "selective":
        res, path = selective(model, args.model, Path(args.selective_results)), OUT / "reeval_ioi" / f"selective__{slug}.json"
    else:
        fn = necessity if args.mode == "necessity" else cross
        res = fn(model, args.model, args.method, args.granularity)
        path = OUT / "reeval_ioi" / f"{args.mode}__{slug}__{args.method}_{args.granularity}.json"
    write_json(path, {"model": args.model, "method": args.method, "granularity": args.granularity, **res})


if __name__ == "__main__":
    main()
