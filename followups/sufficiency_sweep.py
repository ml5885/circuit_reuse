"""Sufficiency against circuit size: for every task, keep the task circuit at each consensus
threshold P (nested circuits at K=10%), replace every other unit of the same granularity by its
per-position mean over the task's corrupted prompts, and record accuracy and gold log-prob, next
to a random circuit of the same composition. Together with the full model and the model with
nothing kept this gives faithfulness curves over circuit size, as in MIB and Arora et al.

--sweep K instead sweeps top-K at P=50. --keep-mlps (component level) replaces only the heads
outside the circuit and keeps every MLP, as in Wang et al.'s IOI circuit evaluation; the random
circuit then matches the circuit's head count. At neuron level attention is always kept.

    python -m followups.sufficiency_sweep --model meta-llama/Llama-3.2-3B --granularity head_mlp
"""
import argparse
import json
import random
from pathlib import Path

from circuit_reuse.evaluate import _build_mean_ablation_hooks, compute_corrupted_means, evaluate_graded, summarize_graded
from cross_task_experiment import find_metrics_file, parse_component_str
from followups.common import EXTRACT, OUT, TASKS, eval_datasets, load_model, write_json
from followups.sufficiency import all_units, matched_random

PS = ("50", "75", "85", "90", "95", "100")
KS = ("1", "5", "10", "20", "30")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--granularity", default="head_mlp")
    parser.add_argument("--method", default="eap_ig")
    parser.add_argument("--K", default="10", help="top-K for --sweep P")
    parser.add_argument("--P", default="50", help="consensus threshold for --sweep K")
    parser.add_argument("--sweep", choices=("P", "K"), default="P")
    parser.add_argument("--keep-mlps", action="store_true")
    parser.add_argument("--results-dir", default=None, help="extraction root (default: the paper's)")
    parser.add_argument("--output-dir", default=str(OUT / "sufficiency_sweep"))
    args = parser.parse_args()
    model = load_model(args.model)
    datasets = eval_datasets(args.model)
    units = all_units(model, args.granularity)
    if args.keep_mlps:
        units = [u for u in units if u.kind == "head"]
    kinds = ("neuron",) if args.granularity == "neuron" else ("head",) if args.keep_mlps else ("head", "mlp")
    rng = random.Random(0)
    root = Path(args.results_dir or EXTRACT) / f"granularity_parity_{args.method}_{args.granularity}"
    suffix = ("_sweepK" if args.sweep == "K" else "") + ("_keepmlp" if args.keep_mlps else "")
    path = Path(args.output_dir) / f"{args.model.replace('/', '_')}__{args.method}_{args.granularity}{suffix}.json"
    cells = [(K, args.P) for K in KS] if args.sweep == "K" else [(args.K, P) for P in PS]
    results = {}
    for task in TASKS:
        ds = datasets[task]
        means = compute_corrupted_means(model, ds, per_position=True, kinds=kinds)
        m = json.loads(find_metrics_file(root, args.model, None, task, method=args.method,
                                         granularity=args.granularity).read_text())

        def keep(subset):
            subset = set(subset)
            hooks = _build_mean_ablation_hooks([u for u in units if u not in subset], means)
            return summarize_graded(evaluate_graded(model, ds, task, hooks))

        res = {"full": summarize_graded(evaluate_graded(model, ds, task)), "nothing": keep(()), "points": {}}
        for K, P in cells:
            entry = m["by_k"][K]["thresholds"].get(P)
            if not entry or not entry["shared_components"]:
                continue
            kept = [parse_component_str(c) for c in entry["shared_components"]]
            if args.keep_mlps:
                kept = [c for c in kept if c.kind == "head"]
            res["points"][K if args.sweep == "K" else P] = {
                "size": len(kept), "circuit": keep(kept), "random": keep(matched_random(units, kept, rng))}
        results[task] = res
        print(f"[{task}] " + " ".join(f"{args.sweep}{x}:{v['size']}->{v['circuit']['correct']:.2f}"
                                      for x, v in res["points"].items()), flush=True)
        write_json(path, {"model": args.model, "method": args.method, "granularity": args.granularity,
                          "sweep": args.sweep, "K": args.K, "P": args.P, "keep_mlps": args.keep_mlps,
                          "tasks": results})


if __name__ == "__main__":
    main()
