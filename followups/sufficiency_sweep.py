"""Sufficiency against circuit size: for every task, keep the task circuit at each consensus
threshold P (nested circuits at K=10%), replace every other unit of the same granularity by its
per-position mean over the task's corrupted prompts, and record accuracy and gold log-prob, next
to a random circuit of the same composition. Together with the full model and the model with
nothing kept this gives faithfulness curves over circuit size, as in MIB and Arora et al.

--sweep K instead sweeps top-K at P=50. --keep-mlps (component level) replaces only the heads
outside the circuit and keeps every MLP, as in Wang et al.'s IOI circuit evaluation; the random
circuit then matches the circuit's head count. At neuron level attention is always kept.

--align end averages each position with the positions at the same distance from the end of the
other corrupted prompts (the BOS token stays aligned), so the last token's mean is a mean over
last tokens when prompts differ in length. Addition keeps start alignment: its prompts all have
the same length, and its scoring runs on past the prompt's end into the answer tokens.

--granularity feature attaches SAEs or a CLT (--sae-* as in main_experiment) and keeps the
feature circuit from --circuits-dir. --errors kept leaves each layer's error term (the SAE's
reconstruction error, or the MLP output minus the CLT's reconstruction) as it is; --errors mean
replaces it with its mean over the corrupted prompts too, so only the circuit's features carry
input-specific information.

    python -m followups.sufficiency_sweep --model meta-llama/Llama-3.2-3B --granularity head_mlp
"""
import argparse
import json
import random
from pathlib import Path

from circuit_reuse.circuit_extraction import Component
from circuit_reuse.evaluate import _build_mean_ablation_hooks, compute_corrupted_means, evaluate_graded, summarize_graded
from cross_task_experiment import find_metrics_file, parse_component_str
from followups.common import EXTRACT, OUT, TASKS, eval_datasets, load_model, write_json
from circuit_reuse.sae import add_sae_args, attach_saes, attached_sae_layers, spec_from_args
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
    parser.add_argument("--align", choices=("start", "end"), default="start")
    parser.add_argument("--results-dir", default=None, help="extraction root (default: the paper's)")
    parser.add_argument("--output-dir", default=str(OUT / "sufficiency_sweep"))
    parser.add_argument("--circuits-dir", default=None, help="extraction folder of the circuits (default: the parity run's)")
    parser.add_argument("--errors", choices=("kept", "mean"), default="kept", help="feature level: the error terms")
    parser.add_argument("--tasks", default=",".join(TASKS))
    add_sae_args(parser)
    args = parser.parse_args()
    model = load_model(args.model)
    datasets = eval_datasets(args.model)
    gran = args.granularity
    if gran == "feature":
        spec = spec_from_args(args)
        attach_saes(model, spec)
        gran = f"feature-{'clt' if spec.kind == 'clt' else 'sae' + spec.width}"
    units = all_units(model, args.granularity)
    if args.keep_mlps:
        units = [u for u in units if u.kind == "head"]
    errors = ([Component(l, "error", 0) for l in attached_sae_layers(model)]
              if args.granularity == "feature" and args.errors == "mean" else [])
    kinds = {"neuron": ("neuron",), "feature": ("feature", "error") if errors else ("feature",)}.get(
        args.granularity, ("head",) if args.keep_mlps else ("head", "mlp"))
    rng = random.Random(0)
    root = Path(args.circuits_dir or Path(args.results_dir or EXTRACT) / f"granularity_parity_{args.method}_{args.granularity}")
    suffix = (("_sweepK" if args.sweep == "K" else "") + ("_keepmlp" if args.keep_mlps else "")
              + ("_alignend" if args.align == "end" else "") + (f"_errors{args.errors}" if args.granularity == "feature" else ""))
    path = Path(args.output_dir) / f"{args.model.replace('/', '_')}__{args.method}_{gran}{suffix}.json"
    cells = [(K, args.P) for K in KS] if args.sweep == "K" else [(args.K, P) for P in PS]
    results = {}
    for task in args.tasks.split(","):
        ds = datasets[task]
        means = compute_corrupted_means(model, ds, per_position=True, kinds=kinds,
                                        align="start" if task == "addition" else args.align)
        m = json.loads(find_metrics_file(root, args.model, None, task, method=args.method,
                                         granularity=args.granularity).read_text())

        def keep(subset):
            subset = set(subset)
            hooks = _build_mean_ablation_hooks([u for u in units if u not in subset] + errors, means)
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
        write_json(path, {"model": args.model, "method": args.method, "granularity": gran, "errors": args.errors,
                          "sweep": args.sweep, "K": args.K, "P": args.P, "keep_mlps": args.keep_mlps,
                          "align": args.align,
                          "tasks": results})


if __name__ == "__main__":
    main()
