"""Sufficiency of a task's circuit: keep the circuit, replace every other unit of the
same granularity with its mean over the task's corrupted prompts (per position), and
compare against keeping nothing and against random circuits of the same composition.

Faithfulness is (m(circuit) - m(nothing)) / (m(full) - m(nothing)) for accuracy and
for the gold log-probability. At neuron granularity only MLP neurons are replaced;
attention is left intact.

    python -m followups.sufficiency --model meta-llama/Llama-3.2-3B
"""
import argparse
import random
from collections import Counter

from circuit_reuse.circuit_extraction import Component
from circuit_reuse.evaluate import _build_mean_ablation_hooks, compute_corrupted_means, evaluate_graded, summarize_graded
from followups.common import OUT, TASKS, circuit, eval_datasets, load_model, write_json


def all_units(model, granularity: str) -> list[Component]:
    L, H, d_mlp = model.cfg.n_layers, model.cfg.n_heads, model.cfg.d_mlp
    if granularity == "neuron":
        return [Component(l, "neuron", i) for l in range(L) for i in range(d_mlp)]
    return [Component(l, "head", h) for l in range(L) for h in range(H)] + [Component(l, "mlp", 0) for l in range(L)]


def matched_random(units: list[Component], kept: list[Component], rng: random.Random) -> set[Component]:
    """A random set with the same number of units of each kind (and, for neurons, per layer)."""
    key = (lambda c: (c.kind, c.layer)) if kept and kept[0].kind == "neuron" else (lambda c: c.kind)
    pools: dict = {}
    for u in units:
        pools.setdefault(key(u), []).append(u)
    return {u for k, n in Counter(map(key, kept)).items() for u in rng.sample(pools[k], n)}


def faithfulness(kept: float, full: float, none: float) -> float:
    return (kept - none) / (full - none) if full != none else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-examples", type=int, default=100, help="evaluation examples per task")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B")
    parser.add_argument("--granularities", default="head_mlp,neuron")
    parser.add_argument("--method", default="eap_ig")
    parser.add_argument("--n-random", type=int, default=3)
    args = parser.parse_args()
    model = load_model(args.model)
    datasets = eval_datasets(args.model, num_examples=args.num_examples)
    rng = random.Random(0)
    for gran in args.granularities.split(","):
        path = OUT / "sufficiency" / f"{args.model.replace('/', '_')}__{args.method}_{gran}.json"
        results = {}
        units = all_units(model, gran)
        kinds = ("neuron",) if gran == "neuron" else ("head", "mlp")
        for task in TASKS:
            ds = datasets[task]
            kept = circuit(args.model, task, args.method, gran)
            means = compute_corrupted_means(model, ds, per_position=True, kinds=kinds)

            def keep(subset):
                subset = set(subset)
                hooks = _build_mean_ablation_hooks([u for u in units if u not in subset], means)
                return summarize_graded(evaluate_graded(model, ds, task, hooks))

            full = summarize_graded(evaluate_graded(model, ds, task))
            none, own = keep(()), keep(kept)
            rand = [keep(matched_random(units, kept, rng)) for _ in range(args.n_random)]
            results[task] = {
                "circuit_size": len(kept), "full": full, "nothing": none, "circuit": own, "random": rand,
                "faithfulness_accuracy": faithfulness(own["correct"], full["correct"], none["correct"]),
                "faithfulness_logprob": faithfulness(own["logprob"], full["logprob"], none["logprob"]),
                "random_faithfulness_logprob": [faithfulness(r["logprob"], full["logprob"], none["logprob"]) for r in rand],
            }
            print(f"[{gran}/{task}] |S|={len(kept)} faithfulness acc={results[task]['faithfulness_accuracy']:.2f} "
                  f"logprob={results[task]['faithfulness_logprob']:.2f}")
            write_json(path, {"model": args.model, "method": args.method, "granularity": gran, "tasks": results})


if __name__ == "__main__":
    main()
