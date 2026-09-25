"""Do an example's own top-K% neurons matter for that example? For the first training examples of a
task, ablate the example's own EAP-IG top-10% neurons (from the compacted attribution cache) and,
separately, as many random neurons per layer, and record whether the example is still answered
correctly. Separates "the neurons attribution ranks highest do not matter" from "they matter but
differ between examples, so the consensus circuit misses them".

    python -m followups.per_example_neurons --models google/gemma-2-2b,qwen3-4b --tasks ioi,arc_easy
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

from circuit_reuse.circuit_extraction import Component
from circuit_reuse.evaluate import evaluate_accuracy, evaluate_accuracy_with_ablation
from followups.common import load_model, train_examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", required=True)
    parser.add_argument("--tasks", default="ioi,arc_easy")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--compact-dir", default="results/rerun/compact")
    parser.add_argument("--output", default="results/rerun/per_example_neurons.json")
    args = parser.parse_args()
    out = {}
    for name in args.models.split(","):
        model = load_model(name)
        rng = random.Random(0)
        for task in args.tasks.split(","):
            z = np.load(Path(args.compact_dir) / f"eap_ig_neuron__{name.replace('/', '_')}__{task}.npz")
            layer, index = z["layer"], z["index"]
            examples = train_examples(task, name, num_examples=500)
            assert len(examples) == z["tops"].shape[0], (len(examples), z["tops"].shape)
            counts = Counter()
            for i, ex in enumerate(examples[: args.n]):
                own = [Component(int(layer[j]), "neuron", int(index[j])) for j in z["tops"][i]]
                per_layer = Counter(c.layer for c in own)
                rand = [Component(l, "neuron", k) for l, n in per_layer.items()
                        for k in rng.sample(range(model.cfg.d_mlp), n)]
                counts["clean"] += evaluate_accuracy(model, [ex], task=task)[0]
                counts["own top-10%"] += evaluate_accuracy_with_ablation(model, [ex], task=task, removed=own)[0]
                counts["random, same per layer"] += evaluate_accuracy_with_ablation(model, [ex], task=task, removed=rand)[0]
            out[f"{name}|{task}"] = {k: v / args.n for k, v in counts.items()}
            print(name, task, out[f"{name}|{task}"], flush=True)
        del model
    Path(args.output).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
