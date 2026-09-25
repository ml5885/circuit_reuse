"""Why neuron-level IOI circuits are not necessary: IOI accuracy (100 evaluation examples) with
the full model, with the component circuit ablated, with only its MLP blocks ablated, with every
neuron of those MLP layers ablated, with the neuron circuit ablated at K=10% and K=30%, and with
as many random neurons per layer as the K=30% circuit. All ablations are zero ablations at P=50%,
circuits from EAP-IG. If every neuron of the circuit's MLP layers reproduces the MLP-block drop,
neuron ablation reaches IOI and the null result belongs to the neuron circuits.

    python -m followups.ioi_neuron_check --results-dir results/rerun/extraction
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

from circuit_reuse.circuit_extraction import Component
from circuit_reuse.evaluate import evaluate_accuracy, evaluate_accuracy_with_ablation
from cross_task_experiment import find_metrics_file, load_shared_components
from followups.common import MODELS, eval_datasets, load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--output", default="results/rerun/ioi_neuron_check.json")
    args = parser.parse_args()
    root = Path(args.results_dir)
    out = {}
    for name in args.models.split(","):
        model = load_model(name)
        ds = eval_datasets(name)["ioi"]

        def circuit(gran, K):
            f = find_metrics_file(root / f"granularity_parity_eap_ig_{gran}", name, None, "ioi", 50,
                                  method="eap_ig", granularity=gran)
            return load_shared_components(f, K, 50)

        comp = circuit("head_mlp", 10)
        mlp_layers = sorted({c.layer for c in comp if c.kind == "mlp"})
        neurons30 = circuit("neuron", 30)
        rng = random.Random(0)
        per_layer = Counter(c.layer for c in neurons30)
        rand30 = [Component(l, "neuron", i) for l, n in per_layer.items()
                  for i in rng.sample(range(model.cfg.d_mlp), n)]
        conditions = {
            "component circuit": comp,
            "its MLP blocks": [c for c in comp if c.kind == "mlp"],
            "every neuron of those MLP layers": [Component(l, "neuron", i) for l in mlp_layers for i in range(model.cfg.d_mlp)],
            "neuron circuit, K=10%": circuit("neuron", 10),
            "neuron circuit, K=30%": neurons30,
            "random neurons, K=30% layer counts": rand30,
        }
        c, n = evaluate_accuracy(model, ds, task="ioi")
        res = {"full model": {"size": 0, "accuracy": c / n}, "mlp_layers": mlp_layers}
        for label, removed in conditions.items():
            c, n = evaluate_accuracy_with_ablation(model, ds, task="ioi", removed=removed)
            res[label] = {"size": len(removed), "accuracy": c / n}
        out[name] = res
        print(name, {k: (v["size"], round(v["accuracy"], 2)) for k, v in res.items() if k != "mlp_layers"}, flush=True)
        del model
    Path(args.output).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
