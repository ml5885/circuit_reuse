"""Edge-patching necessity of the Mean Greedy circuit as it grows, the counterpart of the sufficiency
curve in followups/faithful_greedy.py: at each edge count, patch only the circuit's edges to their
values on each example's corrupted input, keep every other edge clean, and report the share of the
model's performance lost, 1 - (m - b') / (b - b'). Greedy circuits grown on random edge scores, at the
same edge counts, are the control.

    python -m followups.edge_necessity --model meta-llama/Llama-3.2-3B --task-metric logit_diff \
        --edges-dir cache_edges_ld/eap_ig_head_mlp --output-dir results/edges_ld/necessity
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from circuit_reuse.circuit_extraction import CircuitExtractor
from circuit_reuse.edge_patching import EdgePatcher, greedy_order, layout
from followups.common import TASKS, eval_datasets, load_model, write_json
from followups.faithful_greedy import Task, mask_of

GRID_POINTS = 10
RANDOM_SEEDS = 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--task-metric", default="logit_diff", choices=["logprob", "logit_diff"])
    parser.add_argument("--edges-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-examples", type=int, default=100, help="evaluation examples per task")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model = load_model(args.model)
    extractor = CircuitExtractor(model, method="eap_ig", granularity="head_mlp", task_metric=args.task_metric)
    graph = extractor.graph
    lay = layout(graph.index_names())
    n_bwd, n_valid = graph.n_backward, int(lay["prev"].sum())
    valid = np.arange(lay["n_forward"])[:, None] < lay["prev"][None, :]
    patcher = EdgePatcher(model, graph)
    device = model.cfg.device
    slug = args.model.replace("/", "_")
    evals = eval_datasets(args.model)
    grid = sorted({int(n) for n in np.geomspace(30, n_valid, GRID_POINTS)})
    rng = np.random.default_rng(args.seed)
    out = {"model": args.model, "task_metric": args.task_metric, "n_edges_total": n_valid,
           "n_nodes_total": lay["n_forward"] - 1, "num_examples": args.num_examples, "tasks": {}}

    for task in TASKS:
        mean = np.load(next(Path(args.edges_dir).glob(f"{slug}__none__{task}__eap_ig__*.edges.npy")),
                       mmap_mode="r").mean(0, dtype=np.float64)
        orders = {"mean_greedy": greedy_order(mean, lay, n_valid)}
        orders.update({f"random_{s}": greedy_order(rng.random(mean.shape), lay, n_valid) for s in range(RANDOM_SEEDS)})
        circuits = {(name, n): mask_of(order, n, lay, n_bwd) for name, order in orders.items() for n in grid}
        keys = list(circuits)
        # keep every edge except the circuit's, so only the circuit is patched
        kept = Task(extractor, patcher, evals[task][:args.num_examples]).faithfulness(
            [torch.from_numpy(valid & ~circuits[k]).to(device) for k in keys])
        necessity = {k: 1 - s for k, s in zip(keys, kept)}
        nodes = {k: int(m.any(1)[1:].sum()) for k, m in circuits.items()}
        random = [f"random_{s}" for s in range(RANDOM_SEEDS)]
        out["tasks"][task] = {
            "curve": [[n, necessity["mean_greedy", n], nodes["mean_greedy", n]] for n in grid],
            "random_curve": [[n, float(np.mean([necessity[r, n] for r in random])), float(np.std([necessity[r, n] for r in random])),
                              float(np.mean([nodes[r, n] for r in random]))] for n in grid],
        }
        print(f"[{task}] " + ", ".join(f"{n}: {necessity['mean_greedy', n]:.2f}" for n in grid), flush=True)
    write_json(Path(args.output_dir) / f"{slug}.json", out)


if __name__ == "__main__":
    main()
