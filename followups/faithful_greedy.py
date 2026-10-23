"""The circuit-discovery pipeline of Hanna et al. (2024) on our EAP-IG edge scores: for each task,
average the saved edge scores over the training examples, grow a greedy circuit from the logits, and
keep the smallest circuit whose edge-patching faithfulness, (m - b') / (b - b') on the gold answer's
log-probability (or, with --task-metric logit_diff, the logit difference),
reaches 0.85 on the task's 100 cross-task evaluation examples.

Circuit sizes are swept on a geometric grid of edge counts and then refined by bisection between the
largest grid size below the threshold and the smallest above it. As a control, greedy circuits grown
on random edge scores are scored at the same grid sizes. For each circuit the script also
records its nodes, the necessity of those nodes (zero ablation against random sets of the same
composition, on the training examples), and its faithfulness on every other task (Hanna et al.'s
cross-task faithfulness). It writes one JSON per model and a metrics.json per task that
cross_task_experiment.py reads as K=0, P=85.

    python -m followups.faithful_greedy --model meta-llama/Llama-3.2-3B
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from circuit_reuse.circuit_extraction import CircuitExtractor, Component
from circuit_reuse.edge_patching import EdgePatcher, circuit_nodes, greedy_order, layout, prune
from circuit_reuse.evaluate import evaluate_accuracy, evaluate_accuracy_with_ablation
from circuit_reuse.graph import AttentionNode
from followups.common import TASKS, eval_datasets, load_model, train_examples, write_json
from followups.sufficiency import all_units, matched_random

THRESHOLD = 0.85
GRID_POINTS = 16
BISECTIONS = 4
MAX_BATCH = 8
BUFFER_BYTES = 6e9  # per batch of circuits: the (batch, pos, n_forward, d_model) output buffer
N_CONTROLS = 5
RANDOM_SEEDS = 3


def component(node) -> Component:
    return Component(node.layer, "head", node.head) if isinstance(node, AttentionNode) else Component(node.layer, "mlp", 0)


class Task:
    """A task's evaluation inputs, with the full model's metric on the clean and corrupted inputs."""

    def __init__(self, extractor: CircuitExtractor, patcher: EdgePatcher, examples: list):
        self.patcher, self.inputs = patcher, []
        for ex in examples:
            clean, corrupted, clean_mask, corrupted_mask, metric, _ = extractor._prepare_paired_inputs(ex)
            self.inputs.append((clean, corrupted, clean_mask, corrupted_mask, metric))
        with torch.inference_mode():
            model = patcher.model
            self.full = np.mean([float(m(model(c, attention_mask=cm))) for c, _, cm, _, m in self.inputs])
            self.none = np.mean([float(m(model(x, attention_mask=xm))) for _, x, _, xm, m in self.inputs])

    def faithfulness(self, circuits: list[torch.Tensor]) -> list[float]:
        """Faithfulness of each circuit (an edge mask) on this task."""
        totals = np.zeros(len(circuits))
        for clean, corrupted, clean_mask, corrupted_mask, metric in self.inputs:
            corr = self.patcher.corrupted_outputs(corrupted, corrupted_mask)
            batch = int(min(MAX_BATCH, max(1, BUFFER_BYTES // (corr.numel() * corr.element_size()))))
            for start in range(0, len(circuits), batch):
                logits = self.patcher.run(clean, corr, torch.stack(circuits[start:start + batch]), clean_mask)
                totals[start:start + len(logits)] += [float(metric(row[None])) for row in logits]
        return list((totals / len(self.inputs) - self.none) / (self.full - self.none))


def mask_of(order: np.ndarray, n: int, lay: dict, n_bwd: int) -> np.ndarray:
    edges = np.zeros((lay["n_forward"], n_bwd), dtype=bool)
    edges[order[:n, 0], order[:n, 1]] = True
    return prune(edges, lay)


def find_circuit(task: Task, order: np.ndarray, lay: dict, n_bwd: int, device) -> dict:
    """Faithfulness over a geometric grid of edge counts, then bisection to the smallest count at or
    above the threshold. The curve records, for each edge count, the faithfulness and the number of
    heads and MLP blocks the pruned circuit touches."""
    masks = {}

    def to_torch(n):
        masks[n] = mask_of(order, n, lay, n_bwd)
        return torch.from_numpy(masks[n]).to(device)
    grid = sorted({int(n) for n in np.geomspace(30, len(order), GRID_POINTS)} | {len(order)})
    curve = dict(zip(grid, task.faithfulness([to_torch(n) for n in grid])))
    passing = [n for n in grid if curve[n] >= THRESHOLD]
    if passing:
        hi = passing[0]
        lo = max([n for n in grid if n < hi], default=0)
        for _ in range(BISECTIONS):
            if hi - lo <= 1:
                break
            mid = int(round(np.sqrt(max(lo, 1) * hi)))
            mid = min(max(mid, lo + 1), hi - 1)
            curve[mid] = task.faithfulness([to_torch(mid)])[0]
            lo, hi = (lo, mid) if curve[mid] >= THRESHOLD else (mid, hi)
        n_star = hi
    else:
        n_star = len(order)
    nodes = {n: int(m.any(1)[1:].sum()) for n, m in masks.items()}
    return dict(n_edges=n_star, faithfulness=curve[n_star], grid=grid,
                curve=[[n, f, nodes[n]] for n, f in sorted(curve.items())])


def random_curve(task: Task, lay: dict, n_bwd: int, grid: list, device, seed: int = 0) -> list:
    """The control: greedy circuits grown from the logits on random edge scores, so they are connected
    like the Mean Greedy circuit but pick edges at random, at the same edge counts. Each point is the
    mean and standard deviation of faithfulness, and the mean number of nodes, over RANDOM_SEEDS seeds."""
    rng = np.random.default_rng(seed)
    n_valid = int(lay["prev"].sum())
    masks = {(s, n): mask_of(greedy_order(rng.random((lay["n_forward"], n_bwd)), lay, max(grid)), n, lay, n_bwd)
             for s in range(RANDOM_SEEDS) for n in grid if n <= n_valid}
    keys = list(masks)
    faith = dict(zip(keys, task.faithfulness([torch.from_numpy(masks[k]).to(device) for k in keys])))
    return [[n, float(np.mean([faith[s, n] for s in range(RANDOM_SEEDS)])), float(np.std([faith[s, n] for s in range(RANDOM_SEEDS)])),
             float(np.mean([masks[s, n].any(1)[1:].sum() for s in range(RANDOM_SEEDS)]))] for n in grid if (0, n) in masks]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--edges-dir", default="cache_edges/eap_ig_head_mlp")
    parser.add_argument("--output-dir", default="results/edges/faithful")
    parser.add_argument("--metrics-dir", default="results/edges/extraction/edges_faithful")
    parser.add_argument("--num-examples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--task-metric", default="logprob", choices=["logprob", "logit_diff"])
    args = parser.parse_args()
    task_list = args.tasks.split(",")

    model = load_model(args.model)
    extractor = CircuitExtractor(model, method="eap_ig", granularity="head_mlp", task_metric=args.task_metric)
    graph = extractor.graph
    patcher = EdgePatcher(model, graph)
    device = model.cfg.device
    slug = args.model.replace("/", "_")
    lay = layout(graph.index_names())
    n_valid = int(lay["prev"].sum())
    evals = eval_datasets(args.model)
    tasks, circuits, out = {}, {}, {"model": args.model, "threshold": THRESHOLD, "task_metric": args.task_metric,
                                     "n_edges_total": n_valid, "n_nodes_total": len(all_units(model, "head_mlp")), "tasks": {}}

    for task in task_list:
        path = next(Path(args.edges_dir).glob(f"{slug}__none__{task}__eap_ig__*.edges.npy"))
        mean = np.load(path, mmap_mode="r").mean(0, dtype=np.float64)
        order = greedy_order(mean, lay, n_valid)
        tasks[task] = Task(extractor, patcher, evals[task])
        found = find_circuit(tasks[task], order, lay, graph.n_backward, device)
        edges = mask_of(order, found["n_edges"], lay, graph.n_backward)
        found["random_curve"] = random_curve(tasks[task], lay, graph.n_backward, found.pop("grid"), device, seed=args.seed)
        circuits[task] = torch.from_numpy(edges).to(device)
        nodes = [component(n) for n in circuit_nodes(edges, graph)]

        train = train_examples(task, args.model, seed=args.seed, num_examples=args.num_examples)
        base, total = evaluate_accuracy(model, train, task)
        ablated, _ = evaluate_accuracy_with_ablation(model, train, task, nodes)
        rng, units = random.Random(args.seed), all_units(model, "head_mlp")
        controls = [evaluate_accuracy_with_ablation(model, train, task, list(matched_random(units, nodes, rng)))[0]
                    for _ in range(N_CONTROLS)]
        out["tasks"][task] = dict(**found, n_nodes=len(nodes), full_metric=tasks[task].full, none_metric=tasks[task].none,
                                  nodes=[str(c) for c in nodes], edges=np.argwhere(edges).tolist(),
                                  baseline_accuracy=base / total, ablation_accuracy=ablated / total,
                                  control_accuracy=float(np.mean(controls)) / total,
                                  necessity_gap=100 * (float(np.mean(controls)) - ablated) / total)
        print(f"[{task}] {found['n_edges']} edges ({100 * found['n_edges'] / n_valid:.2f}%), {len(nodes)} nodes, "
              f"faithfulness {found['faithfulness']:.3f}, necessity gap {out['tasks'][task]['necessity_gap']:.1f} pp", flush=True)

        write_json(Path(args.metrics_dir) / f"{slug}__main__{task}__faithful" / "metrics.json", {
            "model_name": args.model, "hf_revision": None, "task": task, "method": "eap_ig", "granularity": "head_mlp",
            "num_examples": args.num_examples, "selection": "greedy_faithful", "threshold_faithfulness": THRESHOLD,
            "by_k": {"0": {"thresholds": {"85": {"shared_components": [str(c) for c in nodes],
                                                 "shared_circuit_size": len(nodes)}}}}})

    out["cross_faithfulness"] = {target: dict(zip(task_list, tasks[target].faithfulness([circuits[t] for t in task_list])))
                                 for target in task_list}
    write_json(Path(args.output_dir) / f"{slug}.json", out)


if __name__ == "__main__":
    main()
