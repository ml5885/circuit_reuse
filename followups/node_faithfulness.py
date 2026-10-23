"""Edge-patching faithfulness (Hanna et al., 2024) of node circuits, next to the Mean Greedy circuit of
the same size: for each task and top-K at P=50%, the Top-K and Greedy task circuits (from the
extraction's metrics.json) become edge circuits that keep every edge between their nodes, the input
and the logits, and the Mean Greedy circuit is cut at the first edge count whose pruned circuit touches
as many nodes. All are scored on the task's 100 cross-task evaluation examples with the same metric.

    python -m followups.node_faithfulness --model meta-llama/Llama-3.2-3B --task-metric logit_diff \
        --edges-dir cache_edges_ld/eap_ig_head_mlp --results-dir results/edges_ld
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from circuit_reuse.circuit_extraction import CircuitExtractor
from circuit_reuse.edge_patching import EdgePatcher, greedy_order, layout
from cross_task_experiment import find_metrics_file, load_shared_components
from followups.common import TASKS, eval_datasets, load_model, write_json
from followups.faithful_greedy import Task, mask_of

KS = (1, 5, 10, 20, 30)
P = 50
RULES = {"signed_sum": "Top-K", "greedy": "Greedy"}


def node_circuit_edges(nodes: list, names: dict, lay: dict) -> np.ndarray:
    """Every edge from the input or a circuit node into a circuit node or the logits."""
    index = {n: i for i, n in enumerate(names["forward"])}
    rows = np.zeros(lay["n_forward"], dtype=bool)
    rows[0] = True
    rows[[index[f"a{c.layer}.h{c.index}" if c.kind == "head" else f"m{c.layer}"] for c in nodes]] = True
    cols = np.array([b == len(lay["children"]) - 1 or rows[kids].any() for b, kids in enumerate(lay["children"])])
    valid = np.arange(lay["n_forward"])[:, None] < lay["prev"][None, :]
    return valid & rows[:, None] & cols[None, :]


def matched_mean_greedy(order: np.ndarray, lay: dict, n_bwd: int, n_nodes: int) -> np.ndarray:
    """The Mean Greedy circuit at the first edge count whose pruned circuit touches at least n_nodes."""
    count = lambda n: int(mask_of(order, n, lay, n_bwd).any(1)[1:].sum())
    lo, hi = 1, len(order)
    while lo < hi:
        mid = (lo + hi) // 2
        lo, hi = (mid + 1, hi) if count(mid) < n_nodes else (lo, mid)
    return mask_of(order, lo, lay, n_bwd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--task-metric", default="logit_diff", choices=["logprob", "logit_diff"])
    parser.add_argument("--edges-dir", required=True)
    parser.add_argument("--results-dir", required=True, help="run root with extraction/edges_<rule> folders")
    args = parser.parse_args()

    model = load_model(args.model)
    extractor = CircuitExtractor(model, method="eap_ig", granularity="head_mlp", task_metric=args.task_metric)
    graph = extractor.graph
    names = graph.index_names()
    lay = layout(names)
    patcher = EdgePatcher(model, graph)
    device = model.cfg.device
    slug = args.model.replace("/", "_")
    evals = eval_datasets(args.model)
    out = {"model": args.model, "task_metric": args.task_metric, "P": P, "n_edges_total": int(lay["prev"].sum()),
           "n_nodes_total": lay["n_forward"] - 1, "tasks": {}}

    for task in TASKS:
        mean = np.load(next(Path(args.edges_dir).glob(f"{slug}__none__{task}__eap_ig__*.edges.npy")),
                       mmap_mode="r").mean(0, dtype=np.float64)
        order = greedy_order(mean, lay, int(lay["prev"].sum()))
        entries, masks = [], []
        for rule, label in RULES.items():
            metrics = find_metrics_file(Path(args.results_dir) / "extraction" / f"edges_{rule}", args.model, None, task, P,
                                        method="eap_ig", granularity="head_mlp")
            for k in KS:
                nodes = load_shared_components(metrics, k, P)
                if not nodes:
                    continue
                for kind, mask in ((label, node_circuit_edges(nodes, names, lay)),
                                   (f"Mean Greedy matched to {label}", matched_mean_greedy(order, lay, graph.n_backward, len(nodes)))):
                    entries.append(dict(selection=kind, k=k, n_nodes=int(mask.any(1)[1:].sum()), n_edges=int(mask.sum())))
                    masks.append(torch.from_numpy(mask).to(device))
        faith = Task(extractor, patcher, evals[task]).faithfulness(masks)
        out["tasks"][task] = [dict(e, faithfulness=f) for e, f in zip(entries, faith)]
        print(f"[{task}] " + ", ".join(f"{e['selection']} K={e['k']}: {f:.2f}" for e, f in zip(entries, faith)), flush=True)
    write_json(Path(args.results_dir) / "node_faithfulness" / f"{slug}.json", out)


if __name__ == "__main__":
    main()
