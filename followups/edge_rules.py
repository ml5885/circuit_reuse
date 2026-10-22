"""Derive per-example component scores under different node-level rules from saved EAP-IG edge scores
(main_experiment.py --save-edges), and write them as attribution caches that main_experiment.py reads
in place of an extraction (same file name, one folder per rule):

    abs_sum     sum of |edge| over the node's outgoing edges (the default extraction)
    signed_sum  minus the signed sum, which equals the direct node attribution (--node-scores)
    greedy      order in which the node enters a greedy circuit grown from the logits, as in the
                EAP-IG library's apply_greedy (score = number of nodes - entry rank)

    python -m followups.edge_rules derive cache_edges/eap_ig_head_mlp --out cache_edges_rules
    python -m followups.edge_rules check EDGES.npy --node-cache NODE.jsonl --old-cache OLD.jsonl

`check` is the pod's gate before the full run: the signed rule must reproduce the --node-scores cache
(same examples), and the abs_sum rule is compared with the earlier bf16-accumulated cache.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from circuit_reuse.circuit_extraction import Component
from main_experiment import _save_attributions_to_cache

RULES = ("abs_sum", "signed_sum", "greedy")


def greedy_order(scores: np.ndarray, children: list[list[int]]) -> np.ndarray:
    """Forward nodes in the order a greedy circuit grown from the logits adds them.

    The circuit starts with the logits column. Each step adds the edge with the largest |score| among
    edges into the circuit; it adds a node when the edge's parent is new, and the parent's input
    columns then become available. Edges to parents already in the circuit add no node, so the next
    node is the one whose largest available edge is the largest."""
    mag = np.abs(scores.astype(np.float64))
    n_fwd, n_bwd = mag.shape
    feeds = [[] for _ in range(n_fwd)]
    for b, kids in enumerate(children):
        for f in kids:
            feeds[f].append(b)
    best = mag[:, n_bwd - 1].copy()
    open_cols = np.zeros(n_bwd, dtype=bool)
    open_cols[n_bwd - 1] = True
    order = []
    for _ in range(n_fwd):
        node = int(np.argmax(best))
        order.append(node)
        best[node] = -np.inf
        new = [b for b in feeds[node] if not open_cols[b]]
        if new:
            open_cols[new] = True
            best = np.maximum(best, np.where(np.isinf(best), -np.inf, mag[:, new].max(1)))
    return np.array(order)


def component(name: str) -> Component | None:
    if name.startswith("a"):
        layer, head = name[1:].split(".h")
        return Component(layer=int(layer), kind="head", index=int(head))
    if name.startswith("m"):
        return Component(layer=int(name[1:]), kind="mlp", index=0)
    return None


def node_scores(edges: np.ndarray, names: dict, rule: str) -> np.ndarray:
    """(n_forward,) node scores of one example's edge matrix under `rule`."""
    e = edges.astype(np.float64)
    if rule == "abs_sum":
        return np.abs(e).sum(1)
    if rule == "signed_sum":
        return -e.sum(1)
    if rule == "greedy":
        out = np.zeros(len(e))
        out[greedy_order(e, names["backward_children"])] = np.arange(len(e), 0, -1)
        return out
    raise ValueError(rule)


def rule_scores(edges: np.ndarray, names: dict, rule: str) -> np.ndarray:
    """(n_examples, n_forward) node scores under `rule`, one example at a time so a memory-mapped
    stack is never loaded whole."""
    return np.stack([node_scores(ex, names, rule) for ex in edges])


def write_caches(edges_path: Path, out: Path, rules=RULES):
    edges = np.load(edges_path, mmap_mode="r")
    names = json.loads(edges_path.with_suffix(".json").read_text())
    comps = [component(n) for n in names["forward"]]
    keep = [i for i, c in enumerate(comps) if c is not None]
    cache_name = edges_path.name.removesuffix(".edges.npy") + ".jsonl"
    for rule in rules:
        scores = rule_scores(edges, names, rule)
        (out / rule).mkdir(parents=True, exist_ok=True)
        _save_attributions_to_cache(out / rule / cache_name, [{comps[i]: float(row[i]) for i in keep} for row in scores])
        print(f"[{rule}] {out / rule / cache_name}")


def cache_matrix(path: Path, comps: list) -> np.ndarray:
    """(n_examples, n_forward) scores from an attribution cache, columns in edge-matrix row order."""
    col = {(c.layer, c.kind, c.index): i for i, c in enumerate(comps) if c is not None}
    rows = [json.loads(line)["components"] for line in path.read_text().splitlines()]
    out = np.zeros((len(rows), len(comps)))
    for i, row in enumerate(rows):
        for c in row:
            out[i, col[c["layer"], c["kind"], c["index"]]] = c["score"]
    return out


def check(edges_path: Path, node_cache: Path, old_cache: Path, min_rho: float = 0.9) -> bool:
    edges = np.load(edges_path, mmap_mode="r")
    names = json.loads(edges_path.with_suffix(".json").read_text())
    comps = [component(n) for n in names["forward"]]
    keep = [i for i, c in enumerate(comps) if c is not None]
    signed = rule_scores(edges, names, "signed_sum")[:, keep]
    node = cache_matrix(node_cache, comps)[:, keep]
    rho = np.array([spearmanr(a, b)[0] for a, b in zip(signed, node)])
    r = np.array([np.corrcoef(a, b)[0, 1] for a, b in zip(signed, node)])
    print(f"signed_sum vs node cache: median Spearman {np.median(rho):.4f} (min {rho.min():.4f}), "
          f"median Pearson {np.median(r):.4f}")
    new, old = rule_scores(edges, names, "abs_sum")[:, keep], cache_matrix(old_cache, comps)[:, keep]
    n = max(1, len(keep) // 10)
    top = lambda m: np.argsort(-m, 1)[:, :n]
    jac = [len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in zip(top(new), top(old))]
    print(f"abs_sum vs old cache: mean top-10% Jaccard {np.mean(jac):.4f}")
    return bool(np.median(rho) >= min_rho)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("derive")
    d.add_argument("edges_dir", type=Path)
    d.add_argument("--out", type=Path, required=True)
    d.add_argument("--rules", default=",".join(RULES))
    d.add_argument("--match", default="*", help="glob on the cache name, e.g. 'qwen3-4b__*'")
    c = sub.add_parser("check")
    c.add_argument("edges", type=Path)
    c.add_argument("--node-cache", type=Path, required=True)
    c.add_argument("--old-cache", type=Path, required=True)
    args = parser.parse_args()
    if args.cmd == "derive":
        for path in sorted(args.edges_dir.glob(f"{args.match}.edges.npy")):
            write_caches(path, args.out, args.rules.split(","))
    elif not check(args.edges, args.node_cache, args.old_cache):
        raise SystemExit("signed edge sums do not reproduce the node scores")


if __name__ == "__main__":
    main()
