"""Compact per-example attribution caches (neuron caches are 16-25 GB of JSON lines)
into what the follow-up analyses need, so only small files leave the pod.

``summarize CACHE OUT.npz`` makes two passes. The first takes each example's top-K%
by signed score (the paper's selection) and counts how often each unit is chosen,
which gives the task circuit S_P. The second records, per example: total positive
and absolute attribution; the share of each captured by S_P (P = 50, 75, 85, 100)
and by the example's own top-K%; the positive mass captured by its top-n units for
a grid of n; and |S_P ∩ C_i|. It also stores every example's top-K% indices, so
reuse, null models and correct-only reuse can be recomputed on the laptop.

``compare A B OUT.npz`` streams two caches of the same examples (for example two
RelP rule sets) and records per example the Spearman correlation of the scores and
the Jaccard overlap of the two top-K% sets.

    python -m followups.compact_cache summarize cache/x.jsonl results/followup/compact/x.npz
"""
import argparse
import json
from pathlib import Path

import numpy as np

PS = (50, 75, 85, 100)
TOP_N = (100, 300, 1000, 3000, 10000, 30000, 100000)


def rows(path: Path):
    """(keys, scores) per line, keys in the order of the first line."""
    order = None
    with open(path) as f:
        for line in f:
            comps = json.loads(line)["components"]
            if order is None:
                keys = [(c["layer"], c["kind"], c["index"]) for c in comps]
                order = {k: i for i, k in enumerate(keys)}
            scores = np.zeros(len(order), dtype=np.float32)
            for c in comps:
                scores[order[c["layer"], c["kind"], c["index"]]] = c["score"]
            yield keys, scores


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    n = max(1, int(len(scores) * k / 100))
    idx = np.argpartition(-scores, n - 1)[:n]
    return idx[np.argsort(-scores[idx])]


def summarize(cache: Path, out: Path, k: int = 10):
    tops, keys = [], None
    for keys, scores in rows(cache):
        tops.append(top_k(scores, k).astype(np.uint32))
    n_ex, n_units = len(tops), len(keys)
    freq = np.bincount(np.concatenate(tops), minlength=n_units) / n_ex
    shared = {p: freq >= p / 100 for p in PS}

    stats = {f: np.zeros(n_ex, dtype=np.float32) for f in ["pos_total", "abs_total", "own_pos", "own_abs"]}
    for p in PS:
        for f in ("pos", "abs", "overlap"):
            stats[f"s{p}_{f}"] = np.zeros(n_ex, dtype=np.float32)
    top_n_pos = np.zeros((n_ex, len(TOP_N)), dtype=np.float32)
    for i, (_, scores) in enumerate(rows(cache)):
        pos, mag = np.clip(scores, 0, None), np.abs(scores)
        stats["pos_total"][i], stats["abs_total"][i] = pos.sum(), mag.sum()
        own = np.zeros(n_units, dtype=bool)
        own[tops[i]] = True
        stats["own_pos"][i], stats["own_abs"][i] = pos[own].sum(), mag[own].sum()
        for p in PS:
            stats[f"s{p}_pos"][i], stats[f"s{p}_abs"][i] = pos[shared[p]].sum(), mag[shared[p]].sum()
            stats[f"s{p}_overlap"][i] = (own & shared[p]).sum()
        ranked = np.sort(pos)[::-1].cumsum()
        top_n_pos[i] = [ranked[min(n, n_units) - 1] for n in TOP_N]

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, layer=np.array([k[0] for k in keys], dtype=np.int16),
                        kind=np.array([k[1] for k in keys]), index=np.array([k[2] for k in keys], dtype=np.int32),
                        k=k, tops=np.stack(tops), freq=freq.astype(np.float32),
                        shared_size=np.array([shared[p].sum() for p in PS]), ps=np.array(PS),
                        top_n=np.array(TOP_N), top_n_pos=top_n_pos, **stats)
    print(f"{cache.name}: {n_ex} examples, {n_units} units, |S_50|={shared[50].sum()}")


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def compare(path_a: Path, path_b: Path, out: Path, k: int = 10):
    rho, jac, perm = [], [], None
    for (keys_a, a), (keys_b, b) in zip(rows(path_a), rows(path_b)):
        if perm is None:
            where = {key: i for i, key in enumerate(keys_b)}
            perm = np.array([where[key] for key in keys_a])
        b = b[perm]
        rho.append(spearman(a, b))
        ta, tb = set(top_k(a, k).tolist()), set(top_k(b, k).tolist())
        jac.append(len(ta & tb) / len(ta | tb))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, spearman=np.array(rho), topk_jaccard=np.array(jac))
    print(f"{out.name}: {len(rho)} examples, median Spearman {np.median(rho):.3f}, "
          f"median top-{k}% Jaccard {np.median(jac):.3f}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summarize")
    s.add_argument("cache", type=Path)
    s.add_argument("out", type=Path)
    c = sub.add_parser("compare")
    c.add_argument("a", type=Path)
    c.add_argument("b", type=Path)
    c.add_argument("out", type=Path)
    args = parser.parse_args()
    if args.cmd == "summarize":
        summarize(args.cache, args.out)
    else:
        compare(args.a, args.b, args.out)


if __name__ == "__main__":
    main()
