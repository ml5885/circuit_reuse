"""
Compute Spearman-based circuit-stability metrics on the e2 neuron-granularity
attribution cache (Qwen3-4B only, n=200). Designed to run on a compute node
with /data/user_data/ml6 mounted. Writes a small NPZ; the heatmap is rendered
locally afterwards.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

TASKS = ["addition", "arc_challenge", "arc_easy", "boolean", "ioi", "mcqa"]
K_PCTS = [1, 5, 10, 20, 30]
N_PAIRS = 2000
SEED = 42


def cache_path(cache_dir: Path, task: str) -> Path:
    digit_slug = "d3" if task == "addition" else "dna"
    return cache_dir / f"qwen3-4b__none__{task}__relp__gneuron__n200__{digit_slug}__s42.jsonl"


def load_score_matrix(path: Path) -> np.ndarray:
    examples = []
    comp_idx: dict[tuple, int] = {}
    with path.open() as f:
        for line in f:
            ex = json.loads(line)
            examples.append(ex["components"])
            for c in ex["components"]:
                key = (c["kind"], c["layer"], c["index"])
                if key not in comp_idx:
                    comp_idx[key] = len(comp_idx)
    N, D = len(examples), len(comp_idx)
    M = np.zeros((N, D), dtype=np.float32)
    for i, comps in enumerate(examples):
        for c in comps:
            M[i, comp_idx[(c["kind"], c["layer"], c["index"])]] = abs(float(c["score"]))
    return M


def topk_idx(row: np.ndarray, K_abs: int) -> np.ndarray:
    if K_abs >= row.shape[0]:
        return np.arange(row.shape[0])
    return np.argpartition(-row, K_abs)[:K_abs]


def spearman_on_union(x: np.ndarray, y: np.ndarray) -> float:
    if x.shape[0] < 2 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    rx, ry = rankdata(x), rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def pairwise_spearman(scores: np.ndarray, K_abs: int, n_pairs: int, rng: random.Random) -> float:
    N = scores.shape[0]
    top_per_ex = [topk_idx(scores[i], K_abs) for i in range(N)]
    vals = []
    for _ in range(n_pairs):
        i, j = rng.sample(range(N), 2)
        union = np.unique(np.concatenate([top_per_ex[i], top_per_ex[j]]))
        rho = spearman_on_union(scores[i, union], scores[j, union])
        if not np.isnan(rho):
            vals.append(rho)
    return float(np.mean(vals)) if vals else float("nan")


def reference_spearman(scores: np.ndarray, K_abs: int) -> float:
    aggregate = scores.mean(axis=0)
    agg_top = topk_idx(aggregate, K_abs)
    vals = []
    for i in range(scores.shape[0]):
        ex_top = topk_idx(scores[i], K_abs)
        union = np.unique(np.concatenate([agg_top, ex_top]))
        rho = spearman_on_union(scores[i, union], aggregate[union])
        if not np.isnan(rho):
            vals.append(rho)
    return float(np.mean(vals)) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path,
                    help="Output NPZ file path.")
    args = ap.parse_args()

    rng = random.Random(SEED)
    pairwise = np.full((len(TASKS), len(K_PCTS)), np.nan, dtype=np.float32)
    reference = np.full_like(pairwise, np.nan)

    for ti, task in enumerate(TASKS):
        path = cache_path(args.cache_dir, task)
        if not path.exists():
            print(f"[skip] {task}: cache missing at {path}")
            continue
        print(f"[load] {task} from {path}", flush=True)
        M = load_score_matrix(path)
        N, D = M.shape
        for ki, K_pct in enumerate(K_PCTS):
            K_abs = max(1, int(round(D * K_pct / 100)))
            pairwise[ti, ki] = pairwise_spearman(M, K_abs, N_PAIRS, rng)
            reference[ti, ki] = reference_spearman(M, K_abs)
        del M
        print(f"[done] {task} N={N} D={D}  "
              f"pair@1%={pairwise[ti,0]:.3f} pair@30%={pairwise[ti,-1]:.3f}  "
              f"ref@1%={reference[ti,0]:.3f} ref@30%={reference[ti,-1]:.3f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, pairwise=pairwise, reference=reference,
             tasks=TASKS, K_pcts=K_PCTS, model="Qwen3-4B")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
