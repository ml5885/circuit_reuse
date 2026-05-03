"""
Compute per-task histograms of |attribution score| from the e2 neuron-cache
jsonls (Qwen3-4B, n=200). Designed to run on a babel compute node where
/data/user_data/ml6/circuit_reuse/cache_e2_neuron is mounted.

Writes a small NPZ that the local plotter reads.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TASKS = ["addition", "arc_challenge", "arc_easy", "boolean", "ioi", "mcqa"]
BINS = np.logspace(-6, 2, 80)


def cache_path(cache_dir: Path, task: str) -> Path:
    digit_slug = "d3" if task == "addition" else "dna"
    return cache_dir / f"qwen3-4b__none__{task}__relp__gneuron__n200__{digit_slug}__s42.jsonl"


def histogram_for_task(path: Path) -> np.ndarray:
    counts = np.zeros(len(BINS) - 1, dtype=np.int64)
    if not path.exists():
        return counts
    with path.open() as f:
        for line in f:
            ex = json.loads(line)
            vals = np.fromiter(
                (abs(c["score"]) for c in ex["components"]),
                dtype=np.float32,
                count=len(ex["components"]),
            )
            vals = vals[vals > 0]
            if vals.size == 0:
                continue
            h, _ = np.histogram(vals, bins=BINS)
            counts += h
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    counts_by_task = {}
    for task in TASKS:
        path = cache_path(args.cache_dir, task)
        if not path.exists():
            print(f"[skip] {task}: missing {path}", flush=True)
            continue
        print(f"[load] {task}", flush=True)
        counts = histogram_for_task(path)
        counts_by_task[task] = counts
        print(f"[done] {task} total_nonzero={counts.sum()}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        bins=BINS,
        tasks=list(counts_by_task.keys()),
        **{f"counts_{t}": v for t, v in counts_by_task.items()},
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
