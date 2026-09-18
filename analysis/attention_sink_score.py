"""Attention-sink score for every head in Llama-3.2-3B, across the six tasks.

The role heatmap in `shared_head_attention.py` shows what the three shared heads
attend to, but it cannot say whether that behaviour is unusual, since it only
covers those three heads. Scoring every head on the same prompts gives the
baseline: a head's sink score is the attention it sends from the final query
position to the beginning-of-text token, averaged over prompts.

Run: python -m analysis.attention_sink_score
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "circuit_reuse_mplconfig"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from analysis.shared_head_attention import MODEL_NAME, N_PER_TASK, device, load_examples

TASKS = ["addition", "boolean", "ioi", "mcqa", "arc_easy", "arc_challenge"]
SHARED = [(8, 17), (9, 21), (13, 12)]
OUT = Path("results2/attention_sink")


def main():
    dev = device()
    print(f"device: {dev}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32 if dev == "cpu" else torch.float16,
        attn_implementation="eager",
    ).to(dev)
    model.eval()
    n_layer = model.config.num_hidden_layers
    n_head = model.config.num_attention_heads
    print(f"{n_layer} layers x {n_head} heads = {n_layer * n_head} heads")

    # per task: (n_layer, n_head) mean attention from the final query position to
    # the beginning-of-text token, and to everything that is neither that token
    # nor the final position itself, which is the attention available to do task
    # work with.
    per_task, per_task_content = {}, {}
    for task in TASKS:
        acc = np.zeros((n_layer, n_head))
        acc_content = np.zeros((n_layer, n_head))
        examples = load_examples(task, N_PER_TASK)
        for i, ex in enumerate(examples):
            enc = tokenizer(ex.prompt, return_tensors="pt").to(dev)
            with torch.no_grad():
                out = model(**enc, output_attentions=True, use_cache=False)
            # attentions[L] is (batch, head, query, key); take the final query row.
            for L, a in enumerate(out.attentions):
                row = a[0, :, -1, :].float().cpu().numpy()
                acc[L] += row[:, 0]
                acc_content[L] += row[:, 1:-1].sum(axis=1)
            if i == 0:
                print(f"  [{task}] {enc.input_ids.shape[1]} tokens")
        per_task[task] = acc / len(examples)
        per_task_content[task] = acc_content / len(examples)
        print(f"  [{task}] done, max sink {per_task[task].max():.3f}")

    sink = np.mean([per_task[t] for t in TASKS], axis=0)
    content = np.mean([per_task_content[t] for t in TASKS], axis=0)
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "sink_by_head.npy", sink)
    np.save(OUT / "content_by_head.npy", content)
    json.dump({t: per_task[t].tolist() for t in per_task},
              open(OUT / "sink_by_task.json", "w"))

    flat, cflat = sink.ravel(), content.ravel()
    print(f"\nall heads sink:    mean {flat.mean():.3f}, median {np.median(flat):.3f}, "
          f"90th pct {np.percentile(flat, 90):.3f}, max {flat.max():.3f}")
    print(f"all heads content: mean {cflat.mean():.3f}, median {np.median(cflat):.3f}, "
          f"90th pct {np.percentile(cflat, 90):.3f}, max {cflat.max():.3f}")
    print("\nshared heads:")
    for L, H in SHARED:
        v, cv = sink[L, H], content[L, H]
        print(f"  L{L}H{H}: sink {v:.3f} ({(flat < v).mean() * 100:.1f}th pct), "
              f"content {cv:.3f} ({(cflat < cv).mean() * 100:.1f}th pct)")
    top = np.argsort(flat)[::-1][:10]
    print("\ntop 10 heads by sink score:")
    for r, idx in enumerate(top, 1):
        L, H = divmod(int(idx), n_head)
        mark = "  <-- shared" if (L, H) in SHARED else ""
        print(f"  {r:2d}. L{L}H{H}: {flat[idx]:.3f}{mark}")


if __name__ == "__main__":
    main()
