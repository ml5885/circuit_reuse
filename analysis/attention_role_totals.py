"""Total attention by template role, for every head in Llama-3.2-3B.

`shared_head_attention.py` reports the *mean* attention per token of a role, which
is small for any role that spans many tokens and cannot be compared across roles
or summed. This computes the total attention a head sends from the final query
position to all tokens of a role, which is the quantity the case study needs, and
does it for every head so the three shared heads have a baseline.

Run: python -m analysis.attention_role_totals
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "circuit_reuse_mplconfig"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from analysis.shared_head_attention import (MODEL_NAME, N_PER_TASK, classify_token,
                                            device, load_examples)

TASKS = ["addition", "boolean", "ioi", "mcqa", "arc_easy", "arc_challenge"]
SHARED = [(8, 17), (9, 21), (13, 12)]
# The role a head would have to read to do the task, as opposed to prompt
# scaffolding. IOI needs the names, Addition the digits, the MCQA-style tasks the
# answer labels, Boolean the literals.
CRITICAL = {"addition": "DIGIT", "boolean": "LITERAL", "ioi": "NAME",
            "mcqa": "LABEL", "arc_easy": "LABEL", "arc_challenge": "LABEL"}
OUT = Path("results2/attention_sink")


def main():
    dev = device()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32 if dev == "cpu" else torch.float16,
        attn_implementation="eager",
    ).to(dev)
    model.eval()
    n_layer = model.config.num_hidden_layers
    n_head = model.config.num_attention_heads
    print(f"device {dev}, {n_layer}x{n_head} heads")

    totals = {}  # task -> role -> (n_layer, n_head) summed attention
    for task in TASKS:
        acc = defaultdict(lambda: np.zeros((n_layer, n_head)))
        examples = load_examples(task, N_PER_TASK)
        for ex in examples:
            enc = tokenizer(ex.prompt, return_tensors="pt").to(dev)
            tokens = tokenizer.convert_ids_to_tokens(enc.input_ids[0])
            n_tok = len(tokens)
            lower = ex.prompt.lower()
            roles = [classify_token(task, t, p, n_tok, lower, tokens)
                     for p, t in enumerate(tokens)]
            idx = defaultdict(list)
            for p, r in enumerate(roles):
                idx[r].append(p)
            with torch.no_grad():
                out = model(**enc, output_attentions=True, use_cache=False)
            rows = np.stack([a[0, :, -1, :].float().cpu().numpy()
                             for a in out.attentions])  # (layer, head, key)
            for r, positions in idx.items():
                acc[r] += rows[:, :, positions].sum(axis=2)
        totals[task] = {r: v / len(examples) for r, v in acc.items()}
        print(f"  [{task}] roles: {sorted(totals[task])}")

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({t: {r: v.tolist() for r, v in d.items()} for t, d in totals.items()},
              open(OUT / "role_totals.json", "w"))

    # Body = everything that is not the sink, the final token, or the two tokens
    # before it, i.e. the part of the prompt that carries the task.
    struct = {"BOS", "FINAL", "RECENT"}
    print("\n=== total attention to the prompt body (not BOS/FINAL/RECENT) ===")
    body = np.mean([sum(v for r, v in totals[t].items() if r not in struct)
                    for t in TASKS], axis=0)
    bf = body.ravel()
    print(f"all heads: median {np.median(bf):.3f}, 90th pct {np.percentile(bf, 90):.3f}")
    for L, H in SHARED:
        print(f"  L{L}H{H}: {body[L, H]:.3f} ({(bf < body[L, H]).mean() * 100:.1f}th pct)")

    print("\n=== total attention to the task-critical role, per task ===")
    for task in TASKS:
        role = CRITICAL[task]
        if role not in totals[task]:
            print(f"  [{task}] role {role} absent")
            continue
        m = totals[task][role]
        mf = m.ravel()
        line = ", ".join(f"L{L}H{H} {m[L, H]:.3f} ({(mf < m[L, H]).mean() * 100:.0f}th)"
                         for L, H in SHARED)
        print(f"  [{task}] {role}: all-head median {np.median(mf):.3f} | {line}")


if __name__ == "__main__":
    main()
