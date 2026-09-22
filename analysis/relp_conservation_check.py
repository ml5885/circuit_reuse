#!/usr/bin/env python3
"""Numerical check of relevance conservation under each RelP rule set.

Relevance at a hook is the gradient-times-activation sum R(h) = <h, dL/dh>,
with the backward pass run exactly as in RelP extraction (corrupted input,
log-prob metric, rule set installed through CircuitExtractor). For every
layer we compare the relevance entering and leaving each linearised
operation:

  norm       R(ln2 input) / R(ln2 normalized)      LN-rule       -> 1
  attention  [R(q)+R(k)+R(v)] / R(z)               AH-rule       -> 1
  gate       [R(sigma(pre))+R(pre_linear)] / R(post)  Half-rule  -> 1 (2 without it)
  act_fn     R(pre) / R(sigma(pre))                Identity-rule -> 1

Under the default rule set every ratio is 1; "legacy" (no Identity-rule)
fails act_fn and "jafari" (no AH-rule) fails attention.

where R(sigma(pre)) = R(pre_linear) exactly in a gated MLP (both factors of the
elementwise product receive <sigma(u) * v, g>), and R(post) in a plain MLP.

Ratios are reported per rule set as the aggregate over layers and examples
(sum of numerators over sum of denominators) and as the median and 5-95%
range of the per-layer ratios.

    python analysis/relp_conservation_check.py --model meta-llama/Llama-3.2-3B \
        --task ioi --num-examples 8 --out results2/relp_conservation
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from circuit_reuse.circuit_extraction import CircuitExtractor
from circuit_reuse.dataset import apply_few_shot_prefix, get_dataset
from circuit_reuse.lrp_patch import DEFAULT_LRP_RULES, LEGACY_LRP_RULES, PUBLISHED_LRP_RULES, lrp_rules_tag
from models.olmo_adapter import load_model_any

RULE_SETS = {
    "ours": DEFAULT_LRP_RULES,                 # Arora et al. 2026: LN, Identity, AH, Half
    "legacy": LEGACY_LRP_RULES,                # our pre-2026-09-21 set: no Identity-rule
    "jafari": PUBLISHED_LRP_RULES["jafari"],   # RelP repository default: no AH-rule
    "ours-Half": ["LN-rule", "Identity-rule", "AH-rule"],
    "ours-LN": ["Identity-rule", "AH-rule", "Half-rule"],
    "none": [],
}

CHECKS = {
    "norm": (["hook_mlp_in"], ["ln2.hook_normalized"]),
    "attention": (["attn.hook_q", "attn.hook_k", "attn.hook_v"], ["attn.hook_z"]),
    # R(sigma(pre)) equals R(pre_linear) in a gated MLP (the product is
    # symmetric in its two factors), so the gate sum is twice R(pre_linear).
    "gate": (["mlp.hook_pre_linear", "mlp.hook_pre_linear"], ["mlp.hook_post"]),
    "act_fn": (["mlp.hook_pre"], ["mlp.hook_pre_linear"]),
    "act_fn_plain": (["mlp.hook_pre"], ["mlp.hook_post"]),
}
HOOKS = sorted({h for ins, outs in CHECKS.values() for h in ins + outs})


def relevances(model, extractor: CircuitExtractor, example) -> dict[str, float]:
    """Per-hook relevance sums <h, dL/dh> for one example's RelP backward pass."""
    (_, corrupted_tokens, _, corrupted_mask, metric, _) = extractor._prepare_paired_inputs(example)
    names = [f"blocks.{l}.{h}" for l in range(model.cfg.n_layers) for h in HOOKS
             if f"blocks.{l}.{h}" in model.hook_dict]
    saved: dict[str, torch.Tensor] = {}

    def keep(act, hook):
        act.retain_grad()
        saved[hook.name] = act

    with model.hooks(fwd_hooks=[(n, keep) for n in names]):
        logits = model(corrupted_tokens, attention_mask=corrupted_mask)
        metric(logits).backward()
    model.zero_grad(set_to_none=True)
    model.reset_hooks()
    # A hook whose gradient is None received no relevance (q and k under the AH-rule).
    return {name: 0.0 if act.grad is None else float((act.detach().double() * act.grad.double()).sum())
            for name, act in saved.items()}


def layer_ratios(rel: dict[str, float], n_layers: int, gated: bool) -> list[dict]:
    rows = []
    for l in range(n_layers):
        for check, (ins, outs) in CHECKS.items():
            if check == ("act_fn_plain" if gated else "act_fn"):
                continue
            hooks = [f"blocks.{l}.{h}" for h in ins + outs]
            if any(h not in rel for h in hooks):
                continue
            rows.append({"layer": l, "check": check.removesuffix("_plain"),
                         "num": sum(rel[f"blocks.{l}.{h}"] for h in ins),
                         "den": sum(rel[f"blocks.{l}.{h}"] for h in outs)})
    return rows


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (rules, check), g in df.groupby(["rules", "check"], sort=False):
        ratio = g["num"] / g["den"]
        rows.append({
            "rules": rules, "check": check,
            "aggregate": g["num"].sum() / g["den"].sum(),
            "median": ratio.median(),
            "q05": ratio.quantile(.05), "q95": ratio.quantile(.95),
            "n": len(g),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", default="ioi")
    ap.add_argument("--num-examples", type=int, default=8)
    ap.add_argument("--digits", type=int, default=3)
    ap.add_argument("--rule-sets", default=",".join(RULE_SETS))
    ap.add_argument("--dtype", default="float32", choices=["float32", "bf16"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results2/relp_conservation")
    args = ap.parse_args()

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    model = load_model_any(args.model, device=args.device, torch_dtype=dtype)
    model.cfg.dtype = dtype
    random.seed(args.seed)
    dataset = apply_few_shot_prefix(get_dataset(args.task, num_examples=args.num_examples, digits=args.digits),
                                    args.task, args.model)
    examples = list(dataset)[: args.num_examples]

    gated = "blocks.0.mlp.hook_pre_linear" in model.hook_dict
    rows = []
    for name in args.rule_sets.split(","):
        rules = RULE_SETS[name]
        extractor = CircuitExtractor(model, method="relp", granularity="head_mlp", use_lrp=True, lrp_rules=rules)
        for i, ex in enumerate(examples):
            rel = relevances(model, extractor, ex)
            for row in layer_ratios(rel, model.cfg.n_layers, gated):
                rows.append({"rules": name, "rule_tag": lrp_rules_tag(rules), "example": i, **row})
        print(f"[{name}] done ({lrp_rules_tag(rules) or 'no rules'})", flush=True)

    df = pd.DataFrame(rows)
    summary = summarize(df)
    out = Path(args.out) / args.model.replace("/", "_")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_layer.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    (out / "config.json").write_text(json.dumps({
        "model": args.model, "task": args.task, "num_examples": len(examples),
        "dtype": args.dtype, "rule_sets": {k: RULE_SETS[k] for k in args.rule_sets.split(",")},
        "act_fn": model.cfg.act_fn, "gated_mlp": gated, "n_layers": model.cfg.n_layers,
    }, indent=2))
    pd.set_option("display.width", 200)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
