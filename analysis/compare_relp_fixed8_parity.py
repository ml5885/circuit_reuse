#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from scipy.stats import pearsonr


REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_TL_ROOT = REPO_ROOT / "reference_code" / "RelP" / "TransformerLens"

PROMPTS = [
    "When John and Mary went to the shops, John gave the bag to",
    "When John and Mary went to the shops, Mary gave the bag to",
    "When Tom and James went to the park, James gave the ball to",
    "When Tom and James went to the park, Tom gave the ball to",
    "When Dan and Sid went to the shops, Sid gave an apple to",
    "When Dan and Sid went to the shops, Dan gave an apple to",
    "After Martin and Amy went to the park, Amy gave a drink to",
    "After Martin and Amy went to the park, Martin gave a drink to",
]
ANSWERS = [
    (" Mary", " John"),
    (" John", " Mary"),
    (" Tom", " James"),
    (" James", " Tom"),
    (" Dan", " Sid"),
    (" Sid", " Dan"),
    (" Martin", " Amy"),
    (" Amy", " Martin"),
]
PATCH_TYPES = ("resid_pre", "attn_out", "mlp_out")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare fixed-8 RelP/AtP parity against the reference fork.")
    parser.add_argument("--backend", choices=["ours", "reference"], required=True)
    parser.add_argument("--mode", choices=["atp", "relp"], required=True)
    parser.add_argument("--model-name", default="gpt2-small")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float32", "bf16", "float16"], default="float32")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _import_backend(backend: str):
    if backend == "reference":
        sys.path.insert(0, str(REFERENCE_TL_ROOT))
    from transformer_lens import ActivationCache, HookedTransformer, patching, utils

    return ActivationCache, HookedTransformer, patching, utils


def _dtype_from_arg(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
    }[name]


def _answer_token_indices(model) -> torch.Tensor:
    return torch.tensor(
        [[model.to_single_token(ANSWERS[i][j]) for j in range(2)] for i in range(len(ANSWERS))],
        device=model.cfg.device,
    )


def _get_logit_diff(logits: torch.Tensor, answer_token_indices: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 3:
        logits = logits[:, -1, :]
    correct_logits = logits.gather(1, answer_token_indices[:, 0].unsqueeze(1))
    incorrect_logits = logits.gather(1, answer_token_indices[:, 1].unsqueeze(1))
    return (correct_logits - incorrect_logits).mean()


def _get_cache_fwd_and_bwd(model, tokens, metric, ActivationCache):
    model.reset_hooks()
    cache = {}

    def filter_not_qkv_input(name: str) -> bool:
        return "_input" not in name

    def forward_cache_hook(act, hook):
        cache[hook.name] = act.detach()

    grad_cache = {}

    def backward_cache_hook(act, hook):
        grad_cache[hook.name] = act.detach()

    model.add_hook(filter_not_qkv_input, forward_cache_hook, "fwd")
    model.add_hook(filter_not_qkv_input, backward_cache_hook, "bwd")

    value = metric(model(tokens))
    value.backward()
    model.reset_hooks()
    return value.item(), ActivationCache(cache, model), ActivationCache(grad_cache, model)


def _get_attr_patch_block_every(attr_cache) -> torch.Tensor:
    import einops

    resid_pre_attr = einops.reduce(
        attr_cache.stack_activation("resid_pre"),
        "layer batch pos d_model -> layer pos",
        "sum",
    )
    attn_out_attr = einops.reduce(
        attr_cache.stack_activation("attn_out"),
        "layer batch pos d_model -> layer pos",
        "sum",
    )
    mlp_out_attr = einops.reduce(
        attr_cache.stack_activation("mlp_out"),
        "layer batch pos d_model -> layer pos",
        "sum",
    )
    return torch.stack([resid_pre_attr, attn_out_attr, mlp_out_attr], dim=0)


def main() -> None:
    args = parse_args()
    ActivationCache, HookedTransformer, patching, _utils = _import_backend(args.backend)
    sys.path.insert(0, str(REPO_ROOT))
    from circuit_reuse.lrp_patch import disable_lrp, enable_lrp

    model = HookedTransformer.from_pretrained(
        args.model_name,
        device=args.device,
        dtype=_dtype_from_arg(args.dtype),
    )
    model.set_use_attn_result(True)
    model.set_use_hook_mlp_in(True)
    model.set_use_attn_in(True)
    model.set_use_split_qkv_input(True)

    if args.mode == "relp":
        if args.backend == "ours":
            enable_lrp(model, rules=["LN-rule", "Identity-rule", "Half-rule"])
        else:
            model.cfg.use_lrp = True
            model.cfg.LRP_rules = ["LN-rule", "Identity-rule", "Half-rule"]
    else:
        disable_lrp(model)
        model.cfg.use_lrp = False

    clean_tokens = model.to_tokens(PROMPTS)
    corrupted_tokens = clean_tokens[[(i + 1 if i % 2 == 0 else i - 1) for i in range(len(clean_tokens))]]
    answer_token_indices = _answer_token_indices(model)

    clean_logits, clean_cache = model.run_with_cache(clean_tokens)
    corrupted_logits, _corrupted_cache = model.run_with_cache(corrupted_tokens)
    clean_logit_diff = _get_logit_diff(clean_logits, answer_token_indices).item()
    corrupted_logit_diff = _get_logit_diff(corrupted_logits, answer_token_indices).item()

    def ioi_metric(logits):
        return (_get_logit_diff(logits, answer_token_indices) - corrupted_logit_diff) / (
            clean_logit_diff - corrupted_logit_diff
        )

    _, clean_cache_bwd, _clean_grad_cache = _get_cache_fwd_and_bwd(
        model, clean_tokens, ioi_metric, ActivationCache
    )
    _, corrupted_cache_bwd, corrupted_grad_cache = _get_cache_fwd_and_bwd(
        model, corrupted_tokens, ioi_metric, ActivationCache
    )

    attribution_cache_dict = {}
    for key in corrupted_grad_cache.cache_dict.keys():
        attribution_cache_dict[key] = corrupted_grad_cache.cache_dict[key] * (
            clean_cache_bwd.cache_dict[key] - corrupted_cache_bwd.cache_dict[key]
        )
    attr_cache = ActivationCache(attribution_cache_dict, model)
    attr_map = _get_attr_patch_block_every(attr_cache).detach().cpu()

    ap_map = patching.get_act_patch_block_every(model, corrupted_tokens, clean_cache_bwd, ioi_metric).detach().cpu()

    results = {
        "backend": args.backend,
        "mode": args.mode,
        "clean_logit_diff": clean_logit_diff,
        "corrupted_logit_diff": corrupted_logit_diff,
        "pcc": {
            patch_type: float(pearsonr(ap_map[i].reshape(-1).numpy(), attr_map[i].reshape(-1).numpy()).statistic)
            for i, patch_type in enumerate(PATCH_TYPES)
        },
        "ap_map_shape": list(ap_map.shape),
        "attr_map_shape": list(attr_map.shape),
    }

    if args.output is not None:
        args.output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
