from __future__ import annotations

from typing import Iterable, List, Tuple, Any
import torch
from .dataset import ArithmeticExample  # type: ignore
from .circuit_extraction import Component  # type: ignore


def _extract_gold_ids(
    model: Any,
    prompt: str,
    target: str,
    device,
    verbose: bool = False,
) -> List[int]:
    """
    Derive gold target token ids robustly even when BPE merges across the
    prompt/target boundary.

    We tokenize:
      A = tokens(prompt)
      B = tokens(prompt + target)
    Then take the longest common prefix length between A and B; the remainder
    of B are the gold target ids. This avoids failures where simple slicing by
    len(A) yields an empty remainder because the final prompt token in A merges
    with the first answer characters in B.
    """
    prompt_tok = model.to_tokens(prompt, prepend_bos=True)
    full_tok = model.to_tokens(prompt + target, prepend_bos=True)
    prompt_tok = prompt_tok.to(device)
    full_tok = full_tok.to(device)
    p_ids = prompt_tok[0].tolist()
    f_ids = full_tok[0].tolist()
    lcp = 0

    for a, b in zip(p_ids, f_ids):
        if a == b:
            lcp += 1
        else:
            break

    if lcp == len(f_ids):
        # Entire full sequence identical -> fall back to standalone target
        # tokenization
        alt = model.to_tokens(target, prepend_bos=False)
        alt = alt.to(device)
        fallback_ids = alt[0].tolist() if alt.ndim == 2 else alt.tolist()
        if verbose:
            print(
                "[WARN] No divergent token boundary; fallback standalone "
                f"target tokenization target='{target}' "
                f"ids={fallback_ids}"
            )
        return [int(x) for x in fallback_ids]

    gold_ids = f_ids[lcp:]

    if verbose and not gold_ids:
        print(
            "[WARN] Derived empty gold ids unexpectedly; "
            f"prompt_len={len(p_ids)} full_len={len(f_ids)}"
        )

    return [int(x) for x in gold_ids]


# Simple caches to avoid repeated tokenization per example
_BOOL_CACHE = {}
_MC_CACHE = {}


def _boolean_token_id_groups(model) -> Tuple[set, set]:
    """
    Collect single-token ids for (true,false) across common spacing/casing
    variants.
    """
    cache_key = id(model)
    if cache_key in _BOOL_CACHE:
        return _BOOL_CACHE[cache_key]

    variants_true = [" true", "true", " True", "True"]
    variants_false = [" false", "false", " False", "False"]

    def collect(variants):
        out = set()
        for v in variants:
            toks = model.to_tokens(v, prepend_bos=False)
            ids = toks[0].tolist()
            if len(ids) == 1:
                out.add(int(ids[0]))
        return out

    res = (collect(variants_true), collect(variants_false))
    _BOOL_CACHE[cache_key] = res
    return res


def _classify_boolean(
    logits_last: Any,
    model,
    verbose: bool = False,
) -> Tuple[str, dict]:
    true_ids, false_ids = _boolean_token_id_groups(model)
    id_logits = {
        f"true:{tid}": float(logits_last[tid].item())
        for tid in true_ids
    }
    id_logits.update({
        f"false:{fid}": float(logits_last[fid].item())
        for fid in false_ids
    })
    true_score = max(
        (logits_last[tid].item() for tid in true_ids),
        default=float("-inf"),
    )
    false_score = max(
        (logits_last[fid].item() for fid in false_ids),
        default=float("-inf"),
    )
    label = "true" if true_score >= false_score else "false"
    if verbose:
        print(
            f"[BOOL-CLASSIFY] true_score={true_score:.3f} "
            f"false_score={false_score:.3f} label={label}"
        )
    return label, id_logits


def _mc_letter_token_id_groups(model) -> dict:
    cache_key = id(model)
    if cache_key in _MC_CACHE:
        return _MC_CACHE[cache_key]
    out = {}
    for L in "ABCD":
        ids = set()
        for variant in (L, f" {L}", f"\n{L}"):
            toks = model.to_tokens(variant, prepend_bos=False)
            t_ids = toks[0].tolist()
            if len(t_ids) == 1:
                ids.add(int(t_ids[0]))
        out[L] = ids
    _MC_CACHE[cache_key] = out
    return out


def _classify_multiple_choice(
    logits_last: Any,
    model,
    verbose: bool = False,
) -> str:
    """Select letter A-D with highest logit (max over variant token ids)."""
    groups = _mc_letter_token_id_groups(model)
    best_letter = "A"
    best_score = float("-inf")
    for L, id_set in groups.items():
        if not id_set:
            continue
        score = max(float(logits_last[tid].item()) for tid in id_set)
        if score > best_score:
            best_score = score
            best_letter = L
    if verbose:
        parts = []
        for L, ids in groups.items():
            if not ids:
                parts.append(f"{L}:none")
            else:
                parts.append(
                    f"{L}:{max(logits_last[i].item() for i in ids):.3f}"
                )
        print("[MC-CLASSIFY] scores=" + ", ".join(parts))
    return best_letter


def evaluate_accuracy(
    model: Any,
    dataset: Iterable[ArithmeticExample],
    task: str,
    verbose: bool = False,
) -> Tuple[int, int]:
    """Evaluate accuracy on a dataset for a given task. Returns (correct, total)."""

    model.eval()
    correct = 0
    total = 0
    device = model.cfg.device
    with torch.no_grad():
        for ex in dataset:
            prompt, target = ex.prompt, ex.target
            toks = model.to_tokens(prompt, prepend_bos=True).to(device)
            logits = model(toks)
            logits_last = logits[0, -1]
            if task == "boolean":
                pred_label, _ = _classify_boolean(
                    logits_last, model, verbose=verbose
                )
                if pred_label == target:
                    correct += 1
            elif task == "mmlu":
                pred_label = _classify_multiple_choice(
                    logits_last, model, verbose=verbose
                )
                if pred_label == target:
                    correct += 1
            else:
                gold_ids = _extract_gold_ids(
                    model, prompt, target, device=device, verbose=verbose
                )
                pred_id = int(logits_last.argmax().item())
                if gold_ids and pred_id in gold_ids:
                    correct += 1
            total += 1
    return correct, total


def _resolve_attn_result_hook_name(model, layer: int) -> str:
    """
    Return a valid hook name for per-head attention results on `layer`.
    Prefer the canonical 'blocks.{layer}.attn.hook_result', but fall back
    to other variants if needed.
    """
    # TransformerLens exposes hook points in model.mod_dict
    available = getattr(model, "mod_dict", {})
    candidates = [
        f"blocks.{layer}.attn.hook_result",  # canonical TL name
        f"blocks.{layer}.hook_result",       # (rare/older code paths)
    ]
    for name in candidates:
        if name in available:
            return name

    # Last resort: scan for something that looks right on this layer
    layer_keys = [k for k in available.keys() if k.startswith(f"blocks.{layer}.")]
    result_like = [k for k in layer_keys if "hook_result" in k]
    if result_like:
        return result_like[0]

    raise KeyError(
        f"No attention result hook found for layer {layer}. "
        f"Available layer hooks: {sorted(layer_keys)}"
    )


def evaluate_accuracy_with_ablation(
    model: Any,
    dataset: Iterable[ArithmeticExample],
    task: str,
    removed: Iterable[Component],
    verbose: bool = False,
) -> Tuple[int, int]:
    """
    Evaluate accuracy under component ablation for a specified task.
    Returns (correct, total).
    """

    model.eval()
    hooks: List[Tuple[str, callable]] = []
    for comp in removed:
        if comp.kind == "head":

            def hook_head(act, hook=None, head_index=comp.index):
                # zero only this head's contribution
                if act.ndim == 4:
                    # [batch, pos, n_heads, d_head]
                    act[:, :, head_index, :] = 0.0
                elif act.ndim == 3:
                    # extremely old TL variant; no per-head index
                    # safest is to zero all
                    act[:, :, :] = 0.0
                return act

            hook_name = _resolve_attn_result_hook_name(model, comp.layer)
            hooks.append((hook_name, hook_head))

        elif comp.kind == "mlp":

            def hook_mlp(act, hook=None):
                act[:, :, :] = 0.0
                return act

            hooks.append((f"blocks.{comp.layer}.hook_mlp_out", hook_mlp))

    correct = 0
    total = 0
    device = model.cfg.device
    with torch.no_grad(), model.hooks(hooks):
        for ex in dataset:
            prompt, target = ex.prompt, ex.target
            toks = model.to_tokens(prompt, prepend_bos=True).to(device)
            logits = model(toks)
            logits_last = logits[0, -1]
            if task == "boolean":
                pred_label, _ = _classify_boolean(
                    logits_last, model, verbose=verbose
                )
                if pred_label == target:
                    correct += 1
            elif task == "mmlu":
                pred_label = _classify_multiple_choice(
                    logits_last, model, verbose=verbose
                )
                if pred_label == target:
                    correct += 1
            else:
                gold_ids = _extract_gold_ids(
                    model, prompt, target, device=device, verbose=verbose
                )
                pred_id = int(logits_last.argmax().item())
                if gold_ids and pred_id in gold_ids:
                    correct += 1
            total += 1
    return correct, total


__all__ = ["evaluate_accuracy", "evaluate_accuracy_with_ablation"]