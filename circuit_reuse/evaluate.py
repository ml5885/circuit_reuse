from __future__ import annotations

from collections import defaultdict
from typing import Iterable, List, Tuple, Any, Dict, Optional
import torch
from .dataset import Example
from .circuit_extraction import Component
from contextlib import nullcontext


def _build_ablation_hooks(removed: Iterable[Component]) -> List[Tuple[str, callable]]:
    """Build forward hooks that zero-ablate the given components.

    Groups components by (kind, layer) so each hook point gets at most one hook,
    even when ablating thousands of neurons.
    """
    heads_by_layer: Dict[int, List[int]] = defaultdict(list)
    neurons_by_layer: Dict[int, List[int]] = defaultdict(list)
    mlp_layers: set = set()

    for comp in removed:
        if comp.kind == "head":
            heads_by_layer[comp.layer].append(comp.index)
        elif comp.kind == "mlp":
            mlp_layers.add(comp.layer)
        elif comp.kind == "neuron":
            neurons_by_layer[comp.layer].append(comp.index)

    hooks: List[Tuple[str, callable]] = []

    for layer, indices in heads_by_layer.items():
        idx = torch.tensor(indices)
        def hook_heads(act, hook=None, idx=idx):
            act[:, :, idx, :] = 0.0
            return act
        # Use hook_z (always wired into downstream computation) instead of
        # hook_result — the latter is observe-only unless cfg.use_attn_result
        # is set True before the model runs, which not all callers do.
        hooks.append((f"blocks.{layer}.attn.hook_z", hook_heads))

    for layer in mlp_layers:
        def hook_mlp(act, hook=None):
            act[:, :, :] = 0.0
            return act
        hooks.append((f"blocks.{layer}.hook_mlp_out", hook_mlp))

    for layer, indices in neurons_by_layer.items():
        idx = torch.tensor(indices)
        def hook_neurons(act, hook=None, idx=idx):
            act[:, :, idx] = 0.0
            return act
        hooks.append((f"blocks.{layer}.mlp.hook_post", hook_neurons))

    return hooks


def compute_corrupted_means(
    model: Any,
    dataset: Iterable[Example],
    layers: Optional[Iterable[int]] = None,
    per_position: bool = False,
) -> Dict[str, "torch.Tensor"]:
    """Mean activations of heads, MLP outputs and MLP neurons over the corrupted
    prompts of ``dataset`` (Wang et al. 2022, Miller et al. 2024).

    Keys are ``z_L{layer}`` (n_heads, d_head), ``mlp_out_L{layer}`` (d_model,) and
    ``neuron_post_L{layer}`` (d_mlp,). With ``per_position`` every tensor gains a
    leading position axis, holding the mean at each token position over the
    prompts that reach it; ``means["_pooled"]`` then carries the position-averaged
    means for positions beyond the longest reference prompt. A pooled mean written
    into every position is off-distribution for neurons that fire only at one
    position, such as the massive-activation neurons at the first token.
    """
    layer_set = set(range(model.cfg.n_layers) if layers is None else (int(l) for l in layers))
    hook_names = {"z": "attn.hook_z", "mlp_out": "hook_mlp_out", "neuron_post": "mlp.hook_post"}
    sums: Dict[str, torch.Tensor] = {}
    counts: Dict[str, torch.Tensor] = {}

    def make_hook(key):
        def hook(act, hook=None):
            a = act.detach().float()  # (batch, pos, ...)
            n = a.shape[1]
            if key not in sums:
                sums[key] = torch.zeros((0,) + a.shape[2:], device=a.device)
                counts[key] = torch.zeros(0, device=a.device)
            if n > sums[key].shape[0]:
                pad = n - sums[key].shape[0]
                sums[key] = torch.cat([sums[key], torch.zeros((pad,) + a.shape[2:], device=a.device)])
                counts[key] = torch.cat([counts[key], torch.zeros(pad, device=a.device)])
            sums[key][:n] += a.sum(dim=0)
            counts[key][:n] += a.shape[0]
            return act
        return hook

    hooks = [(f"blocks.{layer}.{name}", make_hook(f"{key}_L{layer}"))
             for layer in layer_set for key, name in hook_names.items()]
    model.eval()
    with torch.inference_mode(), model.hooks(fwd_hooks=hooks):
        for ex in dataset:
            model(model.to_tokens(ex.corrupted_prompt, prepend_bos=True).to(model.cfg.device))

    means: Dict[str, torch.Tensor] = {}
    pooled: Dict[str, torch.Tensor] = {}
    for key, total in sums.items():
        c = counts[key]
        shape = (-1,) + (1,) * (total.ndim - 1)
        pooled[key] = total.sum(dim=0) / c.sum().clamp(min=1)
        means[key] = total / c.clamp(min=1).view(shape) if per_position else pooled[key]
    if per_position:
        means["_pooled"] = pooled
    return means


def _select_mean(means: Dict[str, "torch.Tensor"], key: str, positions: int) -> "torch.Tensor":
    """The cached mean for ``key`` with a leading axis of length ``positions``:
    the per-position means where the reference prompts reach, the pooled mean beyond."""
    mean = means.get(key)
    if mean is None:
        raise KeyError(f"No mean cached for {key}")
    if "_pooled" not in means:
        return mean.unsqueeze(0).expand(positions, *mean.shape)
    if positions <= mean.shape[0]:
        return mean[:positions]
    tail = means["_pooled"][key].unsqueeze(0).expand(positions - mean.shape[0], *mean.shape[1:])
    return torch.cat([mean, tail])


def _build_mean_ablation_hooks(
    removed: Iterable[Component],
    means: Dict[str, "torch.Tensor"],
) -> List[Tuple[str, callable]]:
    """Same as _build_ablation_hooks but replaces components with their
    pre-computed corrupted-distribution means instead of zero."""
    heads_by_layer: Dict[int, List[int]] = defaultdict(list)
    mlp_layers: set = set()
    neurons_by_layer: Dict[int, List[int]] = defaultdict(list)
    for comp in removed:
        if comp.kind == "head":
            heads_by_layer[comp.layer].append(comp.index)
        elif comp.kind == "mlp":
            mlp_layers.add(comp.layer)
        elif comp.kind == "neuron":
            neurons_by_layer[comp.layer].append(comp.index)

    hooks: List[Tuple[str, callable]] = []

    for layer, indices in heads_by_layer.items():
        idx = torch.tensor(indices)
        def hook_heads(act, hook=None, idx=idx, key=f"z_L{layer}"):
            mean = _select_mean(means, key, act.shape[1]).to(device=act.device, dtype=act.dtype)
            act[:, :, idx.to(act.device), :] = mean[:, idx.to(mean.device), :][None]
            return act
        hooks.append((f"blocks.{layer}.attn.hook_z", hook_heads))

    for layer in mlp_layers:
        def hook_mlp(act, hook=None, key=f"mlp_out_L{layer}"):
            act[:, :, :] = _select_mean(means, key, act.shape[1]).to(device=act.device, dtype=act.dtype)[None]
            return act
        hooks.append((f"blocks.{layer}.hook_mlp_out", hook_mlp))

    for layer, indices in neurons_by_layer.items():
        idx = torch.tensor(indices)
        def hook_neurons(act, hook=None, idx=idx, key=f"neuron_post_L{layer}"):
            mean = _select_mean(means, key, act.shape[1]).to(device=act.device, dtype=act.dtype)
            act[:, :, idx.to(act.device)] = mean[:, idx.to(mean.device)][None]
            return act
        hooks.append((f"blocks.{layer}.mlp.hook_post", hook_neurons))

    return hooks


def evaluate_accuracy_with_mean_ablation(
    model: Any,
    dataset: Iterable[Example],
    task: str,
    removed: Iterable[Component],
    means: Dict[str, "torch.Tensor"],
    verbose: bool = False,
) -> Tuple[int, int]:
    """Same evaluation loop as evaluate_accuracy_with_ablation, but ablation
    replaces each removed component's activation with its pre-computed
    corrupted-distribution mean rather than zero."""
    model.eval()
    hooks = _build_mean_ablation_hooks(removed, means)

    correct, total = 0, 0
    device = model.cfg.device
    with torch.inference_mode(), model.hooks(fwd_hooks=hooks):
        for ex in dataset:
            logits = model(model.to_tokens(ex.prompt, prepend_bos=True).to(device))
            logits_last = logits[0, -1]

            if task == "boolean":
                pred_label, _ = _classify_boolean(logits_last, model, verbose=verbose)
                if pred_label == ex.target:
                    correct += 1
            elif task == "ioi":
                labels = ex.labels or [ex.target, ex.corrupted_target]
                pred_idx = _classify_ioi(logits_last, model, labels)
                gold_idx = ex.answer_idx if ex.answer_idx is not None else 0
                if pred_idx == gold_idx:
                    correct += 1
            elif task in ("mmlu", "mcqa", "arc_easy", "arc_challenge"):
                labels = ex.labels or ["A", "B", "C", "D"]
                pred_label = _classify_from_labels(logits_last, model, labels)
                if pred_label == ex.target:
                    correct += 1
            else:
                if _check_addition_correct(model, ex.prompt, ex.target, device, logits_last, verbose=verbose):
                    correct += 1
            total += 1
    return correct, total


def _extract_gold_ids(model: Any, prompt: str, target: str, device, verbose: bool = False) -> List[int]:
    """Return token ids for the continuation target, relative to the prompt."""
    prompt_tok = model.to_tokens(prompt, prepend_bos=True).to(device)
    full_tok = model.to_tokens(prompt + target, prepend_bos=True).to(device)
    p_ids = prompt_tok[0].tolist()
    f_ids = full_tok[0].tolist()
    lcp = 0
    for a, b in zip(p_ids, f_ids):
        if a == b:
            lcp += 1
        else:
            break
    if lcp == len(f_ids):
        alt = model.to_tokens(target, prepend_bos=False).to(device)
        fallback_ids = alt[0].tolist() if alt.ndim == 2 else alt.tolist()
        if verbose:
            print(f"[WARN] No divergent boundary; fallback target ids={fallback_ids} for target='{target}'")
        return [int(x) for x in fallback_ids]
    gold_ids = f_ids[lcp:]
    if verbose and not gold_ids:
        print(f"[WARN] Empty gold ids; prompt_len={len(p_ids)} full_len={len(f_ids)}")
    return [int(x) for x in gold_ids]


def _greedy_generate(model: Any, prompt_tokens: torch.Tensor, max_new_tokens: int) -> List[int]:
    """Autoregressively generate tokens via greedy decoding."""
    generated: List[int] = []
    tokens = prompt_tokens
    for _ in range(max_new_tokens):
        logits = model(tokens)
        next_id = int(logits[0, -1].argmax().item())
        generated.append(next_id)
        tokens = torch.cat([tokens, torch.tensor([[next_id]], device=tokens.device)], dim=1)
    return generated


def _get_tokenizer(model: Any):
    """Extract the tokenizer from a model (TransformerLens or HFHookedOLMo)."""
    if hasattr(model, "tokenizer"):
        return model.tokenizer
    if hasattr(model, "tokenizerWrapper"):
        return model.tokenizerWrapper
    return None


def _check_addition_correct(model: Any, prompt: str, target: str, device: str,
                            logits_last: torch.Tensor, verbose: bool = False) -> bool:
    """Check addition correctness using autoregressive decoding for multi-token targets."""
    gold_ids = _extract_gold_ids(model, prompt, target, device=device, verbose=verbose)
    n_target_tokens = len(gold_ids)

    if n_target_tokens <= 1:
        pred_id = int(logits_last.argmax().item())
        return bool(gold_ids and pred_id == gold_ids[0])

    # Multi-token target: do greedy autoregressive decoding
    prompt_tokens = model.to_tokens(prompt, prepend_bos=True).to(device)
    generated = _greedy_generate(model, prompt_tokens, n_target_tokens)

    if verbose:
        tokenizer = _get_tokenizer(model)
        if tokenizer:
            pred_str = tokenizer.decode(generated, skip_special_tokens=True)
            print(f"[ADD] target='{target}' gold_ids={gold_ids} gen={generated} "
                  f"pred_str='{pred_str}' match={generated == gold_ids}")

    return generated == gold_ids


_BOOL_CACHE = {}


def _boolean_token_id_groups(model) -> Tuple[set, set]:
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


def _classify_boolean(logits_last: Any, model, verbose: bool = False) -> Tuple[str, dict]:
    true_ids, false_ids = _boolean_token_id_groups(model)
    id_logits = {f"true:{tid}": float(logits_last[tid].item()) for tid in true_ids}
    id_logits.update({f"false:{fid}": float(logits_last[fid].item()) for fid in false_ids})
    true_score = max((logits_last[tid].item() for tid in true_ids), default=float("-inf"))
    false_score = max((logits_last[fid].item() for fid in false_ids), default=float("-inf"))
    label = "true" if true_score >= false_score else "false"
    if verbose:
        print(f"[BOOL] true={true_score:.3f} false={false_score:.3f} -> {label}")
    return label, id_logits


def _score_first_token(logits_last, model, label: str) -> float:
    """Score a label by the best logit among plausible first-token variants."""
    ids = set()
    for v in (label, f" {label}", f"\n{label}", f": {label}", f":\n{label}"):
        toks = model.to_tokens(v, prepend_bos=False)
        t_ids = toks[0].tolist()
        if len(t_ids) >= 1:
            ids.add(int(t_ids[0]))
    return max((float(logits_last[i].item()) for i in ids), default=float("-inf"))


def _classify_from_labels(logits_last, model, labels: List[str]) -> str:
    scores = [(_score_first_token(logits_last, model, L), L) for L in labels]
    return max(scores, key=lambda x: x[0])[1]


def _classify_ioi(logits_last, model, names: List[str]) -> int:
    """Return index of the predicted name among the two candidates."""
    scores = [(_score_first_token(logits_last, model, n), i) for i, n in enumerate(names)]
    return max(scores, key=lambda x: x[0])[1]


def evaluate_accuracy(model: Any, dataset: Iterable[Example], task: str, verbose: bool = False) -> Tuple[int, int]:
    model.eval()
    correct, total = 0, 0
    device = model.cfg.device
    with torch.inference_mode():
        for ex in dataset:
            prompt, target = ex.prompt, ex.target
            logits = model(model.to_tokens(prompt, prepend_bos=True).to(device))
            logits_last = logits[0, -1]

            if task == "boolean":
                pred_label, _ = _classify_boolean(logits_last, model, verbose=verbose)
                if pred_label == target:
                    correct += 1

            elif task == "ioi":
                labels = ex.labels or [ex.target, ex.corrupted_target]
                pred_idx = _classify_ioi(logits_last, model, labels)
                gold_idx = ex.answer_idx if ex.answer_idx is not None else 0
                if pred_idx == gold_idx:
                    correct += 1

            elif task in ("mmlu", "mcqa", "arc_easy", "arc_challenge"):
                labels = ex.labels or ["A", "B", "C", "D"]
                pred_label = _classify_from_labels(logits_last, model, labels)
                if pred_label == target:
                    correct += 1

            else:
                if _check_addition_correct(model, prompt, target, device, logits_last, verbose=verbose):
                    correct += 1

            total += 1
    return correct, total


def evaluate_accuracy_with_ablation(
    model: Any, dataset: Iterable[Example], task: str, removed: Iterable[Component], verbose: bool = False
) -> Tuple[int, int]:
    model.eval()
    hooks = _build_ablation_hooks(removed)

    correct, total = 0, 0
    device = model.cfg.device
    with torch.inference_mode(), model.hooks(fwd_hooks=hooks):
        for ex in dataset:
            logits = model(model.to_tokens(ex.prompt, prepend_bos=True).to(device))
            logits_last = logits[0, -1]

            if task == "boolean":
                pred_label, _ = _classify_boolean(logits_last, model, verbose=verbose)
                if pred_label == ex.target:
                    correct += 1

            elif task == "ioi":
                labels = ex.labels or [ex.target, ex.corrupted_target]
                pred_idx = _classify_ioi(logits_last, model, labels)
                gold_idx = ex.answer_idx if ex.answer_idx is not None else 0
                if pred_idx == gold_idx:
                    correct += 1

            elif task in ("mmlu", "mcqa", "arc_easy", "arc_challenge"):
                labels = ex.labels or ["A", "B", "C", "D"]
                pred_label = _classify_from_labels(logits_last, model, labels)
                if pred_label == ex.target:
                    correct += 1

            else:
                if _check_addition_correct(model, ex.prompt, ex.target, device, logits_last, verbose=verbose):
                    correct += 1

            total += 1
    return correct, total


def evaluate_predictions(
    model: Any,
    dataset: Iterable[Example],
    task: str,
    removed: Iterable[Component] | None = None,
    verbose: bool = False,
) -> Tuple[int, int, List[Dict[str, Any]]]:
    """
    Evaluate and also return per-example predictions.

    Returns:
      correct, total, per_example list with:
        {"prompt": str, "target": str, "pred": str | int, "is_correct": bool}
    """
    model.eval()
    hooks = _build_ablation_hooks(removed) if removed else []

    per_ex: List[Dict[str, Any]] = []
    correct, total = 0, 0
    device = model.cfg.device
    ctx = model.hooks(fwd_hooks=hooks) if hooks else nullcontext()

    with ctx:
        with torch.inference_mode():
            for ex in dataset:
                logits = model(model.to_tokens(ex.prompt, prepend_bos=True).to(device))
                logits_last = logits[0, -1]

                if task == "boolean":
                    pred_label, _ = _classify_boolean(logits_last, model, verbose=verbose)
                    gold = ex.target
                    ok = (pred_label == gold)
                    per_ex.append({"prompt": ex.prompt, "target": gold, "pred": pred_label, "is_correct": bool(ok)})

                elif task == "ioi":
                    labels = ex.labels or [ex.target, ex.corrupted_target]
                    pred_idx = _classify_ioi(logits_last, model, labels)
                    gold_idx = ex.answer_idx if ex.answer_idx is not None else 0
                    ok = (pred_idx == gold_idx)
                    pred_name = labels[pred_idx] if 0 <= pred_idx < len(labels) else str(pred_idx)
                    gold_name = labels[gold_idx] if 0 <= gold_idx < len(labels) else str(gold_idx)
                    per_ex.append({"prompt": ex.prompt, "target": gold_name, "pred": pred_name, "is_correct": bool(ok)})

                elif task in ("mmlu", "mcqa", "arc_easy", "arc_challenge"):
                    labels = ex.labels or ["A", "B", "C", "D"]
                    pred_label = _classify_from_labels(logits_last, model, labels)
                    gold = ex.target
                    ok = (pred_label == gold)
                    per_ex.append({"prompt": ex.prompt, "target": gold, "pred": pred_label, "is_correct": bool(ok)})

                else:
                    ok = _check_addition_correct(model, ex.prompt, ex.target, device, logits_last, verbose=verbose)
                    per_ex.append({"prompt": ex.prompt, "target": ex.target, "is_correct": bool(ok)})

                correct += int(per_ex[-1]["is_correct"])
                total += 1

    return correct, total, per_ex


__all__ = [
    "evaluate_accuracy",
    "evaluate_accuracy_with_ablation",
    "evaluate_accuracy_with_mean_ablation",
    "evaluate_predictions",
    "compute_corrupted_means",
]
