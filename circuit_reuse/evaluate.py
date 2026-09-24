from __future__ import annotations

from collections import defaultdict
from typing import Iterable, List, Tuple, Any, Dict, Optional
import torch
from .dataset import Example
from .circuit_extraction import Component
from .sae import FEATURE_HOOK, attached_sae_layers
from contextlib import nullcontext


KIND_KEYS = {"head": "z", "mlp": "mlp_out", "neuron": "neuron_post", "feature": "sae_acts"}


def _build_ablation_hooks(removed: Iterable[Component]) -> List[Tuple[str, callable]]:
    """Build forward hooks that zero-ablate the given components.

    Groups components by (kind, layer) so each hook point gets at most one hook,
    even when ablating thousands of neurons.
    """
    heads_by_layer: Dict[int, List[int]] = defaultdict(list)
    neurons_by_layer: Dict[int, List[int]] = defaultdict(list)
    features_by_layer: Dict[int, List[int]] = defaultdict(list)
    mlp_layers: set = set()

    for comp in removed:
        if comp.kind == "head":
            heads_by_layer[comp.layer].append(comp.index)
        elif comp.kind == "mlp":
            mlp_layers.add(comp.layer)
        elif comp.kind == "neuron":
            neurons_by_layer[comp.layer].append(comp.index)
        elif comp.kind == "feature":
            features_by_layer[comp.layer].append(comp.index)

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

    for layer, indices in features_by_layer.items():
        idx = torch.tensor(indices)
        def hook_features(act, hook=None, idx=idx):
            act[:, :, idx] = 0.0
            return act
        hooks.append((FEATURE_HOOK.format(layer=layer), hook_features))

    return hooks


def compute_corrupted_means(
    model: Any,
    dataset: Iterable[Example],
    layers: Optional[Iterable[int]] = None,
    per_position: bool = False,
    kinds: Optional[Iterable[str]] = None,
) -> Dict[str, "torch.Tensor"]:
    """Mean activations of heads, MLP outputs and MLP neurons over the corrupted
    prompts of ``dataset`` (Wang et al. 2022, Miller et al. 2024).

    Keys are ``z_L{layer}`` (n_heads, d_head), ``mlp_out_L{layer}`` (d_model,),
    ``neuron_post_L{layer}`` (d_mlp,) and, for layers with an attached SAE,
    ``sae_acts_L{layer}`` (d_sae,). ``kinds`` restricts the cache to the component
    kinds that will be ablated, which matters at neuron and feature granularity
    where the unused caches are the large ones. With ``per_position`` every tensor gains a
    leading position axis, holding the mean at each token position over the
    prompts that reach it; ``means["_pooled"]`` then carries the position-averaged
    means for positions beyond the longest reference prompt. A pooled mean written
    into every position is off-distribution for neurons that fire only at one
    position, such as the massive-activation neurons at the first token.
    """
    layer_set = set(range(model.cfg.n_layers) if layers is None else (int(l) for l in layers))
    wanted = None if kinds is None else {KIND_KEYS[k] for k in kinds}
    hook_names = {key: name for key, name in
                  (("z", "attn.hook_z"), ("mlp_out", "hook_mlp_out"), ("neuron_post", "mlp.hook_post"))
                  if wanted is None or key in wanted}
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
    if wanted is None or "sae_acts" in wanted:
        hooks += [(FEATURE_HOOK.format(layer=layer), make_hook(f"sae_acts_L{layer}"))
                  for layer in attached_sae_layers(model) if layer in layer_set]
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
    features_by_layer: Dict[int, List[int]] = defaultdict(list)
    for comp in removed:
        if comp.kind == "head":
            heads_by_layer[comp.layer].append(comp.index)
        elif comp.kind == "mlp":
            mlp_layers.add(comp.layer)
        elif comp.kind == "neuron":
            neurons_by_layer[comp.layer].append(comp.index)
        elif comp.kind == "feature":
            features_by_layer[comp.layer].append(comp.index)

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

    for layer, indices in features_by_layer.items():
        idx = torch.tensor(indices)
        def hook_features(act, hook=None, idx=idx, key=f"sae_acts_L{layer}"):
            mean = _select_mean(means, key, act.shape[1]).to(device=act.device, dtype=act.dtype)
            act[:, :, idx.to(act.device)] = mean[:, idx.to(mean.device)][None]
            return act
        hooks.append((FEATURE_HOOK.format(layer=layer), hook_features))

    return hooks


def evaluate_accuracy_with_mean_ablation(
    model: Any,
    dataset: Iterable[Example],
    task: str,
    removed: Iterable[Component],
    means: Dict[str, "torch.Tensor"],
    verbose: bool = False,
) -> Tuple[int, int]:
    """Same as evaluate_accuracy_with_ablation, but each removed component's activation is
    replaced with its pre-computed corrupted-distribution mean rather than zero."""
    rows = _predict(model, dataset, task, _build_mean_ablation_hooks(removed, means), verbose)
    return sum(r["is_correct"] for r in rows), len(rows)


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
    cache_key = model.cfg.model_name
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
    label = " true" if true_score >= false_score else " false"
    if verbose:
        print(f"[BOOL] true={true_score:.3f} false={false_score:.3f} -> {label}")
    return label, id_logits


def _score_first_token(logits_last, model, label: str) -> float:
    """Score a label by the best logit over its first token with and without a leading space.
    Spellings with a leading newline or colon are not used: their first token is the newline
    or colon, which is shared by every label and makes them tie."""
    ids = set()
    for v in (label, f" {label}"):
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


EVAL_TOKENS_PER_BATCH = 8192


def _length_batches(model: Any, dataset: List[Example]):
    """Yield (indices, tokens) for batches of prompts with the same token length, so that a
    batch needs no padding and each row gets the logits it would get on its own."""
    by_len: Dict[int, List[Tuple[int, torch.Tensor]]] = defaultdict(list)
    for i, ex in enumerate(dataset):
        tok = model.to_tokens(ex.prompt, prepend_bos=True)[0]
        by_len[len(tok)].append((i, tok))
    for length, items in by_len.items():
        size = max(1, EVAL_TOKENS_PER_BATCH // length)
        for b in range(0, len(items), size):
            chunk = items[b:b + size]
            yield [i for i, _ in chunk], torch.stack([t for _, t in chunk]).to(model.cfg.device)


def _greedy_batch(model: Any, tokens: torch.Tensor, first: torch.Tensor, steps: int) -> torch.Tensor:
    """Greedy continuation of equal-length prompts; ``first`` is the argmax at the last prompt
    position. Returns (batch, steps) generated ids."""
    generated = [first]
    for _ in range(steps - 1):
        tokens = torch.cat([tokens, generated[-1][:, None]], dim=1)
        generated.append(model(tokens)[:, -1].argmax(-1))
    return torch.stack(generated, dim=1)


def _predict(model: Any, dataset: Iterable[Example], task: str, hooks=(), verbose: bool = False) -> List[Dict[str, Any]]:
    """Per-example predictions of the model under ``hooks``:
    {"prompt", "target", "pred", "is_correct"} (no "pred" for addition)."""
    model.eval()
    dataset = list(dataset)
    rows: List[Dict[str, Any]] = [None] * len(dataset)
    ctx = model.hooks(fwd_hooks=list(hooks)) if hooks else nullcontext()
    with ctx, torch.inference_mode():
        for idx, tokens in _length_batches(model, dataset):
            logits_last = model(tokens)[:, -1]
            if task not in ("boolean", "ioi", "mmlu", "mcqa", "arc_easy", "arc_challenge"):  # addition
                gold = [_extract_gold_ids(model, dataset[i].prompt, dataset[i].target, tokens.device, verbose) for i in idx]
                steps = max(1, max(len(g) for g in gold))
                gen = _greedy_batch(model, tokens, logits_last.argmax(-1), steps).tolist()
                for r, i in enumerate(idx):
                    ok = bool(gold[r]) and gen[r][:len(gold[r])] == gold[r]
                    rows[i] = {"prompt": dataset[i].prompt, "target": dataset[i].target, "is_correct": ok}
                continue
            for r, i in enumerate(idx):
                ex, logits = dataset[i], logits_last[r]
                if task == "boolean":
                    pred, _ = _classify_boolean(logits, model, verbose=verbose)
                    gold = ex.target
                elif task == "ioi":
                    labels = ex.labels or [ex.target, ex.corrupted_target]
                    pred = labels[_classify_ioi(logits, model, labels)]
                    gold = labels[ex.answer_idx if ex.answer_idx is not None else 0]
                else:
                    pred = _classify_from_labels(logits, model, ex.labels or ["A", "B", "C", "D"])
                    gold = ex.target.strip()
                rows[i] = {"prompt": ex.prompt, "target": gold, "pred": pred, "is_correct": pred == gold}
    return rows


def evaluate_accuracy(model: Any, dataset: Iterable[Example], task: str, verbose: bool = False) -> Tuple[int, int]:
    rows = _predict(model, dataset, task, verbose=verbose)
    return sum(r["is_correct"] for r in rows), len(rows)


def evaluate_accuracy_with_ablation(
    model: Any, dataset: Iterable[Example], task: str, removed: Iterable[Component], verbose: bool = False
) -> Tuple[int, int]:
    rows = _predict(model, dataset, task, _build_ablation_hooks(removed), verbose)
    return sum(r["is_correct"] for r in rows), len(rows)


def evaluate_predictions(
    model: Any,
    dataset: Iterable[Example],
    task: str,
    removed: Iterable[Component] | None = None,
    verbose: bool = False,
) -> Tuple[int, int, List[Dict[str, Any]]]:
    """Evaluate and also return per-example predictions
    ({"prompt", "target", "pred", "is_correct"})."""
    rows = _predict(model, dataset, task, _build_ablation_hooks(removed) if removed else (), verbose)
    return sum(r["is_correct"] for r in rows), len(rows), rows


__all__ = [
    "evaluate_accuracy",
    "evaluate_accuracy_with_ablation",
    "evaluate_accuracy_with_mean_ablation",
    "evaluate_predictions",
    "compute_corrupted_means",
]


def _label_token_ids(model, label: str) -> List[int]:
    """First-token ids of the variants _score_first_token considers for ``label``."""
    ids = set()
    for v in (label, f" {label}"):
        toks = model.to_tokens(v, prepend_bos=False)[0].tolist()
        if toks:
            ids.add(int(toks[0]))
    return sorted(ids)


def evaluate_graded(model: Any, dataset: Iterable[Example], task: str,
                    hooks: List[Tuple[str, callable]] = ()) -> List[Dict[str, float]]:
    """Per-example correctness, under the same rule as evaluate_accuracy, together
    with a graded measure that does not saturate at chance: the log-probability of
    the gold answer (summed over its tokens for addition, over the first-token
    variants of the gold label otherwise). For IOI also the logit difference
    between the indirect object and the subject, and the indirect object's rank in
    the full vocabulary."""
    model.eval()
    device = model.cfg.device
    out: List[Dict[str, float]] = []
    with torch.inference_mode(), model.hooks(fwd_hooks=list(hooks)):
        for ex in dataset:
            tokens = model.to_tokens(ex.prompt, prepend_bos=True).to(device)
            logits_last = model(tokens)[0, -1].float()
            logp = torch.log_softmax(logits_last, dim=-1)
            row: Dict[str, float] = {}
            if task == "boolean":
                true_ids, false_ids = _boolean_token_id_groups(model)
                gold = sorted(true_ids if ex.target == " true" else false_ids)
                row["correct"] = float(_classify_boolean(logits_last, model)[0] == ex.target)
                row["logprob"] = float(torch.logsumexp(logp[gold], 0))
            elif task == "ioi":
                labels = ex.labels or [ex.target, ex.corrupted_target]
                gold_idx = ex.answer_idx if ex.answer_idx is not None else 0
                row["correct"] = float(_classify_ioi(logits_last, model, labels) == gold_idx)
                io, s = labels[gold_idx], labels[1 - gold_idx]
                io_ids = _label_token_ids(model, io)
                row["logprob"] = float(torch.logsumexp(logp[io_ids], 0))
                row["logit_diff"] = _score_first_token(logits_last, model, io) - _score_first_token(logits_last, model, s)
                best = max(io_ids, key=lambda i: float(logits_last[i]))
                row["io_rank"] = float((logits_last > logits_last[best]).sum())
            elif task in ("mmlu", "mcqa", "arc_easy", "arc_challenge"):
                labels = ex.labels or ["A", "B", "C", "D"]
                row["correct"] = float(_classify_from_labels(logits_last, model, labels) == ex.target.strip())
                row["logprob"] = float(torch.logsumexp(logp[_label_token_ids(model, ex.target.strip())], 0))
            else:
                gold_ids = _extract_gold_ids(model, ex.prompt, ex.target, device)
                row["correct"] = float(_check_addition_correct(model, ex.prompt, ex.target, device, logits_last))
                forced = torch.cat([tokens, torch.tensor([gold_ids[:-1]], device=device, dtype=tokens.dtype)], dim=1)
                step_logp = torch.log_softmax(model(forced)[0, tokens.shape[1] - 1:].float(), dim=-1)
                row["logprob"] = float(step_logp[torch.arange(len(gold_ids), device=device), torch.tensor(gold_ids, device=device)].sum())
            out.append(row)
    return out


def summarize_graded(rows: List[Dict[str, float]]) -> Dict[str, float]:
    """Means of every per-example field, plus the median IOI rank."""
    summary = {k: float(sum(r[k] for r in rows) / len(rows)) for k in rows[0]} if rows else {}
    if rows and "io_rank" in rows[0]:
        ranks = sorted(r["io_rank"] for r in rows)
        summary["io_rank_median"] = ranks[len(ranks) // 2]
    return summary
