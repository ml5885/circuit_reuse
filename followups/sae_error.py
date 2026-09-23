"""How much of each task's attribution and ablation effect goes through the SAE error
terms, which the feature-circuit experiments hold fixed at their clean values.

Attribution: the SAE error term of every layer is scored with the same EAP-IG
node score as the latents (activation difference times integrated gradient), on
the first --n-attr training examples. We report the share of positive and of
absolute attribution on the error nodes, and how many error nodes would enter
each example's top-K% if they were ranked with the latents.

Ablation: on the cross-task evaluation examples, the task's feature circuit is
zero-ablated as in the paper, the error terms of all layers are replaced by their
per-position mean over the task's corrupted prompts, and both are done together.

    python -m followups.sae_error
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from circuit_reuse.circuit_extraction import CircuitExtractor, Component
from circuit_reuse.evaluate import _build_ablation_hooks, evaluate_graded, summarize_graded
from circuit_reuse.sae import ERROR_HOOK, SAESpec, attach_saes
from followups.common import OUT, TASKS, circuit, eval_datasets, load_model, train_examples, write_json

SAE_RESULTS = Path("results/sae/extraction/sae_eap_ig_feature")


def error_scorer(layer: int):
    def scorer(attr: torch.Tensor):
        return {Component(layer, "error", 0): float(attr.sum())}
    return scorer


def attribution(model, model_name: str, task: str, n: int, k: int = 10) -> dict:
    extractor = CircuitExtractor(model, method="eap_ig", granularity="feature", ig_steps=5)
    feature_targets = extractor._hook_targets
    extractor._hook_targets = lambda: feature_targets() + [
        (ERROR_HOOK.format(layer=l), error_scorer(l)) for l in range(model.cfg.n_layers)]
    _, per_example = extractor.extract_circuits_from_examples(
        train_examples(task, model_name)[:n], task, amp=True, device=model.cfg.device)
    stats = defaultdict(list)
    for scores in per_example:
        err = np.array([v for c, v in scores.items() if c.kind == "error"])
        feat = np.array([v for c, v in scores.items() if c.kind == "feature"])
        stats["error_share_pos"].append(np.clip(err, 0, None).sum() / (np.clip(err, 0, None).sum() + np.clip(feat, 0, None).sum()))
        stats["error_share_abs"].append(np.abs(err).sum() / (np.abs(err).sum() + np.abs(feat).sum()))
        cutoff = np.sort(feat)[::-1][max(1, int(len(feat) * k / 100)) - 1]
        stats["errors_in_top_k"].append(int((err >= cutoff).sum()))
        stats["error_by_layer"].append(err.tolist())
    return {key: (np.mean(v, axis=0).tolist() if key == "error_by_layer" else float(np.mean(v)))
            for key, v in stats.items()} | {"n_examples": len(per_example)}


def error_means(model, dataset) -> dict[int, torch.Tensor]:
    """Per-position mean of each layer's error term over the corrupted prompts."""
    n_pos = max(model.to_tokens(ex.corrupted_prompt, prepend_bos=True).shape[1] for ex in dataset)
    layers = range(model.cfg.n_layers)
    sums = {l: torch.zeros((n_pos, model.cfg.d_model), device=model.cfg.device) for l in layers}
    counts = torch.zeros(n_pos, device=model.cfg.device)

    def hook(layer):
        def fn(act, hook=None):
            sums[layer][: act.shape[1]] += act[0].float()
            return act
        return fn

    with torch.inference_mode(), model.hooks(fwd_hooks=[(ERROR_HOOK.format(layer=l), hook(l)) for l in layers]):
        for ex in dataset:
            tokens = model.to_tokens(ex.corrupted_prompt, prepend_bos=True).to(model.cfg.device)
            model(tokens)
            counts[: tokens.shape[1]] += 1
    return {l: sums[l] / counts.clamp(min=1)[:, None] for l in layers}


def error_hooks(means: dict[int, torch.Tensor]):
    def hook(layer):
        def fn(act, hook=None):
            m = means[layer]
            n = min(act.shape[1], m.shape[0])
            act[:, :n] = m[:n].to(act.dtype)[None]
            act[:, n:] = m.mean(0).to(act.dtype)
            return act
        return fn
    return [(ERROR_HOOK.format(layer=l), hook(l)) for l in means]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-2-2b")
    parser.add_argument("--n-attr", type=int, default=200)
    args = parser.parse_args()
    model = load_model(args.model)
    attach_saes(model, SAESpec())
    datasets = eval_datasets(args.model)
    path = OUT / "sae_error" / f"{args.model.replace('/', '_')}.json"
    results = {}
    for task in TASKS:
        kept = circuit(args.model, task, "eap_ig", "feature", results_dir=SAE_RESULTS)
        ds = datasets[task]
        err = error_hooks(error_means(model, ds))
        results[task] = {
            "circuit_size": len(kept),
            "attribution": attribution(model, args.model, task, args.n_attr),
            "full": summarize_graded(evaluate_graded(model, ds, task)),
            "circuit": summarize_graded(evaluate_graded(model, ds, task, _build_ablation_hooks(kept))),
            "errors": summarize_graded(evaluate_graded(model, ds, task, err)),
            "circuit_and_errors": summarize_graded(evaluate_graded(model, ds, task, _build_ablation_hooks(kept) + err)),
        }
        r = results[task]
        print(f"[{task}] error share of |attr| {r['attribution']['error_share_abs']:.2f}; accuracy "
              f"full {r['full']['correct']:.2f} circuit {r['circuit']['correct']:.2f} "
              f"errors {r['errors']['correct']:.2f} both {r['circuit_and_errors']['correct']:.2f}")
        write_json(path, {"model": args.model, "sae": SAESpec().slug, "tasks": results})


if __name__ == "__main__":
    main()
