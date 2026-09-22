"""Sparse-autoencoder features as circuit units.

A JumpReLU SAE (Gemma Scope, Lieberum et al. 2024) is spliced into the residual
stream of every chosen layer. The block output becomes

    decode(acts) + (x - decode(acts))

which equals ``x`` on the unablated forward pass, so the model's predictions are
unchanged, while the feature activations ``acts`` become a hook point
(``blocks.{L}.sae.hook_sae_acts_post``) that the extractor scores and the
evaluator ablates. The error term is held at its clean value, so ablating a
feature removes only that feature's contribution (Marks et al. 2025).

Gradients follow the same convention: the metric's gradient with respect to
``acts`` is the gradient through the decoder, and the gradient with respect to
``x`` is the identity, so upstream layers see the same gradient as without the
SAE.

    python -m circuit_reuse.sae --model google/gemma-2-2b   # reconstruction check
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
from torch import nn
from huggingface_hub import hf_hub_download, list_repo_files
from transformer_lens.hook_points import HookPoint

FEATURE_HOOK = "blocks.{layer}.sae.hook_sae_acts_post"


@dataclass(frozen=True)
class SAESpec:
    """Which SAEs to attach. ``l0`` picks, per layer, the trained SAE whose average
    L0 is closest to it; Gemma Scope ships several per layer and width."""
    release: str = "google/gemma-scope-2b-pt-res"
    width: str = "16k"
    l0: int = 100
    layers: Optional[tuple[int, ...]] = None

    @property
    def slug(self) -> str:
        return f"{self.release.split('/')[-1]}-{self.width}-l0{self.l0}"


class JumpReLUSAE(nn.Module):
    def __init__(self, W_enc, W_dec, b_enc, b_dec, threshold, path: str = ""):
        super().__init__()
        self.W_enc = nn.Parameter(W_enc, requires_grad=False)
        self.W_dec = nn.Parameter(W_dec, requires_grad=False)
        self.b_enc = nn.Parameter(b_enc, requires_grad=False)
        self.b_dec = nn.Parameter(b_dec, requires_grad=False)
        self.threshold = nn.Parameter(threshold, requires_grad=False)
        self.path = path
        self.hook_sae_acts_post = HookPoint()

    @property
    def d_sae(self) -> int:
        return self.W_enc.shape[1]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = x @ self.W_enc + self.b_enc
        return torch.relu(pre) * (pre > self.threshold)

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        return acts @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.to(self.W_enc.dtype)
        acts = self.encode(x32.detach())
        err = x32 - self.decode(acts).detach()
        if torch.is_grad_enabled():
            acts.requires_grad_(True)
        acts = self.hook_sae_acts_post(acts)
        return (self.decode(acts) + err).to(x.dtype)


def _closest_l0_dir(files: List[str], layer: int, width: str, l0: int) -> str:
    pattern = re.compile(rf"^layer_{layer}/width_{width}/average_l0_(\d+)/params\.npz$")
    found = {int(m.group(1)): m.group(0) for f in files if (m := pattern.match(f))}
    if not found:
        raise FileNotFoundError(f"no layer_{layer}/width_{width} SAE in release")
    return found[min(found, key=lambda v: abs(v - l0))]


def load_saes(spec: SAESpec, n_layers: int, device: str, dtype=torch.float32) -> Dict[int, JumpReLUSAE]:
    files = list_repo_files(spec.release)
    layers = spec.layers or tuple(range(n_layers))
    saes = {}
    for layer in layers:
        path = _closest_l0_dir(files, layer, spec.width, spec.l0)
        params = np.load(hf_hub_download(spec.release, path))
        tensors = {k: torch.tensor(params[k], dtype=dtype, device=device) for k in ("W_enc", "W_dec", "b_enc", "b_dec", "threshold")}
        saes[layer] = JumpReLUSAE(**tensors, path=path).to(device)
    return saes


def attach_saes(model, spec: SAESpec, saes: Optional[Dict[int, JumpReLUSAE]] = None) -> Dict[int, JumpReLUSAE]:
    """Splice one SAE per layer into ``blocks.{L}.hook_resid_post`` as a permanent
    hook, and register the feature hook points with the model."""
    if saes is None:
        saes = load_saes(spec, model.cfg.n_layers, model.cfg.device)
    for layer, sae in saes.items():
        block = model.blocks[layer]
        block.add_module("sae", sae)
        model.add_hook(f"blocks.{layer}.hook_resid_post", lambda x, hook, sae=sae: sae(x), is_permanent=True)
    model.setup()  # registers blocks.{L}.sae.hook_sae_acts_post in model.hook_dict
    model.sae_spec = spec
    return saes


def attached_sae_layers(model) -> List[int]:
    return [i for i, block in enumerate(getattr(model, "blocks", [])) if hasattr(block, "sae")]


def feature_hook_names(model) -> Dict[int, str]:
    return {layer: FEATURE_HOOK.format(layer=layer) for layer in attached_sae_layers(model)}


def add_sae_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sae-release", default=SAESpec.release, help="HF repo with Gemma Scope-style params.npz files")
    parser.add_argument("--sae-width", default=SAESpec.width)
    parser.add_argument("--sae-l0", type=int, default=SAESpec.l0, help="per layer, use the SAE whose average L0 is closest to this")
    parser.add_argument("--sae-layers", default=None, help="comma-separated layers (default: all)")


def spec_from_args(args) -> SAESpec:
    layers = tuple(int(x) for x in args.sae_layers.split(",")) if args.sae_layers else None
    return SAESpec(release=args.sae_release, width=args.sae_width, l0=args.sae_l0, layers=layers)


@torch.inference_mode()
def reconstruction_report(model, prompts: Iterable[str]) -> Dict[int, dict]:
    """Per layer: fraction of variance unexplained by the SAE and mean L0, over
    the non-BOS positions of ``prompts``. A basis mismatch shows up as FVU near
    or above 1."""
    layers = attached_sae_layers(model)
    stats = {layer: dict(resid=0.0, err=0.0, l0=0.0, n=0) for layer in layers}

    def resid_hook(layer):
        def hook(x, hook=None):
            sae = model.blocks[layer].sae
            x32 = x[:, 1:].to(sae.W_enc.dtype)
            acts = sae.encode(x32)
            s = stats[layer]
            s["resid"] += float(((x32 - x32.mean(dim=(0, 1))) ** 2).sum())
            s["err"] += float(((x32 - sae.decode(acts)) ** 2).sum())
            s["l0"] += float((acts > 0).sum())
            s["n"] += x32.shape[0] * x32.shape[1]
            return x
        return hook

    # Whether this hook runs before or after the splice, x equals the residual stream.
    with model.hooks(fwd_hooks=[(f"blocks.{layer}.hook_resid_post", resid_hook(layer)) for layer in layers]):
        for prompt in prompts:
            model(model.to_tokens(prompt, prepend_bos=True).to(model.cfg.device))
    return {layer: dict(fvu=s["err"] / s["resid"], l0=s["l0"] / s["n"], path=model.blocks[layer].sae.path)
            for layer, s in stats.items()}


def main():
    from models.olmo_adapter import load_model_any
    parser = argparse.ArgumentParser(description="Attach SAEs and report reconstruction quality.")
    parser.add_argument("--model", default="google/gemma-2-2b")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add_sae_args(parser)
    args = parser.parse_args()
    model = load_model_any(args.model, device=args.device, torch_dtype=torch.bfloat16)
    prompts = ["The capital of France is Paris.", "Compute: 123 + 456 = 579",
               "When Mary and John went to the store, John gave a drink to Mary."]
    clean = [model(model.to_tokens(p, prepend_bos=True).to(args.device)) for p in prompts]
    attach_saes(model, spec_from_args(args))
    spliced = [model(model.to_tokens(p, prepend_bos=True).to(args.device)) for p in prompts]
    print("max |logit diff| with SAEs spliced (should be ~0):",
          max(float((a - b).abs().max()) for a, b in zip(clean, spliced)))
    for layer, r in reconstruction_report(model, prompts).items():
        print(f"layer {layer:2d}  FVU {r['fvu']:.3f}  L0 {r['l0']:6.1f}  {r['path']}")


if __name__ == "__main__":
    main()
