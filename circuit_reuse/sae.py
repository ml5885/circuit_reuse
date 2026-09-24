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

A cross-layer transcoder (CLT; circuit-tracer's ``mntss/clt-gemma-2-2b-426k``, Ameisen et
al. 2025) is spliced the same way with ``--sae-kind clt``: layer L's features read
``blocks.{L}.hook_resid_mid`` and their decoders write into ``hook_mlp_out`` of layer L and
every later layer. Each MLP output becomes ``mlp_out + sum_L (acts_L - acts_L.clean) W_dec``,
so the unablated forward pass is unchanged, ablating a feature removes its decoded
contribution from every downstream MLP output, and the MLPs themselves play the role of the
error term. Layer L's feature activations sit at the same hook point as an SAE's.

    python -m circuit_reuse.sae --model google/gemma-2-2b   # reconstruction check
    python -m circuit_reuse.sae --model google/gemma-2-2b --sae-kind clt --sae-release mntss/clt-gemma-2-2b-426k
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
from safetensors.torch import load_file
from transformer_lens.hook_points import HookPoint

FEATURE_HOOK = "blocks.{layer}.sae.hook_sae_acts_post"
ERROR_HOOK = "blocks.{layer}.sae.hook_sae_error"


@dataclass(frozen=True)
class SAESpec:
    """Which SAEs to attach. ``l0`` picks, per layer, the trained SAE whose average
    L0 is closest to it; Gemma Scope ships several per layer and width."""
    release: str = "google/gemma-scope-2b-pt-res"
    width: str = "16k"
    l0: int = 100
    layers: Optional[tuple[int, ...]] = None
    dtype: str = "float32"
    kind: str = "sae"  # or "clt"

    @property
    def slug(self) -> str:
        if self.kind == "clt":
            return self.release.split("/")[-1]
        suffix = "" if self.dtype == "float32" else f"-{self.dtype}"
        return f"{self.release.split('/')[-1]}-{self.width}-l0{self.l0}{suffix}"


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
        self.hook_sae_error = HookPoint()

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
        err = self.hook_sae_error(x32 - self.decode(acts).detach())
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


class CLTLayer(nn.Module):
    """Layer ``layer``'s CLT features: a ReLU encoder on the pre-MLP residual stream and a
    decoder into the MLP output of this and every later layer (``W_dec``: d_sae x
    (n_layers - layer) x d_model, flattened over the last two axes)."""

    def __init__(self, layer, W_enc, b_enc, W_dec, b_dec, path: str = ""):
        super().__init__()
        self.layer = layer
        self.W_enc = nn.Parameter(W_enc, requires_grad=False)
        self.b_enc = nn.Parameter(b_enc, requires_grad=False)
        self.W_dec = nn.Parameter(W_dec.flatten(1), requires_grad=False)
        self.b_dec = nn.Parameter(b_dec, requires_grad=False)
        self.path = path
        self.delta = None
        self.hook_sae_acts_post = HookPoint()

    @property
    def d_sae(self) -> int:
        return self.W_enc.shape[0]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x.to(self.W_enc.dtype) @ self.W_enc.T + self.b_enc).float()

    def decoder_to(self, target: int) -> torch.Tensor:
        d, j = self.b_dec.shape[0], target - self.layer
        return self.W_dec[:, j * d:(j + 1) * d]

    def read(self, x: torch.Tensor) -> torch.Tensor:
        """Encode, pass the activations through the hook point, and keep the change the hooks
        made (zero with a gradient path when the extractor scores the features)."""
        acts = self.encode(x.detach())
        if torch.is_grad_enabled():
            acts.requires_grad_(True)
        post = self.hook_sae_acts_post(acts)
        self.delta = None if post is acts and not torch.is_grad_enabled() else post - acts.detach()
        return x


def load_clt(release: str, n_layers: int, device: str, dtype=torch.bfloat16) -> Dict[int, CLTLayer]:
    layers = {}
    for layer in range(n_layers):
        enc = load_file(hf_hub_download(release, f"W_enc_{layer}.safetensors"), device=device)
        dec = load_file(hf_hub_download(release, f"W_dec_{layer}.safetensors"), device=device)
        layers[layer] = CLTLayer(layer, enc[f"W_enc_{layer}"].to(dtype), enc[f"b_enc_{layer}"].to(dtype),
                                 dec[f"W_dec_{layer}"].to(dtype), enc[f"b_dec_{layer}"].to(dtype),
                                 path=f"{release}/W_enc_{layer}")
    return layers


def attach_clt(model, spec: SAESpec) -> Dict[int, CLTLayer]:
    """Splice a CLT: encode at ``hook_resid_mid``, add the hooks' changes to the features,
    decoded, to ``hook_mlp_out`` of every layer at or after the features' own."""
    layers = load_clt(spec.release, model.cfg.n_layers, model.cfg.device, getattr(torch, spec.dtype))
    for layer, clt in layers.items():
        model.blocks[layer].add_module("sae", clt)
        model.add_hook(f"blocks.{layer}.hook_resid_mid", lambda x, hook, clt=clt: clt.read(x), is_permanent=True)

        def write(y, hook=None, target=layer):
            deltas = [(src, src.delta) for src in (layers[l] for l in range(target + 1)) if src.delta is not None]
            if not deltas:
                return y
            out = y.float() + sum(d.to(src.W_dec.dtype) @ src.decoder_to(target) for src, d in deltas).float()
            return out.to(y.dtype)
        model.add_hook(f"blocks.{layer}.hook_mlp_out", write, is_permanent=True)
    model.setup()
    model.sae_spec = spec
    model.clt = layers
    return layers


def attach_saes(model, spec: SAESpec, saes: Optional[Dict[int, JumpReLUSAE]] = None) -> Dict[int, JumpReLUSAE]:
    """Splice one SAE per layer into ``blocks.{L}.hook_resid_post`` as a permanent
    hook, and register the feature hook points with the model. A CLT spec is
    spliced by ``attach_clt``."""
    if spec.kind == "clt":
        return attach_clt(model, spec)
    if saes is None:
        saes = load_saes(spec, model.cfg.n_layers, model.cfg.device, getattr(torch, spec.dtype))
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
    parser.add_argument("--sae-dtype", default=SAESpec.dtype, choices=["float32", "bfloat16"],
                        help="width 65k needs bfloat16 to fit on a 32 GB card (29 GiB against 15 GiB)")
    parser.add_argument("--sae-kind", default=SAESpec.kind, choices=["sae", "clt"],
                        help="clt: a cross-layer transcoder in circuit-tracer's format (--sae-release mntss/clt-gemma-2-2b-426k)")


def spec_from_args(args) -> SAESpec:
    layers = tuple(int(x) for x in args.sae_layers.split(",")) if args.sae_layers else None
    kind = getattr(args, "sae_kind", SAESpec.kind)
    dtype = "bfloat16" if kind == "clt" else getattr(args, "sae_dtype", SAESpec.dtype)
    return SAESpec(release=args.sae_release, width=args.sae_width, l0=args.sae_l0, layers=layers, dtype=dtype, kind=kind)


@torch.inference_mode()
def reconstruction_report(model, prompts: Iterable[str]) -> Dict[int, dict]:
    """Per layer: fraction of variance unexplained by the SAE and mean L0, over
    the non-BOS positions of ``prompts``. A basis mismatch shows up as FVU near
    or above 1."""
    if hasattr(model, "clt"):
        return clt_reconstruction_report(model, prompts)
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


@torch.inference_mode()
def clt_reconstruction_report(model, prompts: Iterable[str]) -> Dict[int, dict]:
    """Per layer: fraction of the MLP output's variance the CLT leaves unexplained, and the
    mean L0 of the layer's features, over the non-BOS positions of ``prompts``."""
    clt = model.clt
    n = len(clt)
    stats = {layer: dict(resid=0.0, err=0.0, l0=0.0, n=0) for layer in clt}
    acts, outs = {}, {}
    hooks = [(f"blocks.{l}.hook_resid_mid", lambda x, hook, l=l: acts.__setitem__(l, clt[l].encode(x[:, 1:]))) for l in clt]
    hooks += [(f"blocks.{l}.hook_mlp_out", lambda y, hook, l=l: outs.__setitem__(l, y[:, 1:].float())) for l in clt]
    for prompt in prompts:
        with model.hooks(fwd_hooks=hooks):
            model(model.to_tokens(prompt, prepend_bos=True).to(model.cfg.device))
        for m in range(n):
            recon = sum(acts[l].to(clt[l].W_dec.dtype) @ clt[l].decoder_to(m) for l in range(m + 1)).float() + clt[m].b_dec.float()
            y, s = outs[m], stats[m]
            s["resid"] += float(((y - y.mean(dim=(0, 1))) ** 2).sum())
            s["err"] += float(((y - recon) ** 2).sum())
            s["l0"] += float((acts[m] > 0).sum())
            s["n"] += y.shape[0] * y.shape[1]
    return {layer: dict(fvu=s["err"] / s["resid"], l0=s["l0"] / s["n"], path=clt[layer].path) for layer, s in stats.items()}


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
