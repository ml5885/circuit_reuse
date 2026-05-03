"""Layer-wise Relevance Propagation rules for TransformerLens.

Ports the four propagation rules from the RelP paper (Jafari et al. 2025,
https://arxiv.org/abs/2508.21258) onto our installed TransformerLens by
class-level monkey-patching. Matches the semantics of the RelP fork
(reference_code/RelP/TransformerLens) without forking the dependency.

Rules:
- LN-rule (Ali et al. 2022): RMSNorm / LayerNorm normalize by a detached scale,
  so the backward pass is linear through the norm.
- AH-rule (Ali et al. 2022): the softmaxed attention pattern is detached,
  so gradient flows only through the OV circuit.
- Half-rule (Arras et al. 2019; Jafari et al. 2024): the gate*up elementwise
  multiply in gated MLPs passes half the gradient to each branch (Shapley).
- Identity-rule: pass-through on the activation function itself — no code
  change needed in TL because the Half-rule already takes care of the
  multiplicative interaction, and act_fn's gradient is the identity component
  of the LRP decomposition for SiLU/GELU.
"""
from __future__ import annotations

from typing import Iterable, List, Optional

import torch
from transformer_lens import HookedTransformer
from transformer_lens.components import AbstractAttention
from transformer_lens.components.layer_norm import LayerNorm
from transformer_lens.components.layer_norm_pre import LayerNormPre
from transformer_lens.components.mlps.can_be_used_as_mlp import CanBeUsedAsMLP
from transformer_lens.components.mlps.gated_mlp import GatedMLP
from transformer_lens.components.rms_norm import RMSNorm
from transformer_lens.components.rms_norm_pre import RMSNormPre
from transformer_lens.utilities.addmm import batch_addmm

DEFAULT_LRP_RULES: List[str] = ["LN-rule", "AH-rule", "Half-rule"]

_PATCHED = False
_ORIGINALS: dict = {}


def _stabilize(z: torch.Tensor) -> torch.Tensor:
    return z + ((z == 0.0).to(z) + z.sign()) * 1e-6


class ModifiedAct(torch.nn.Module):
    """Reference RelP activation wrapper: same forward, LRP-style backward."""

    def __init__(self, act):
        super().__init__()
        self.act = act
        self.modified_act = torch.nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.act(x)
        if isinstance(self.act, torch.nn.ReLU):
            return z
        zp = _stabilize(self.modified_act(x))
        return zp * (z / zp).detach()


def _rule_active(cfg, rule: str) -> bool:
    return getattr(cfg, "use_lrp", False) and rule in getattr(cfg, "LRP_rules", [])


def _patched_rms_norm_forward(self, x):
    if self.cfg.dtype not in [torch.float32, torch.float64]:
        x = x.to(torch.float32)
    scale = self.hook_scale((x.pow(2).mean(-1, keepdim=True) + self.eps).sqrt())
    denom = scale.detach() if _rule_active(self.cfg, "LN-rule") else scale
    x = self.hook_normalized(x / denom).to(self.cfg.dtype)
    if x.device != self.w.device:
        self.to(x.device)
    return x * self.w


def _patched_layer_norm_forward(self, x):
    if self.cfg.dtype not in [torch.float32, torch.float64]:
        x = x.to(torch.float32)
    x = x - x.mean(-1, keepdim=True)
    scale = self.hook_scale((x.pow(2).mean(-1, keepdim=True) + self.eps).sqrt())
    denom = scale.detach() if _rule_active(self.cfg, "LN-rule") else scale
    x = x / denom
    return self.hook_normalized(x * self.w + self.b).to(self.cfg.dtype)


def _patched_layer_norm_pre_forward(self, x):
    if self.cfg.dtype not in [torch.float32, torch.float64]:
        x = x.to(torch.float32)

    x = x - x.mean(-1, keepdim=True)
    scale = self.hook_scale((x.pow(2).mean(-1, keepdim=True) + self.eps).sqrt())
    denom = scale.detach() if _rule_active(self.cfg, "LN-rule") else scale
    return self.hook_normalized(x / denom).to(self.cfg.dtype)


def _patched_rms_norm_pre_forward(self, x):
    if self.cfg.dtype not in [torch.float32, torch.float64]:
        x = x.to(torch.float32)

    scale = self.hook_scale((x.pow(2).mean(-1, keepdim=True) + self.eps).sqrt())
    denom = scale.detach() if _rule_active(self.cfg, "LN-rule") else scale
    return self.hook_normalized(x / denom).to(self.cfg.dtype)


def _patched_gated_mlp_forward(self, x):
    if self.W_gate.device != x.device:
        x = x.to(self.W_gate.device)
    pre_act = self.hook_pre(torch.matmul(x, self.W_gate))

    if (
        self.cfg.is_layer_norm_activation()
        and self.hook_mid is not None
        and self.ln is not None
    ):
        mid_act = self.hook_mid(self.act_fn(pre_act))
        post_act = self.hook_post(self.ln(mid_act))
    else:
        pre_linear = self.hook_pre_linear(torch.matmul(x, self.W_in))
        gated = self.act_fn(pre_act) * pre_linear
        if _rule_active(self.cfg, "Half-rule"):
            gated = gated / 2.0 + (gated / 2.0).detach()
        post_act = self.hook_post(gated + self.b_in)

    return batch_addmm(self.b_out, self.W_out, post_act)


def _patched_calculate_z_scores(self, v, pattern):
    """Detaches the attention pattern before OV multiplication under AH-rule.

    Covers both AbstractAttention and GroupedQueryAttention: GQA's override
    does its repeat_interleave and then calls `super().calculate_z_scores`,
    which resolves to this patched method.
    """
    if _rule_active(self.cfg, "AH-rule"):
        pattern = pattern.detach()
    return _ORIGINALS["AbstractAttention.calculate_z_scores"](self, v, pattern)


def install_lrp_patches() -> None:
    """Install LRP-aware forwards on TL component classes. Idempotent."""
    global _PATCHED
    if _PATCHED:
        return
    _ORIGINALS["RMSNorm.forward"] = RMSNorm.forward
    _ORIGINALS["LayerNorm.forward"] = LayerNorm.forward
    _ORIGINALS["RMSNormPre.forward"] = RMSNormPre.forward
    _ORIGINALS["LayerNormPre.forward"] = LayerNormPre.forward
    _ORIGINALS["CanBeUsedAsMLP.select_activation_function"] = CanBeUsedAsMLP.select_activation_function
    _ORIGINALS["GatedMLP.forward"] = GatedMLP.forward
    _ORIGINALS["AbstractAttention.calculate_z_scores"] = AbstractAttention.calculate_z_scores

    RMSNorm.forward = _patched_rms_norm_forward
    LayerNorm.forward = _patched_layer_norm_forward
    RMSNormPre.forward = _patched_rms_norm_pre_forward
    LayerNormPre.forward = _patched_layer_norm_pre_forward
    GatedMLP.forward = _patched_gated_mlp_forward
    AbstractAttention.calculate_z_scores = _patched_calculate_z_scores

    def _patched_select_activation_function(self):
        _ORIGINALS["CanBeUsedAsMLP.select_activation_function"](self)
        if _rule_active(self.cfg, "Identity-rule"):
            self.act_fn = ModifiedAct(self.act_fn)

    CanBeUsedAsMLP.select_activation_function = _patched_select_activation_function
    _PATCHED = True


def uninstall_lrp_patches() -> None:
    global _PATCHED
    if not _PATCHED:
        return
    RMSNorm.forward = _ORIGINALS["RMSNorm.forward"]
    LayerNorm.forward = _ORIGINALS["LayerNorm.forward"]
    RMSNormPre.forward = _ORIGINALS["RMSNormPre.forward"]
    LayerNormPre.forward = _ORIGINALS["LayerNormPre.forward"]
    CanBeUsedAsMLP.select_activation_function = _ORIGINALS["CanBeUsedAsMLP.select_activation_function"]
    GatedMLP.forward = _ORIGINALS["GatedMLP.forward"]
    AbstractAttention.calculate_z_scores = _ORIGINALS["AbstractAttention.calculate_z_scores"]
    _ORIGINALS.clear()
    _PATCHED = False


def _wrap_identity_rule_modules(model: HookedTransformer) -> None:
    for module in model.modules():
        if not isinstance(module, CanBeUsedAsMLP):
            continue
        if not hasattr(module, "act_fn"):
            continue
        if isinstance(module.act_fn, ModifiedAct):
            continue
        module._lrp_original_act_fn = module.act_fn
        module.act_fn = ModifiedAct(module.act_fn)


def _unwrap_identity_rule_modules(model: HookedTransformer) -> None:
    for module in model.modules():
        if hasattr(module, "_lrp_original_act_fn"):
            module._modules.pop("act_fn", None)
            object.__setattr__(module, "act_fn", module._lrp_original_act_fn)
            delattr(module, "_lrp_original_act_fn")


def enable_lrp(model: HookedTransformer, rules: Optional[Iterable[str]] = None) -> None:
    """Turn on LRP backward rules on `model`. Installs class patches on first call."""
    install_lrp_patches()
    chosen_rules = list(rules) if rules is not None else list(DEFAULT_LRP_RULES)
    model.cfg.use_lrp = True
    model.cfg.LRP_rules = chosen_rules
    if "Identity-rule" in chosen_rules:
        _wrap_identity_rule_modules(model)
    else:
        _unwrap_identity_rule_modules(model)


def disable_lrp(model: HookedTransformer) -> None:
    _unwrap_identity_rule_modules(model)
    model.cfg.use_lrp = False


__all__ = [
    "DEFAULT_LRP_RULES",
    "enable_lrp",
    "disable_lrp",
    "install_lrp_patches",
    "uninstall_lrp_patches",
]
