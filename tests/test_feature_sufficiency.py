"""Feature-level sufficiency on a tiny random model with a random SAE or CLT spliced in: the splice
and an untouched error hook leave the forward pass unchanged, and with every feature and error term
mean-ablated, a layer's output is the decoded mean features plus the mean error term.

    pytest tests/test_feature_sufficiency.py
"""
import pytest
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from circuit_reuse.circuit_extraction import Component
from circuit_reuse.dataset import Example
from circuit_reuse.evaluate import _build_mean_ablation_hooks, _select_mean, compute_corrupted_means
from circuit_reuse.sae import ERROR_HOOK, CLTLayer, JumpReLUSAE, SAESpec, attach_clt, attach_saes

N_LAYERS, D_MODEL, D_SAE = 2, 8, 6
PROMPTS = ["1 2 3", "4 5 6 7 8", "9 10 11 12"]


def tiny_model():
    torch.manual_seed(0)
    cfg = HookedTransformerConfig(n_layers=N_LAYERS, d_model=D_MODEL, n_ctx=16, d_head=4, n_heads=2, d_mlp=16,
                                  d_vocab=20, act_fn="relu", device="cpu")
    m = HookedTransformer(cfg)
    m.to_tokens = lambda s, prepend_bos=True: torch.tensor([[0] + [int(t) for t in s.split()]])
    return m


def with_sae():
    m = tiny_model()
    g = torch.Generator().manual_seed(1)
    saes = {l: JumpReLUSAE(torch.randn(D_MODEL, D_SAE, generator=g), torch.randn(D_SAE, D_MODEL, generator=g),
                           torch.zeros(D_SAE), torch.zeros(D_MODEL), torch.zeros(D_SAE)) for l in range(N_LAYERS)}
    attach_saes(m, SAESpec(), saes)
    return m


def with_clt():
    m = tiny_model()
    g = torch.Generator().manual_seed(2)
    layers = {l: CLTLayer(l, torch.randn(D_SAE, D_MODEL, generator=g), torch.rand(D_SAE, generator=g),
                          torch.randn(D_SAE, N_LAYERS - l, D_MODEL, generator=g), torch.randn(D_MODEL, generator=g))
              for l in range(N_LAYERS)}
    attach_clt(m, SAESpec(kind="clt", dtype="float32"), layers)
    return m


def means_of(model):
    ds = [Example(prompt=p, target="", corrupted_prompt=p, corrupted_target="") for p in PROMPTS]
    return compute_corrupted_means(model, ds, per_position=True, kinds=("feature", "error"), align="end")


def everything(model):
    return ([Component(l, "feature", i) for l in range(N_LAYERS) for i in range(D_SAE)]
            + [Component(l, "error", 0) for l in range(N_LAYERS)])


def run(model, prompt, hooks=()):
    seen = {}
    capture = [(f"blocks.{l}.hook_resid_{w}", lambda x, hook, k=(l, w): seen.__setitem__(k, x[0].clone()))
               for l in range(N_LAYERS) for w in ("mid", "post")]
    with torch.inference_mode(), model.hooks(fwd_hooks=list(hooks) + capture):
        logits = model(model.to_tokens(prompt))
    return logits, seen


@pytest.mark.parametrize("make", [with_sae, with_clt])
def test_splice_and_untouched_error_hook_leave_the_output_unchanged(make):
    model = make()
    plain = tiny_model()
    identity = [(ERROR_HOOK.format(layer=l), lambda x, hook: x) for l in range(N_LAYERS)]
    for p in PROMPTS:
        clean = plain(plain.to_tokens(p))
        torch.testing.assert_close(run(model, p)[0], clean, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(run(model, p, identity)[0], clean, atol=1e-4, rtol=1e-4)


def test_sae_layer_output_is_mean_features_plus_mean_error():
    model = with_sae()
    means = means_of(model)
    hooks = _build_mean_ablation_hooks(everything(model), means)
    for p in PROMPTS:
        n = len(p.split()) + 1
        _, seen = run(model, p, hooks)
        for l in range(N_LAYERS):
            sae = model.blocks[l].sae
            expected = sae.decode(_select_mean(means, f"sae_acts_L{l}", n)) + _select_mean(means, f"sae_error_L{l}", n)
            torch.testing.assert_close(seen[l, "post"], expected, atol=1e-4, rtol=1e-4)


def test_clt_mlp_output_is_decoded_mean_features_plus_mean_error():
    model = with_clt()
    means = means_of(model)
    hooks = _build_mean_ablation_hooks(everything(model), means)
    clt = model.clt
    for p in PROMPTS:
        n = len(p.split()) + 1
        _, seen = run(model, p, hooks)
        for m in range(N_LAYERS):
            recon = sum(_select_mean(means, f"sae_acts_L{l}", n) @ clt[l].decoder_to(m) for l in range(m + 1)) + clt[m].b_dec
            expected = recon + _select_mean(means, f"sae_error_L{m}", n)
            torch.testing.assert_close(seen[m, "post"] - seen[m, "mid"], expected, atol=1e-4, rtol=1e-4)
