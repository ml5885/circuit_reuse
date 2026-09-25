"""The CLT splice on a small random CLT, without model weights: the unablated forward pass is
unchanged, and ablating features (hooks edit the activations in place, as the evaluator's do)
adds their decoded change to this and every later MLP output.

    pytest tests/test_clt.py
"""
import torch

from circuit_reuse.sae import CLTLayer

N_LAYERS, D_SAE, D_MODEL = 3, 8, 4


def make_clt(seed=0):
    g = torch.Generator().manual_seed(seed)
    return {l: CLTLayer(l, torch.randn(D_SAE, D_MODEL, generator=g), torch.rand(D_SAE, generator=g),
                        torch.randn(D_SAE, N_LAYERS - l, D_MODEL, generator=g), torch.zeros(D_MODEL))
            for l in range(N_LAYERS)}


def run(clt, xs, ys):
    """What attach_clt's hooks do: read at every resid_mid, then write into every mlp_out."""
    outs = []
    for layer in range(N_LAYERS):
        clt[layer].read(xs[layer])
        deltas = [(clt[l], clt[l].delta) for l in range(layer + 1) if clt[l].delta is not None]
        outs.append(ys[layer] + sum(d @ src.decoder_to(layer) for src, d in deltas) if deltas else ys[layer])
    return outs


def test_unablated_forward_is_unchanged():
    clt = make_clt()
    xs, ys = torch.randn(N_LAYERS, 1, 5, D_MODEL), torch.randn(N_LAYERS, 1, 5, D_MODEL)
    with torch.inference_mode():
        for out, y in zip(run(clt, xs, ys), ys):
            assert torch.equal(out, y)


def test_in_place_ablation_reaches_every_later_layer():
    clt = make_clt()
    xs, ys = torch.randn(N_LAYERS, 1, 5, D_MODEL), torch.randn(N_LAYERS, 1, 5, D_MODEL)

    def zero_all(act, hook=None):
        act[:] = 0.0
        return act

    clt[0].hook_sae_acts_post.add_hook(zero_all)
    with torch.inference_mode():
        outs = run(clt, xs, ys)
    acts = clt[0].encode(xs[0])
    for layer in range(N_LAYERS):
        expected = ys[layer] - acts @ clt[0].decoder_to(layer)
        assert torch.allclose(outs[layer], expected, atol=1e-5)
