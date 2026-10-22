"""Per-position corrupted means on a tiny random model, for prompts of different lengths: with
align="end" the last position gets the mean over every prompt's last token and the first
position the mean over BOS tokens; with align="start" positions are counted from the start.

    pytest tests/test_mean_alignment.py
"""
import pytest
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from circuit_reuse.dataset import Example
from circuit_reuse.evaluate import _select_mean, compute_corrupted_means

PROMPTS = ["1 2 3", "4 5 6 7 8", "9 10 11 12"]
KEY = "mlp_out_L1"


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = HookedTransformerConfig(n_layers=2, d_model=8, n_ctx=16, d_head=4, n_heads=2, d_mlp=16,
                                  d_vocab=20, act_fn="relu", device="cpu")
    m = HookedTransformer(cfg)
    m.to_tokens = lambda s, prepend_bos=True: torch.tensor([[0] + [int(t) for t in s.split()]])
    return m


def activations(model, prompt):
    _, cache = model.run_with_cache(model.to_tokens(prompt))
    return cache["blocks.1.hook_mlp_out"][0]


@pytest.fixture(scope="module")
def acts(model):
    return [activations(model, p) for p in PROMPTS]


def means(model, align):
    ds = [Example(prompt=p, target="", corrupted_prompt=p, corrupted_target="") for p in PROMPTS]
    return compute_corrupted_means(model, ds, per_position=True, kinds=("mlp",), align=align)


@pytest.mark.parametrize("n", [4, 6, 5])
def test_end_alignment_averages_last_tokens_and_bos(model, acts, n):
    m = _select_mean(means(model, "end"), KEY, n)
    assert m.shape[0] == n
    torch.testing.assert_close(m[-1], torch.stack([a[-1] for a in acts]).mean(0))
    torch.testing.assert_close(m[-2], torch.stack([a[-2] for a in acts]).mean(0))
    torch.testing.assert_close(m[0], torch.stack([a[0] for a in acts]).mean(0))


def test_start_alignment_counts_from_the_first_token(model, acts):
    m = _select_mean(means(model, "start"), KEY, 4)
    torch.testing.assert_close(m[3], torch.stack([a[3] for a in acts]).mean(0))
    torch.testing.assert_close(m[1], torch.stack([a[1] for a in acts]).mean(0))


def test_positions_beyond_the_longest_prompt_get_the_pooled_mean(model, acts):
    m = means(model, "end")
    sel = _select_mean(m, KEY, 9)
    torch.testing.assert_close(sel[1], m["_pooled"][KEY])
    torch.testing.assert_close(sel[-1], torch.stack([a[-1] for a in acts]).mean(0))
