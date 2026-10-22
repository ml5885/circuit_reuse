"""Edge patching and greedy edge selection against exact interventions and the EAP-IG library."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from circuit_reuse.edge_patching import EdgePatcher, column_parents, greedy_order, prune
from circuit_reuse.graph import Graph

REF_ROOT = Path(__file__).resolve().parent.parent / "reference_code" / "EAP-IG" / "src"
CLEAN = torch.tensor([[0, 3, 5, 7, 2, 9]])
CORRUPTED = torch.tensor([[0, 4, 1, 7, 8, 6]])


def tiny_model(n_kv_heads=None, post_norm=False) -> HookedTransformer:
    torch.manual_seed(0)
    cfg = HookedTransformerConfig(n_layers=3, d_model=16, n_ctx=16, d_head=4, n_heads=4, d_mlp=32, d_vocab=12,
                                  act_fn="gelu", normalization_type="RMS" if post_norm else "LN",
                                  positional_embedding_type="rotary", n_key_value_heads=n_kv_heads,
                                  use_normalization_before_and_after=post_norm, device="cpu", dtype=torch.float64)
    model = HookedTransformer(cfg)
    model.cfg.use_split_qkv_input = model.cfg.use_attn_result = model.cfg.use_hook_mlp_in = True
    if post_norm:  # non-trivial post-norm weights, so a wrong norm treatment shows up
        for block in model.blocks:
            block.ln1_post.w.data = torch.rand_like(block.ln1_post.w) + .5
    return model


CONFIGS = [dict(), dict(n_kv_heads=2), dict(post_norm=True), dict(n_kv_heads=2, post_norm=True)]


def patched_logits(model, edges):
    graph = Graph.from_model(model)
    patcher = EdgePatcher(model, graph)
    return patcher.run(CLEAN, patcher.corrupted_outputs(CORRUPTED), torch.as_tensor(edges))


@pytest.mark.parametrize("cfg", CONFIGS)
def test_full_and_empty_circuits(cfg):
    model = tiny_model(**cfg)
    graph = Graph.from_model(model)
    full = np.zeros((graph.n_forward, graph.n_backward), dtype=bool)
    full[np.arange(graph.n_forward)[:, None] < column_parents(graph)[None, :]] = True
    with torch.inference_mode():
        torch.testing.assert_close(patched_logits(model, full), model(CLEAN))
        torch.testing.assert_close(patched_logits(model, np.zeros_like(full)), model(CORRUPTED))


@pytest.mark.parametrize("cfg", CONFIGS)
@pytest.mark.parametrize("node", ["a0.h1", "a1.h3", "m0", "m1"])
def test_cutting_a_node_equals_patching_it(cfg, node):
    model = tiny_model(**cfg)
    graph = Graph.from_model(model)
    edges = np.zeros((graph.n_forward, graph.n_backward), dtype=bool)
    edges[np.arange(graph.n_forward)[:, None] < column_parents(graph)[None, :]] = True
    n = graph.nodes[node]
    edges[graph.forward_index(n)] = False

    cache = {}
    with torch.inference_mode():
        model.run_with_hooks(CORRUPTED, fwd_hooks=[(n.out_hook, lambda a, hook: cache.setdefault("x", a.clone()))])

        def patch(act, hook):
            if node.startswith("a"):
                act[:, :, n.head] = cache["x"][:, :, n.head]
            else:
                act[:] = cache["x"]
            return act
        expected = model.run_with_hooks(CLEAN, fwd_hooks=[(n.out_hook, patch)])
    torch.testing.assert_close(patched_logits(model, edges), expected)


@pytest.fixture(scope="module")
def gpt2():
    model = HookedTransformer.from_pretrained("gpt2", device="cpu")
    model.cfg.use_split_qkv_input = model.cfg.use_attn_result = model.cfg.use_hook_mlp_in = True
    return model


@pytest.fixture(scope="module")
def ref():
    sys.path.insert(0, str(REF_ROOT))
    from eap.evaluate import evaluate_graph
    from eap.graph import Graph as RefGraph
    return RefGraph, evaluate_graph


def test_greedy_and_prune_match_reference(gpt2, ref):
    RefGraph, _ = ref
    graph, ref_graph = Graph.from_model(gpt2), RefGraph.from_model(gpt2)
    scores = torch.randn(graph.n_forward, graph.n_backward, generator=torch.Generator().manual_seed(0))
    ref_graph.scores[:] = scores
    for n in (40, 300):
        order = greedy_order(scores.numpy(), graph, n)
        ours = np.zeros((graph.n_forward, graph.n_backward), dtype=bool)
        ours[order[:, 0], order[:, 1]] = True
        ref_graph.apply_greedy(n, absolute=True, prune=False)
        np.testing.assert_array_equal(ours, ref_graph.in_graph.numpy())
        ref_graph.prune()
        np.testing.assert_array_equal(prune(ours, graph), ref_graph.in_graph.numpy())


def test_edge_patching_matches_reference(gpt2, ref):
    from torch.utils.data import DataLoader
    RefGraph, evaluate_graph = ref
    clean, corrupted = "The cat sat on the mat", "The dog ran on the rug"
    graph, ref_graph = Graph.from_model(gpt2), RefGraph.from_model(gpt2)
    ref_graph.scores[:] = torch.randn(graph.n_forward, graph.n_backward, generator=torch.Generator().manual_seed(1))
    ref_graph.apply_greedy(8000, absolute=True)
    edges = ref_graph.in_graph.clone()

    target = gpt2.to_tokens(clean)[0, -1]
    metric = lambda logits, clean_logits, input_lengths, label: logits[torch.arange(len(logits)), input_lengths - 1].log_softmax(-1)[:, target]
    expected = evaluate_graph(gpt2, ref_graph, DataLoader([(clean, corrupted, 0)], batch_size=1), metric, quiet=True)

    patcher = EdgePatcher(gpt2, graph)
    logits = patcher.run(gpt2.to_tokens(clean), patcher.corrupted_outputs(gpt2.to_tokens(corrupted)), edges)
    ours = logits[0, -1].log_softmax(-1)[target]
    torch.testing.assert_close(ours, expected[0].to(logits.dtype), rtol=1e-4, atol=1e-5)
    with torch.inference_mode():
        ends = [gpt2(gpt2.to_tokens(s))[0, -1].log_softmax(-1)[target] for s in (clean, corrupted)]
    assert min(abs(float(ours - e)) for e in ends) > .01  # 10x the tolerance: the circuit is neither the clean nor the corrupted run


def test_faithful_search_finds_smallest_passing_size():
    from followups.faithful_greedy import THRESHOLD, find_circuit
    model = tiny_model()
    graph = Graph.from_model(model)
    order = greedy_order(np.random.default_rng(0).standard_normal((graph.n_forward, graph.n_backward)), graph,
                         int(column_parents(graph).sum()))

    class Fake:  # faithfulness grows with the number of edges the pruned circuit keeps
        def faithfulness(self, circuits):
            return [float(c.sum()) / len(order) * 1.3 for c in circuits]

    found = find_circuit(Fake(), order, graph, "cpu")
    assert found["faithfulness"] >= THRESHOLD
    below = [f for n, f in found["curve"] if n < found["n_edges"]]
    assert max(below) < THRESHOLD
