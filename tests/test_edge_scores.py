"""Saved EAP-IG edge scores determine the node scores: summing |edges| over a node's outgoing edges
gives the cached component score, and the signed sum gives the direct node attribution (--node-scores)."""
import json
from contextlib import nullcontext

import numpy as np
import pytest
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from circuit_reuse.circuit_extraction import CircuitExtractor
from circuit_reuse.graph import attribute_single_example_ig
from main_experiment import _save_edge_scores

CLEAN = torch.tensor([[0, 3, 5, 7, 2, 9]])
CORRUPTED = torch.tensor([[0, 4, 1, 7, 8, 6]])
STEPS = 5


def tiny_model(n_kv_heads: int | None, post_norm: bool = False) -> HookedTransformer:
    """A small random model; ``post_norm`` gives it Gemma-2's normalization of the summed head outputs,
    with non-trivial norm weights."""
    torch.manual_seed(0)
    cfg = HookedTransformerConfig(n_layers=3, d_model=16, n_ctx=16, d_head=4, n_heads=4, d_mlp=32, d_vocab=12,
                                  act_fn="gelu", normalization_type="RMS" if post_norm else "LN",
                                  positional_embedding_type="rotary", n_key_value_heads=n_kv_heads,
                                  use_normalization_before_and_after=post_norm, device="cpu", dtype=torch.float64)
    model = HookedTransformer(cfg)
    if post_norm:
        for block in model.blocks:
            block.ln1_post.w.data = torch.rand_like(block.ln1_post.w) + .5
    return model


def metric(logits, *args):
    return logits[0, -1].log_softmax(-1)[3]


def paired_inputs(_example):
    return CLEAN, CORRUPTED, None, None, metric, CLEAN.shape[1]


def edge_extractor(model, keep_edges=False) -> CircuitExtractor:
    ex = CircuitExtractor(model, method="eap_ig", granularity="head_mlp", ig_steps=STEPS, keep_edges=keep_edges)
    ex._prepare_paired_inputs = paired_inputs
    return ex


def edge_scores(model) -> tuple[CircuitExtractor, torch.Tensor]:
    ex = edge_extractor(model)
    buf = torch.zeros((1, CLEAN.shape[1], ex.graph.n_forward, model.cfg.d_model), dtype=model.cfg.dtype)
    return ex, attribute_single_example_ig(model, ex.graph, metric, CLEAN, CORRUPTED, buf, steps=STEPS)


@pytest.mark.parametrize("post_norm", [False, True])
@pytest.mark.parametrize("n_kv_heads", [None, 2])
def test_signed_edge_sum_is_node_attribution(n_kv_heads, post_norm):
    model = tiny_model(n_kv_heads, post_norm)
    ex, scores = edge_scores(model)
    node_ex = CircuitExtractor(model, method="eap_ig", granularity="head_mlp", ig_steps=STEPS, node_scores=True)
    node_ex._prepare_paired_inputs = paired_inputs
    node = node_ex._extract_eap_ig_neuron_example(None, nullcontext(), node_ex._hook_targets())
    # The edge graph takes corrupted - clean activations and the node path clean - corrupted.
    from_edges = (-scores.sum(1)).tolist()
    by_name = {n.name: from_edges[i] for i, n in ex.graph.idx_to_forward_node.items()}
    for comp, value in node.items():
        name = f"a{comp.layer}.h{comp.index}" if comp.kind == "head" else f"m{comp.layer}"
        assert by_name[name] == pytest.approx(value, rel=1e-6, abs=1e-9), name


@pytest.mark.parametrize("n_kv_heads", [None, 2])
def test_saved_edges_reconstruct_cached_scores(tmp_path, n_kv_heads):
    model = tiny_model(n_kv_heads)
    ex = edge_extractor(model, keep_edges=True)
    _, cached = ex._extract_edge_graph([None, None], "toy", "cpu", nullcontext())
    path = tmp_path / "toy.edges.npy"
    _save_edge_scores(path, ex)

    edges = np.load(path)
    names = json.loads(path.with_suffix(".json").read_text())
    assert edges.shape == (2, ex.graph.n_forward, ex.graph.n_backward) and edges.dtype == np.float16
    assert len(names["forward"]) == edges.shape[1] and len(names["backward"]) == edges.shape[2]
    assert all(names["backward"]) and names["backward"][-1] == "logits"
    node_sum = np.abs(edges.astype(np.float64)).sum(2)
    for i, comps in enumerate(cached):
        for comp, value in comps.items():
            name = f"a{comp.layer}.h{comp.index}" if comp.kind == "head" else f"m{comp.layer}"
            assert node_sum[i, names["forward"].index(name)] == pytest.approx(value, rel=2e-3, abs=1e-4), name


def test_kv_columns_feed_their_group():
    model = tiny_model(2)
    names = edge_extractor(model).graph.index_names()
    k_cols = [b for b, n in enumerate(names["backward"]) if n.startswith("a0.") and n.endswith("<k")]
    assert len(k_cols) == 2
    assert [len(names["backward_children"][b]) for b in k_cols] == [2, 2]


def reference_greedy(scores: np.ndarray, children: list[list[int]]) -> list[int]:
    """Edge-by-edge greedy as in the EAP-IG library's apply_greedy (absolute scores), recording the
    order in which parents first enter."""
    import heapq
    n_fwd, n_bwd = scores.shape
    heap = [(-abs(scores[f, n_bwd - 1]), f, n_bwd - 1) for f in range(n_fwd)]
    heapq.heapify(heap)
    in_graph, opened, order = set(), {n_bwd - 1}, []
    while len(order) < n_fwd:
        _, f, _ = heapq.heappop(heap)
        if f in in_graph:
            continue
        in_graph.add(f)
        order.append(f)
        for b, kids in enumerate(children):
            if f in kids and b not in opened:
                opened.add(b)
                for p in range(n_fwd):
                    heapq.heappush(heap, (-abs(scores[p, b]), p, b))
    return order


def test_greedy_order_matches_edge_greedy():
    from followups.edge_rules import greedy_order
    model = tiny_model(2)
    children = edge_extractor(model).graph.index_names()["backward_children"]
    _, scores = edge_scores(model)
    rng = np.random.default_rng(0)
    for s in [scores.numpy(), *(rng.standard_normal(scores.shape) for _ in range(5))]:
        assert greedy_order(s, children).tolist() == reference_greedy(s, children)


def test_rule_caches_load_in_main_experiment(tmp_path):
    from followups.edge_rules import write_caches
    from main_experiment import _load_cached_attributions
    model = tiny_model(2)
    ex = edge_extractor(model, keep_edges=True)
    _, cached = ex._extract_edge_graph([None, None, None], "toy", "cpu", nullcontext())
    _save_edge_scores(tmp_path / "toy.edges.npy", ex)
    write_caches(tmp_path / "toy.edges.npy", tmp_path / "rules")
    loaded = _load_cached_attributions(tmp_path / "rules" / "abs_sum" / "toy.jsonl")
    assert [set(a) for a in loaded] == [set(c) for c in cached]
    for a, c in zip(loaded, cached):
        assert all(a[k] == pytest.approx(c[k], rel=2e-3, abs=1e-4) for k in c)
    greedy = _load_cached_attributions(tmp_path / "rules" / "greedy" / "toy.jsonl")
    ranks = set(greedy[0].values())  # one rank of 1..n_forward belongs to the input node
    assert len(ranks) == len(greedy[0]) and ranks < set(range(1, len(greedy[0]) + 2))


def test_check_passes_on_matching_caches(tmp_path):
    from followups.edge_rules import check, write_caches
    model = tiny_model(2)
    ex = edge_extractor(model, keep_edges=True)
    ex._extract_edge_graph([None, None, None], "toy", "cpu", nullcontext())
    _save_edge_scores(tmp_path / "toy.edges.npy", ex)
    write_caches(tmp_path / "toy.edges.npy", tmp_path / "rules")
    assert check(tmp_path / "toy.edges.npy", tmp_path / "rules" / "signed_sum" / "toy.jsonl",
                 tmp_path / "rules" / "abs_sum" / "toy.jsonl")
    assert not check(tmp_path / "toy.edges.npy", tmp_path / "rules" / "greedy" / "toy.jsonl",
                     tmp_path / "rules" / "abs_sum" / "toy.jsonl", min_rho=0.99)
