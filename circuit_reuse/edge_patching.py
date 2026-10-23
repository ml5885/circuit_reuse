"""Edge-level circuits on the EAP graph, as in Hanna et al. (2024): greedy edge selection from the
logits, pruning, and edge patching, where every edge outside the circuit carries its parent's output
on the corrupted input and every edge inside it the parent's output in the patched run.

Edge patching mirrors ``evaluate_graph`` in the EAP-IG library on our score-matrix layout, including
Gemma-2, whose attention block normalizes the summed head outputs (``ln1_post``), so a head's
contribution to a later node is ln1_post of the patched sum minus ln1_post of the patched-run sum.
"""
from __future__ import annotations

import heapq
import numpy as np
import torch
from einops import einsum
from transformer_lens import HookedTransformer

from .graph import AttentionNode, Graph, InputNode, LogitNode, MLPNode


def column_parents(graph: Graph) -> np.ndarray:
    """For every backward column, the number of forward nodes that can feed it (they come first)."""
    prev = np.zeros(graph.n_backward, dtype=int)
    for node in graph.nodes.values():
        if isinstance(node, AttentionNode):
            for letter in "qkv":
                prev[graph.backward_index(node, qkv=letter)] = graph.prev_index(node)
        elif isinstance(node, (MLPNode, LogitNode)):
            prev[graph.backward_index(node)] = graph.prev_index(node)
    return prev


def layout(names: dict) -> dict:
    """The score matrix's layout from its saved row and column names (Graph.index_names): the nodes
    each column feeds and how many forward nodes can feed each column."""
    forward = {n: i for i, n in enumerate(names["forward"])}
    prev = [len(forward) if col == "logits" else forward[col.split("<")[0].split(".")[0] + ".h0"] if col.startswith("a")
            else forward[col] for col in names["backward"]]
    return {"children": names["backward_children"], "prev": np.array(prev), "n_forward": len(forward)}


def greedy_order(scores: np.ndarray, lay: dict, n_max: int) -> np.ndarray:
    """The first ``n_max`` (forward, backward) edges a greedy search from the logits picks: at each step,
    the edge with the largest |score| among the edges into the current circuit, whose parent then
    joins the circuit and opens its own input columns (EAP-IG's apply_greedy, absolute scores).
    ``lay`` comes from ``layout``."""
    mag = np.abs(scores)
    n_fwd, n_bwd = mag.shape
    prev = lay["prev"]
    feeds = [[] for _ in range(n_fwd)]
    for b, kids in enumerate(lay["children"]):
        for f in kids:
            feeds[f].append(b)
    heap, opened, in_circuit, order = [], np.zeros(n_bwd, dtype=bool), np.zeros(n_fwd, dtype=bool), []

    def open_column(b: int):
        opened[b] = True
        for f in range(prev[b]):
            heapq.heappush(heap, (-mag[f, b], f, b))

    open_column(n_bwd - 1)
    while heap and len(order) < n_max:
        _, f, b = heapq.heappop(heap)
        order.append((f, b))
        if not in_circuit[f]:
            in_circuit[f] = True
            for col in feeds[f]:
                if not opened[col]:
                    open_column(col)
    return np.array(order, dtype=np.int64).reshape(-1, 2)


def prune(edges: np.ndarray, lay: dict) -> np.ndarray:
    """Drop nodes without an incoming or an outgoing circuit edge, and their edges, until nothing
    changes. Such edges carry corrupted values either way, so pruning never changes the patched run."""
    edges = edges.copy()
    children = lay["children"]
    n_fwd, n_bwd = edges.shape
    feeds = np.zeros((n_fwd, n_bwd), dtype=bool)
    for b, kids in enumerate(children):
        feeds[kids, b] = True
    while True:
        has_out = edges.any(1)
        has_in = (edges.any(0)[None, :] & feeds).any(1)
        has_in[0] = True  # the input node
        nodes = has_out & has_in
        alive_col = np.array([b == n_bwd - 1 or nodes[kids].any() for b, kids in enumerate(children)])
        pruned = edges & nodes[:, None] & alive_col[None, :]
        if (pruned == edges).all():
            return edges
        edges = pruned


def circuit_nodes(edges: np.ndarray, graph: Graph) -> list:
    """Heads and MLP blocks with an outgoing edge in a pruned circuit."""
    return [graph.idx_to_forward_node[f] for f in np.flatnonzero(edges.any(1))
            if not isinstance(graph.idx_to_forward_node[f], InputNode)]


class EdgePatcher:
    """Runs the model with every edge outside a circuit patched to its corrupted value."""

    def __init__(self, model: HookedTransformer, graph: Graph):
        self.model, self.graph = model, graph
        self.post_norm = bool(model.cfg.use_normalization_before_and_after)
        cfg = graph.cfg
        self.n_heads, self.n_kv = cfg["n_heads"], cfg["n_kv_heads"]
        self.head_layer = torch.full((graph.n_forward,), -1, dtype=torch.long, device=model.cfg.device)
        self.out_hooks = []  # (hook name, forward index of its first node, head?)
        for node in graph.nodes.values():
            if isinstance(node, InputNode):
                self.out_hooks.append((node.out_hook, 0, False))
            elif isinstance(node, MLPNode):
                self.out_hooks.append((node.out_hook, graph.forward_index(node), False))
            elif isinstance(node, AttentionNode):
                self.head_layer[graph.forward_index(node)] = node.layer
                if node.head == 0:
                    self.out_hooks.append((node.out_hook, graph.forward_index(node), True))
        self.in_hooks = []  # (hook name, prev index, columns, per-head grouping)
        n_layers = cfg["n_layers"]
        for layer in range(n_layers):
            first = graph.nodes[f"a{layer}.h0"]
            prev = graph.prev_index(first)
            for letter in "qkv":
                n = self.n_heads if letter == "q" else self.n_kv
                start = graph.backward_index(first, qkv=letter)
                self.in_hooks.append((f"blocks.{layer}.hook_{letter}_input", prev, list(range(start, start + n))))
            mlp = graph.nodes[f"m{layer}"]
            self.in_hooks.append((mlp.in_hook, graph.prev_index(mlp), [graph.backward_index(mlp)]))
        logits = graph.nodes["logits"]
        self.in_hooks.append((logits.in_hook, graph.n_forward, [graph.n_backward - 1]))

    def _store_hook(self, buf: torch.Tensor, index: int, head: bool):
        """Writes a node's output into ``buf`` (batch, pos, n_forward, d_model)."""
        def hook(act, hook=None):
            n = act.shape[1]
            if head:
                buf[:, :n, index:index + self.n_heads] = act
            else:
                buf[:, :n, index] = act
        return hook

    @torch.inference_mode()
    def corrupted_outputs(self, tokens: torch.Tensor, attention_mask=None) -> torch.Tensor:
        """(1, pos, n_forward, d_model) outputs of every forward node on the corrupted input."""
        buf = torch.zeros((1, tokens.shape[1], self.graph.n_forward, self.model.cfg.d_model),
                          device=self.model.cfg.device, dtype=self.model.cfg.dtype)
        hooks = [(name, self._store_hook(buf, i, head)) for name, i, head in self.out_hooks]
        with self.model.hooks(fwd_hooks=hooks):
            self.model(tokens, attention_mask=attention_mask)
        return buf

    def _input_hook(self, cur, corr, patch, prev: int, cols: list):
        """Adds, at a node's input, the corrupted-minus-current outputs of the parents whose edge into
        it is outside the circuit, for every circuit in the batch."""
        mask = patch[:, :prev, cols]  # (batch, prev, n_cols)
        is_head = self.head_layer[:prev] >= 0

        def masked(x, m, f):  # sum over parents f of x[..., f, :] * m[:, f, c]
            return einsum(x[0, :, f], m[:, f], "pos f d, b f c -> b pos c d") if len(x) == 1 \
                else einsum(x[:, :, f], m[:, f], "b pos f d, b f c -> b pos c d")

        def hook(act, hook=None):
            n = act.shape[1]
            c, k = corr[:, :n, :prev], cur[:, :n, :prev]
            if self.post_norm:
                update = masked(c, mask, ~is_head) - masked(k, mask, ~is_head)
                for layer in self.head_layer[:prev][is_head].unique().tolist():
                    heads = torch.nonzero(self.head_layer[:prev] == layer).flatten()
                    if not mask[:, heads].any():
                        continue
                    block = self.model.blocks[layer]
                    clean = k[:, :, heads].sum(2) + block.attn.b_O
                    patched = clean[:, :, None] + masked(c, mask, heads) - masked(k, mask, heads)
                    update = update + block.ln1_post(patched) - block.ln1_post(clean)[:, :, None]
            else:
                every = slice(None)
                update = masked(c, mask, every) - masked(k, mask, every)
            if act.ndim == 4:
                if act.shape[2] == len(cols):
                    act += update
                else:  # K/V inputs ungrouped to one per head: each head reads its KV group's column
                    act += update.repeat_interleave(act.shape[2] // len(cols), dim=2)
            else:
                act += update[:, :, 0]
            return act
        return hook

    @torch.inference_mode()
    def run(self, tokens: torch.Tensor, corrupted: torch.Tensor, edges: torch.Tensor, attention_mask=None) -> torch.Tensor:
        """Logits on the clean input, one row per circuit in ``edges`` ((batch,) n_forward, n_backward
        bool), with every edge outside the circuit patched to the corrupted outputs from
        ``corrupted_outputs``."""
        edges = edges if edges.ndim == 3 else edges[None]
        batch = len(edges)
        cur = torch.zeros((batch,) + corrupted.shape[1:], device=corrupted.device, dtype=corrupted.dtype)
        patch = (~edges).to(corrupted.dtype)
        hooks = [(name, self._store_hook(cur, i, head)) for name, i, head in self.out_hooks]
        hooks += [(name, self._input_hook(cur, corrupted, patch, prev, cols)) for name, prev, cols in self.in_hooks]
        mask = None if attention_mask is None else attention_mask.expand(batch, -1)
        with self.model.hooks(fwd_hooks=hooks):
            return self.model(tokens.expand(batch, -1), attention_mask=mask)
