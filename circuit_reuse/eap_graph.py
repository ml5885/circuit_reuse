import os
import numpy as np
import torch
from torch import Tensor
from functools import partial
from jaxtyping import Float
from typing import Dict

DEFAULT_GRAPH_PLOT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "ims",
)


def _kv_head_count(cfg):
    n_kv = getattr(cfg, "n_kv_heads", None)
    if n_kv is None:
        n_kv = getattr(cfg, "n_heads_kv", None)
    if n_kv is None:
        n_kv = getattr(cfg, "n_kv", None)
    if n_kv is None:
        n_kv = getattr(cfg, "num_kv_heads", None)
    if n_kv is None:
        n_kv = getattr(cfg, "n_heads", 0)
    return int(n_kv)


def _q_head_count(cfg):
    n_q = getattr(cfg, "n_heads", None)
    if n_q is None:
        n_q = getattr(cfg, "num_attention_heads", 0)
    return int(n_q)


class EAPGraph:
    def __init__(self, cfg, upstream_nodes=None, downstream_nodes=None, edges=None):
        self.cfg = cfg
        self.valid_upstream_node_types = ["resid_pre", "mlp", "head"]
        self.valid_downstream_node_types = ["resid_post", "mlp", "head"]
        self.valid_upstream_hook_types = [
            "hook_resid_pre",
            "hook_result",
            "hook_mlp_out",
        ]
        self.valid_downstream_hook_types = [
            "hook_q_input",
            "hook_k_input",
            "hook_v_input",
            "hook_mlp_in",
            "hook_resid_post",
        ]
        self.upstream_component_ordering = {
            "hook_resid_pre": 0,
            "hook_result": 1,
            "hook_mlp_out": 2,
        }
        self.downstream_component_ordering = {
            "hook_q_input": 0,
            "hook_k_input": 1,
            "hook_v_input": 2,
            "hook_mlp_in": 3,
            "hook_resid_post": 4,
        }
        self.element_size = torch.empty(
            (0),
            device=self.cfg.device,
            dtype=self.cfg.dtype,
        ).element_size()

        self.upstream_nodes = []
        self.downstream_nodes = []
        self.upstream_node_index: Dict[str, int] = {}
        self.downstream_node_index: Dict[str, int] = {}
        self.upstream_hook_slice: Dict[str, slice] = {}
        self.downstream_hook_slice: Dict[str, slice] = {}
        self.upstream_nodes_before_layer: Dict[int, slice] = {}
        self.upstream_nodes_before_attn_layer: Dict[int, slice] = {}
        self.upstream_nodes_before_mlp_layer: Dict[int, slice] = {}

        if edges is not None:
            upstream_nodes = [edge[0] for edge in edges]
            downstream_nodes = [edge[1] for edge in edges]

        self.setup_graph_from_nodes(upstream_nodes, downstream_nodes)

        self.eap_scores: Float[
            Tensor,
            "n_upstream_nodes n_downstream_nodes",
        ] = None
        self.adj_matrix: Float[
            Tensor,
            "n_upstream_nodes n_downstream_nodes",
        ] = None

    def setup_graph_from_nodes(self, upstream_nodes=None, downstream_nodes=None):
        if upstream_nodes is None:
            upstream_nodes = self.valid_upstream_node_types.copy()
        if downstream_nodes is None:
            downstream_nodes = self.valid_downstream_node_types.copy()

        (
            self.upstream_hooks,
            self.downstream_hooks,
        ) = self.get_hooks_from_nodes(upstream_nodes, downstream_nodes)

        upstream_node_index = 0
        for hook_name in self.upstream_hooks:
            layer = int(hook_name.split(".")[1])
            hook_type = hook_name.split(".")[-1]

            if layer not in self.upstream_nodes_before_layer:
                for earlier_layer in range(0, layer + 1):
                    if earlier_layer not in self.upstream_nodes_before_layer:
                        self.upstream_nodes_before_layer[earlier_layer] = slice(
                            0,
                            upstream_node_index,
                        )
                        self.upstream_nodes_before_attn_layer[layer] = slice(
                            0,
                            upstream_node_index,
                        )
                        self.upstream_nodes_before_mlp_layer[layer] = slice(
                            0,
                            upstream_node_index,
                        )

            if hook_type == "hook_resid_pre":
                self.upstream_nodes.append(f"resid_pre.{layer}")
                self.upstream_node_index[f"resid_pre.{layer}"] = upstream_node_index
                self.upstream_hook_slice[hook_name] = slice(
                    upstream_node_index,
                    upstream_node_index + 1,
                )
                self.upstream_nodes_before_attn_layer[layer] = slice(
                    0,
                    upstream_node_index + 1,
                )
                upstream_node_index += 1

            elif hook_type == "hook_result":
                n_heads_q = _q_head_count(self.cfg)
                for head_idx in range(n_heads_q):
                    self.upstream_nodes.append(f"head.{layer}.{head_idx}")
                    self.upstream_node_index[f"head.{layer}.{head_idx}"] = (
                        upstream_node_index + head_idx
                    )
                self.upstream_hook_slice[hook_name] = slice(
                    upstream_node_index,
                    upstream_node_index + n_heads_q,
                )
                self.upstream_nodes_before_mlp_layer[layer] = slice(
                    0,
                    upstream_node_index + n_heads_q,
                )
                upstream_node_index += n_heads_q

            elif hook_type == "hook_mlp_out":
                self.upstream_nodes.append(f"mlp.{layer}")
                self.upstream_node_index[f"mlp.{layer}"] = upstream_node_index
                self.upstream_hook_slice[hook_name] = slice(
                    upstream_node_index,
                    upstream_node_index + 1,
                )
                upstream_node_index += 1

            else:
                raise ValueError(f"Unexpected upstream hook type: {hook_type} in {hook_name}")


        for layer in range(0, self.cfg.n_layers):
            if layer not in self.upstream_nodes_before_layer:
                self.upstream_nodes_before_layer[layer] = slice(
                    0,
                    upstream_node_index,
                )
                self.upstream_nodes_before_attn_layer[layer] = slice(
                    0,
                    upstream_node_index,
                )
                self.upstream_nodes_before_mlp_layer[layer] = slice(
                    0,
                    upstream_node_index,
                )

        downstream_node_index = 0
        for hook_name in self.downstream_hooks:
            layer = int(hook_name.split(".")[1])
            hook_type = hook_name.split(".")[-1]

            if hook_type == "hook_q_input":
                n_heads = _q_head_count(self.cfg)
                for head_idx in range(n_heads):
                    node = f"head.{layer}.{head_idx}.q"
                    self.downstream_nodes.append(node)
                    self.downstream_node_index[node] = downstream_node_index + head_idx
                self.downstream_hook_slice[hook_name] = slice(
                    downstream_node_index,
                    downstream_node_index + n_heads,
                )
                downstream_node_index += n_heads

            elif hook_type == "hook_k_input":
                n_heads = _kv_head_count(self.cfg)
                for head_idx in range(n_heads):
                    node = f"head.{layer}.{head_idx}.k"
                    self.downstream_nodes.append(node)
                    self.downstream_node_index[node] = downstream_node_index + head_idx
                self.downstream_hook_slice[hook_name] = slice(
                    downstream_node_index,
                    downstream_node_index + n_heads,
                )
                downstream_node_index += n_heads

            elif hook_type == "hook_v_input":
                n_heads = _kv_head_count(self.cfg)
                for head_idx in range(n_heads):
                    node = f"head.{layer}.{head_idx}.v"
                    self.downstream_nodes.append(node)
                    self.downstream_node_index[node] = downstream_node_index + head_idx
                self.downstream_hook_slice[hook_name] = slice(
                    downstream_node_index,
                    downstream_node_index + n_heads,
                )
                downstream_node_index += n_heads

            elif hook_type == "hook_mlp_in":
                node = f"mlp.{layer}"
                self.downstream_nodes.append(node)
                self.downstream_node_index[node] = downstream_node_index
                self.downstream_hook_slice[hook_name] = slice(
                    downstream_node_index,
                    downstream_node_index + 1,
                )
                downstream_node_index += 1

            elif hook_type == "hook_resid_post":
                node = f"resid_post.{layer}"
                self.downstream_nodes.append(node)
                self.downstream_node_index[node] = downstream_node_index
                self.downstream_hook_slice[hook_name] = slice(
                    downstream_node_index,
                    downstream_node_index + 1,
                )
                downstream_node_index += 1

            else:
                raise ValueError(f"Unexpected upstream hook type: {hook_type} in {hook_name}")

        self.n_upstream_nodes = len(self.upstream_nodes)
        self.n_downstream_nodes = len(self.downstream_nodes)

        _ = (
            self.n_upstream_nodes
            * self.cfg.d_model
            * self.element_size
            / 2**30
        )

    def get_hooks_from_nodes(self, upstream_nodes, downstream_nodes):
        for node in upstream_nodes:
            node_type = node.split(".")[0]
            assert node_type in self.valid_upstream_node_types

        for node in downstream_nodes:
            node_type = node.split(".")[0]
            assert node_type in self.valid_downstream_node_types

        upstream_hooks = []
        downstream_hooks = []

        for node in upstream_nodes:
            node_is_layer_specific = len(node.split(".")) > 1
            node_type = node.split(".")[0]
            if not node_is_layer_specific:
                hook_type = (
                    "hook_resid_pre"
                    if node_type == "resid_pre"
                    else "hook_mlp_out"
                    if node_type == "mlp"
                    else "hook_result"
                )
                for layer in range(self.cfg.n_layers):
                    upstream_hooks.append(f"blocks.{layer}.{hook_type}")
            else:
                assert node.split(".")[1].isdigit()
                layer = int(node.split(".")[1])
                hook_type = (
                    "hook_resid_pre"
                    if node_type == "resid_pre"
                    else "hook_mlp_out"
                    if node_type == "mlp"
                    else "hook_result"
                )
                upstream_hooks.append(f"blocks.{layer}.{hook_type}")

        for node in downstream_nodes:
            node_is_layer_specific = len(node.split(".")) > 1
            if not node_is_layer_specific:
                if node == "head":
                    for layer in range(self.cfg.n_layers):
                        for letter in "qkv":
                            downstream_hooks.append(
                                f"blocks.{layer}.hook_{letter}_input"
                            )
                elif node == "resid_post" or node == "mlp":
                    hook_type = (
                        "hook_resid_post" if node == "resid_post" else "hook_mlp_in"
                    )
                    for layer in range(self.cfg.n_layers):
                        downstream_hooks.append(f"blocks.{layer}.{hook_type}")
                else:
                    raise NotImplementedError
            else:
                assert node.split(".")[1].isdigit()
                layer = int(node.split(".")[1])

                if node.startswith("resid_post") or node.startswith("mlp"):
                    hook_type = (
                        "hook_resid_post"
                        if node.startswith("resid_post")
                        else "hook_mlp_in"
                    )
                    downstream_hooks.append(f"blocks.{layer}.{hook_type}")

                elif node.startswith("head"):
                    letters = ["q", "k", "v"]
                    if len(node.split(".")) == 4:
                        letter_specified = node.split(".")[3]
                        assert letter_specified in letters
                        letters = [letter_specified]
                    for letter in letters:
                        downstream_hooks.append(
                            f"blocks.{layer}.hook_{letter}_input"
                        )
                else:
                    raise NotImplementedError

        upstream_hooks = list(set(upstream_hooks))
        downstream_hooks = list(set(downstream_hooks))

        def get_hook_level(hook, component_ordering):
            num_components_per_layer = len(component_ordering)
            layer = int(hook.split(".")[1])
            hook_type = hook.split(".")[-1]
            component_order = component_ordering[hook_type]
            level = layer * num_components_per_layer + component_order
            return level

        get_upstream_hook_level = partial(
            get_hook_level,
            component_ordering=self.upstream_component_ordering,
        )
        get_downstream_hook_level = partial(
            get_hook_level,
            component_ordering=self.downstream_component_ordering,
        )

        upstream_hooks = sorted(upstream_hooks, key=get_upstream_hook_level)
        downstream_hooks = sorted(downstream_hooks, key=get_downstream_hook_level)

        return upstream_hooks, downstream_hooks

    def get_slice_previous_upstream_nodes(self, downstream_hook):
        layer = downstream_hook.layer()
        hook_type = downstream_hook.name.split(".")[-1]
        if hook_type == "hook_mlp_in":
            return self.upstream_nodes_before_mlp_layer[layer]
        elif hook_type in [
            "hook_q_input",
            "hook_k_input",
            "hook_v_input",
            "hook_resid_post",
        ]:
            return self.upstream_nodes_before_layer[layer]

    def get_hook_slice(self, hook_name):
        if hook_name in self.upstream_hook_slice:
            return self.upstream_hook_slice[hook_name]
        elif hook_name in self.downstream_hook_slice:
            return self.downstream_hook_slice[hook_name]

    def reset_scores(self):
        self.eap_scores = torch.zeros(
            (self.n_upstream_nodes, self.n_downstream_nodes),
            device=self.cfg.device,
        )

    def top_edges(self, n=1000, threshold=None, abs_scores=True):
        assert self.eap_scores is not None

        if abs_scores:
            top_scores, top_indices = torch.topk(
                self.eap_scores.flatten().abs(),
                k=n,
                dim=0,
            )
        else:
            top_scores, top_indices = torch.topk(
                self.eap_scores.flatten(),
                k=n,
                dim=0,
            )

        top_edges = []
        for i, (abs_score, index) in enumerate(zip(top_scores, top_indices)):
            if threshold is not None and abs_score < threshold:
                break

            upstream_node_idx, downstream_node_idx = np.unravel_index(
                index,
                self.eap_scores.shape,
            )
            score = self.eap_scores[upstream_node_idx, downstream_node_idx]
            top_edges.append(
                (
                    self.upstream_nodes[upstream_node_idx],
                    self.downstream_nodes[downstream_node_idx],
                    score.item(),
                )
            )

        return top_edges
