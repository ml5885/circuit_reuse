import os
import numpy as np
import torch
from torch import Tensor
from functools import partial
from jaxtyping import Float
from typing import Dict

DEFAULT_GRAPH_PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ims")

class EAPGraph:
    def __init__(self, cfg, upstream_nodes=None, downstream_nodes=None, edges=None):
        self.cfg = cfg
        self.valid_upstream_node_types = ["resid_pre", "mlp", "head"]
        self.valid_downstream_node_types = ["resid_post", "mlp", "head"]
        self.valid_upstream_hook_types = [
            "hook_resid_pre", 
            "hook_result", 
            "hook_mlp_out"
        ]
        self.valid_downstream_hook_types = [
            "hook_q_input", 
            "hook_k_input", 
            "hook_v_input", 
            "hook_mlp_in", 
            "hook_resid_post"
        ]
        
        self.upstream_component_ordering = {
            "hook_resid_pre": 0, 
            "hook_result": 1, 
            "hook_mlp_out": 2
        }
        
        self.downstream_component_ordering = {
            "hook_q_input": 0, 
            "hook_k_input": 1, 
            "hook_v_input": 2, 
            "hook_mlp_in": 3, 
            "hook_resid_post": 4
        }
        
        self.element_size = torch.empty(
            (0), 
            device=self.cfg.device,
            dtype=self.cfg.dtype
        ).element_size()
        
        self.upstream_nodes = []
        self.downstream_nodes = []
        
        # Index mappings
        self.upstream_node_index: Dict[str, int] = {}
        self.downstream_node_index: Dict[str, int] = {}
        self.upstream_hook_slice: Dict[str, slice] = {}
        self.downstream_hook_slice: Dict[str, slice] = {}
        
        # Layer-based node groupings
        self.upstream_nodes_before_layer: Dict[int, slice] = {}
        self.upstream_nodes_before_attn_layer: Dict[int, slice] = {}
        self.upstream_nodes_before_mlp_layer: Dict[int, slice] = {}
        
        # Handle edge-based initialization
        if edges is not None:
            upstream_nodes = [edge[0] for edge in edges]
            downstream_nodes = [edge[1] for edge in edges]
            
        self.setup_graph_from_nodes(upstream_nodes, downstream_nodes)
        
        # Initialize score matrices
        self.eap_scores: Float[Tensor, "n_upstream_nodes n_downstream_nodes"] = None
        self.adj_matrix: Float[Tensor, "n_upstream_nodes n_downstream_nodes"] = None

    def setup_graph_from_nodes(self, upstream_nodes=None, downstream_nodes=None):
        if upstream_nodes is None:
            upstream_nodes = self.valid_upstream_node_types.copy()
        if downstream_nodes is None:
            downstream_nodes = self.valid_downstream_node_types.copy()
            
        self.upstream_hooks, self.downstream_hooks = self.get_hooks_from_nodes(
            upstream_nodes, downstream_nodes
        )
        
        # Build upstream nodes
        upstream_node_index = 0
        for hook_name in self.upstream_hooks:
            parts = hook_name.split(".")
            if len(parts) < 3 or not parts[1].isdigit():
                raise ValueError(f"Malformed upstream hook name: {hook_name}")
            layer = int(parts[1])
            hook_type = parts[-1]
            
            # Initialize layer slices if not already present
            if layer not in self.upstream_nodes_before_layer:
                for earlier_layer in range(0, layer + 1):
                    if earlier_layer not in self.upstream_nodes_before_layer:
                        self.upstream_nodes_before_layer[earlier_layer] = slice(
                            0, upstream_node_index
                        )
                        self.upstream_nodes_before_attn_layer[layer] = slice(
                            0, upstream_node_index
                        )
                        self.upstream_nodes_before_mlp_layer[layer] = slice(
                            0, upstream_node_index
                        )
            
            # Handle different hook types
            if hook_type == "hook_resid_pre":
                self.upstream_nodes.append(f"resid_pre.{layer}")
                self.upstream_node_index[f"resid_pre.{layer}"] = upstream_node_index
                self.upstream_hook_slice[hook_name] = slice(
                    upstream_node_index, upstream_node_index + 1
                )
                self.upstream_nodes_before_attn_layer[layer] = slice(
                    0, upstream_node_index + 1
                )
                upstream_node_index += 1
                
            elif hook_type == "hook_result":
                for head_idx in range(self.cfg.n_heads):
                    self.upstream_nodes.append(f"head.{layer}.{head_idx}")
                    self.upstream_node_index[f"head.{layer}.{head_idx}"] = (
                        upstream_node_index + head_idx
                    )
                    
                self.upstream_hook_slice[hook_name] = slice(
                    upstream_node_index, upstream_node_index + self.cfg.n_heads
                )
                self.upstream_nodes_before_mlp_layer[layer] = slice(
                    0, upstream_node_index + self.cfg.n_heads
                )
                upstream_node_index += self.cfg.n_heads
                
            elif hook_type == "hook_mlp_out":
                self.upstream_nodes.append(f"mlp.{layer}")
                self.upstream_node_index[f"mlp.{layer}"] = upstream_node_index
                self.upstream_hook_slice[hook_name] = slice(
                    upstream_node_index, upstream_node_index + 1
                )
                upstream_node_index += 1
            else:
                raise ValueError(f"Unsupported upstream hook type: {hook_type} in {hook_name}")
                
        # Fill in any remaining layer slices
        for layer in range(0, self.cfg.n_layers):
            if layer not in self.upstream_nodes_before_layer:
                self.upstream_nodes_before_layer[layer] = slice(
                    0, upstream_node_index
                )
                self.upstream_nodes_before_attn_layer[layer] = slice(
                    0, upstream_node_index
                )
                self.upstream_nodes_before_mlp_layer[layer] = slice(
                    0, upstream_node_index
                )
        
        # Build downstream nodes
        downstream_node_index = 0
        for hook_name in self.downstream_hooks:
            parts = hook_name.split(".")
            if len(parts) < 3 or not parts[1].isdigit():
                raise ValueError(f"Malformed downstream hook name: {hook_name}")
            layer = int(parts[1])
            hook_type = parts[-1]
            
            if hook_type in ["hook_q_input", "hook_k_input", "hook_v_input"]:
                letter = hook_type.split("_")[1].lower()
                for head_idx in range(self.cfg.n_heads):
                    node_name = f"head.{layer}.{head_idx}.{letter}"
                    self.downstream_nodes.append(node_name)
                    self.downstream_node_index[node_name] = (
                        downstream_node_index + head_idx
                    )
                    
                self.downstream_hook_slice[hook_name] = slice(
                    downstream_node_index, downstream_node_index + self.cfg.n_heads
                )
                downstream_node_index += self.cfg.n_heads
                
            elif hook_type == "hook_mlp_in":
                self.downstream_nodes.append(f"mlp.{layer}")
                self.downstream_node_index[f"mlp.{layer}"] = downstream_node_index
                self.downstream_hook_slice[hook_name] = slice(
                    downstream_node_index, downstream_node_index + 1
                )
                downstream_node_index += 1
                
            elif hook_type == "hook_resid_post":
                self.downstream_nodes.append(f"resid_post.{layer}")
                self.downstream_node_index[f"resid_post.{layer}"] = (
                    downstream_node_index
                )
                self.downstream_hook_slice[hook_name] = slice(
                    downstream_node_index, downstream_node_index + 1
                )
                downstream_node_index += 1
            else:
                raise ValueError(f"Unsupported downstream hook type: {hook_type} in {hook_name}")
                
        self.n_upstream_nodes = len(self.upstream_nodes)
        self.n_downstream_nodes = len(self.downstream_nodes)
        _ = self.n_upstream_nodes * self.cfg.d_model * self.element_size / 2**30

    def get_hooks_from_nodes(self, upstream_nodes, downstream_nodes):
        # Validate node types
        for node in upstream_nodes:
            node_type = node.split(".")[0]
            if node_type not in self.valid_upstream_node_types:
                raise ValueError(f"Invalid upstream node type '{node_type}' from '{node}'. Valid: {self.valid_upstream_node_types}")
            
        for node in downstream_nodes:
            node_type = node.split(".")[0]
            if node_type not in self.valid_downstream_node_types:
                raise ValueError(f"Invalid downstream node type '{node_type}' from '{node}'. Valid: {self.valid_downstream_node_types}")
        
        upstream_hooks = []
        downstream_hooks = []
        
        # Process upstream nodes
        for node in upstream_nodes:
            node_is_layer_specific = (len(node.split(".")) > 1)
            node_type = node.split(".")[0]
            
            if not node_is_layer_specific:
                if node_type == "resid_pre":
                    hook_type = "hook_resid_pre"
                elif node_type == "mlp":
                    hook_type = "hook_mlp_out"
                else:
                    hook_type = "attn.hook_result"
                    
                for layer in range(self.cfg.n_layers):
                    upstream_hooks.append(f"blocks.{layer}.{hook_type}")
            else:
                if not node.split(".")[1].isdigit():
                    raise ValueError(f"Layer not specified as int in upstream node '{node}'")
                layer = int(node.split(".")[1])
                
                if node_type == "resid_pre":
                    hook_type = "hook_resid_pre"
                elif node_type == "mlp":
                    hook_type = "hook_mlp_out"
                else:
                    hook_type = "attn.hook_result"
                    
                upstream_hooks.append(f"blocks.{layer}.{hook_type}")
        
        # Process downstream nodes
        for node in downstream_nodes:
            node_is_layer_specific = (len(node.split(".")) > 1)
            
            if not node_is_layer_specific:
                if node == "head":
                    for layer in range(self.cfg.n_layers):
                        for letter in "qkv":
                            downstream_hooks.append(f"blocks.{layer}.hook_{letter}_input")
                            
                elif node == "resid_post" or node == "mlp":
                    hook_type = "hook_resid_post" if node == "resid_post" else "hook_mlp_in"
                    for layer in range(self.cfg.n_layers):
                        downstream_hooks.append(f"blocks.{layer}.{hook_type}")
                else:
                    raise NotImplementedError
                    
            else:
                if not node.split(".")[1].isdigit():
                    raise ValueError(f"Layer not specified as int in downstream node '{node}'")
                layer = int(node.split(".")[1])
                
                if node.startswith("resid_post") or node.startswith("mlp"):
                    hook_type = "hook_resid_post" if node.startswith("resid_post") else "hook_mlp_in"
                    downstream_hooks.append(f"blocks.{layer}.{hook_type}")
                    
                elif node.startswith("head"):
                    all_heads = len(node.split(".")) <= 2
                    head_idx = None if all_heads else int(node.split(".")[2])
                    letters = ["q", "k", "v"]
                    
                    if len(node.split(".")) == 4:
                        letter_specified = node.split(".")[3]
                        if letter_specified not in letters:
                            raise ValueError(f"Invalid head letter '{letter_specified}' in '{node}', expected one of {letters}")
                        letters = [letter_specified]
                        
                    for letter in letters:
                        downstream_hooks.append(f"blocks.{layer}.hook_{letter}_input")
                else:
                    raise NotImplementedError
        
        # Remove duplicates and sort
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
            get_hook_level, component_ordering=self.upstream_component_ordering
        )
        get_downstream_hook_level = partial(
            get_hook_level, component_ordering=self.downstream_component_ordering
        )
        
        upstream_hooks = sorted(upstream_hooks, key=get_upstream_hook_level)
        downstream_hooks = sorted(downstream_hooks, key=get_downstream_hook_level)
        
        return upstream_hooks, downstream_hooks

    def get_slice_previous_upstream_nodes(self, downstream_hook):
        # Support both HookPoint objects and string hook names
        hook_name = getattr(downstream_hook, "name", downstream_hook)
        parts = str(hook_name).split(".")
        if len(parts) < 3 or not parts[1].isdigit():
            raise ValueError(f"Malformed downstream hook name: {hook_name}")
        layer = int(parts[1])
        hook_type = parts[-1]
        
        if hook_type == "hook_mlp_in":
            return self.upstream_nodes_before_mlp_layer[layer]
        elif hook_type in ["hook_q_input", "hook_k_input", "hook_v_input", "hook_resid_post"]:
            return self.upstream_nodes_before_layer[layer]
        else:
            raise ValueError(f"Unknown downstream hook type for previous slice: {hook_type}")

    def get_hook_slice(self, hook_name):
        if hook_name in self.upstream_hook_slice:
            return self.upstream_hook_slice[hook_name]
        elif hook_name in self.downstream_hook_slice:
            return self.downstream_hook_slice[hook_name]
        else:
            raise KeyError(f"Hook slice not found for '{hook_name}'. Known upstream keys: {list(self.upstream_hook_slice.keys())[:5]}... downstream keys: {list(self.downstream_hook_slice.keys())[:5]}...")

    def reset_scores(self):
        self.eap_scores = torch.zeros(
            (self.n_upstream_nodes, self.n_downstream_nodes), 
            device=self.cfg.device
        )

    def top_edges(self, n=1000, threshold=None, abs_scores=True):
        assert self.eap_scores is not None
        
        if abs_scores:
            top_scores, top_indices = torch.topk(
                self.eap_scores.flatten().abs(), k=n, dim=0
            )
        else:
            top_scores, top_indices = torch.topk(
                self.eap_scores.flatten(), k=n, dim=0
            )
        
        top_edges = []
        for i, (abs_score, index) in enumerate(zip(top_scores, top_indices)):
            if threshold is not None and abs_score < threshold:
                break
                
            upstream_node_idx, downstream_node_idx = np.unravel_index(
                index, self.eap_scores.shape
            )
            score = self.eap_scores[upstream_node_idx, downstream_node_idx]
            
            top_edges.append((
                self.upstream_nodes[upstream_node_idx], 
                self.downstream_nodes[downstream_node_idx], 
                score.item()
            ))
            
        return top_edges
