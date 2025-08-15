from __future__ import annotations

from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass

import torch
from transformer_lens import HookedTransformer
from .eap_wrapper import EAP
from .dataset import corrupt_prompt_for_eap


@dataclass(frozen=True)
class Component:
    layer: int
    kind: str
    index: int
    
    def __hash__(self) -> int:
        return hash((self.layer, self.kind, self.index))
    
    def __repr__(self) -> str:
        return f"{self.kind}[layer={self.layer}, index={self.index}]"


class CircuitExtractor:
    def __init__(self, model: Any, top_k: Optional[int] = 5) -> None:
        self.model = model
        self.top_k = top_k
        self.model.cfg.use_split_qkv_input = True
        self.model.cfg.use_attn_result = True
        self.model.cfg.use_hook_mlp_in = True

    def _gold_ids(self, prompt: str, target: str, device) -> List[int]:
        prompt_tok = self.model.to_tokens(prompt, prepend_bos=True)
        full_tok = self.model.to_tokens(prompt + target, prepend_bos=True)
        prompt_tok = prompt_tok.to(device)
        full_tok = full_tok.to(device)
        
        p_ids = prompt_tok[0].tolist()
        f_ids = full_tok[0].tolist()
        
        lcp = 0
        for a, b in zip(p_ids, f_ids):
            if a == b:
                lcp += 1
            else:
                break
                
        if lcp == len(f_ids):
            alt = self.model.to_tokens(target, prepend_bos=False).to(device)
            tokens = alt[0].tolist() if alt.ndim == 2 else alt.tolist()
            return [int(x) for x in tokens]
            
        return [int(x) for x in f_ids[lcp:]]

    def _metric_fn(self, prompt: str, target: str, device):
        gold_ids = self._gold_ids(prompt, target, device)
        
        def metric(logits):
            last = logits[0, -1]
            return torch.stack([last[i] for i in gold_ids]).sum()
            
        return metric

    def extract_circuit(
        self, 
        prompt: str, 
        target: str, 
        task: Optional[str] = None
    ) -> Set[Component]:
        device = self.model.cfg.device
        clean_tokens = self.model.to_tokens(prompt, prepend_bos=True).to(device)
        
        corrupt_prompt = corrupt_prompt_for_eap(prompt, task_hint=task)
        corrupted_tokens = self.model.to_tokens(
            corrupt_prompt, prepend_bos=True
        ).to(device)
        if clean_tokens.shape[0] != corrupted_tokens.shape[0]:
            raise ValueError(
                f"Token batch mismatch: clean={tuple(clean_tokens.shape)} corrupted={tuple(corrupted_tokens.shape)}"
            )
        if clean_tokens.shape[1] == 0 or corrupted_tokens.shape[1] == 0:
            raise ValueError(
                f"Empty tokenized sequence encountered. prompt={prompt!r} corrupt={corrupt_prompt!r}"
            )
        
        metric = self._metric_fn(prompt, target, device)
        
        graph = EAP(
            self.model,
            clean_tokens,
            corrupted_tokens,
            metric,
            upstream_nodes=["mlp", "head"],
            downstream_nodes=["mlp", "head"],
            batch_size=1,
        )
        
        scores: Dict[Component, float] = {}
        
        for i, name in enumerate(graph.upstream_nodes):
            if name.startswith("head."):
                parts = name.split(".")
                layer = int(parts[1])
                head = int(parts[2])
                val = float(graph.eap_scores[i, :].abs().sum().item())
                scores[Component(layer, "head", head)] = val
                
            elif name.startswith("mlp."):
                parts = name.split(".")
                layer = int(parts[1])
                val = float(graph.eap_scores[i, :].abs().sum().item())
                scores[Component(layer, "mlp", 0)] = val
                
        items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        if self.top_k is not None:
            items = items[:self.top_k]
            
        return {c for c, _ in items}


def compute_shared_circuit(circuits: List[Set[Component]]) -> Set[Component]:
    if not circuits:
        return set()
        
    shared = set(circuits[0])
    for c in circuits[1:]:
        shared.intersection_update(c)
        
    return shared


__all__ = ["Component", "CircuitExtractor", "compute_shared_circuit"]
