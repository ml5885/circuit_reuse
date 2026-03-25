from __future__ import annotations
from typing import Dict, List, Set, Any, Optional, Tuple
from dataclasses import dataclass
import torch
from torch.nn import functional as F
from transformer_lens import HookedTransformer
from contextlib import nullcontext
import gc
from collections import defaultdict

from .graph import Graph, Granularity, attribute_single_example, make_hooks
from .dataset import Example


@dataclass(frozen=True)
class Component:
    layer: int
    kind: str  # "head", "mlp", "attn_block", or "neuron"
    index: int

    def __hash__(self) -> int:
        return hash((self.layer, self.kind, self.index))

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.kind}[layer={self.layer}, index={self.index}]"


class CircuitExtractor:
    """
    Extract per-example attribution scores for components.
    """
    def __init__(self, model: HookedTransformer, method: str = "eap", granularity: Granularity = "head_mlp") -> None:
        self.model = model
        self.method = method
        self.granularity = granularity
        if method == "neuron_attr":
            self.graph = None
        else:
            self.graph = Graph.from_model(model, granularity=granularity)
        # Enable hooks needed for all methods
        self.model.cfg.use_split_qkv_input = True
        self.model.cfg.use_attn_result = True
        self.model.cfg.use_hook_mlp_in = True

    def _get_metric_fn(self, positions: torch.Tensor, target_ids: torch.Tensor):
        def metric(logits: torch.Tensor, corrupted_logits: torch.Tensor, input_lengths: torch.Tensor, label: Any) -> torch.Tensor:
            logprobs = logits.log_softmax(dim=-1)
            selected = logprobs[0, positions, :].gather(dim=1, index=target_ids.view(-1, 1))
            return selected.sum()
        return metric

    def _scores_to_components(self, scores: torch.Tensor) -> Dict[Component, float]:
        """Convert raw scores to a mapping from Component to score."""
        from .graph import InputNode, MLPNode, AttentionNode, AttentionBlockNode
        component_scores: Dict[Component, float] = {}
        per_component_scores = scores.abs().sum(dim=1)

        for fwd_idx, score in enumerate(per_component_scores.tolist()):
            node = self.graph.idx_to_forward_node.get(fwd_idx)
            if node is None or isinstance(node, InputNode):
                continue

            if isinstance(node, AttentionBlockNode):
                comp = Component(layer=node.layer, kind="attn_block", index=0)
            elif isinstance(node, AttentionNode):
                comp = Component(layer=node.layer, kind="head", index=node.head)
            elif isinstance(node, MLPNode):
                comp = Component(layer=node.layer, kind="mlp", index=0)
            else:
                continue
            component_scores[comp] = float(score)
        return component_scores

    def _prepare_eap_inputs(self, example: Example):
        prompt_tok = self.model.to_tokens(example.prompt, prepend_bos=True)
        clean_full_tok = self.model.to_tokens(example.prompt + example.target, prepend_bos=True)
        corrupted_full_tok = self.model.to_tokens(example.corrupted_prompt + example.corrupted_target, prepend_bos=True)

        device = self.model.cfg.device
        p_ids, f_ids = prompt_tok.tolist()[0], clean_full_tok.tolist()[0]
        lcp = 0
        while lcp < len(p_ids) and lcp < len(f_ids) and p_ids[lcp] == f_ids[lcp]:
            lcp += 1
        gold_ids_list = f_ids[lcp:] if lcp < len(f_ids) else self.model.to_tokens(example.target, prepend_bos=False).tolist()[0]
        target_ids = torch.tensor(gold_ids_list, device=device, dtype=torch.long)
        prompt_len = prompt_tok.shape[1]
        positions = torch.arange(prompt_len - 1, prompt_len - 1 + len(gold_ids_list), device=device, dtype=torch.long)
        max_len = max(clean_full_tok.shape[1], corrupted_full_tok.shape[1])
        pad_token = self.model.tokenizer.pad_token_id if self.model.tokenizer.pad_token_id is not None else self.model.tokenizer.eos_token_id

        clean_tokens = F.pad(clean_full_tok, (0, max_len - clean_full_tok.shape[1]), "constant", pad_token).to(device)
        corrupted_tokens = F.pad(corrupted_full_tok, (0, max_len - corrupted_full_tok.shape[1]), "constant", pad_token).to(device)

        metric = self._get_metric_fn(positions=positions, target_ids=target_ids)
        return clean_tokens, corrupted_tokens, metric, max_len

    def _prepare_gradient_inputs(self, example: Example):
        prompt_tok = self.model.to_tokens(example.prompt, prepend_bos=True)
        clean_full_tok = self.model.to_tokens(example.prompt + example.target, prepend_bos=True)

        device = self.model.cfg.device
        p_ids, f_ids = prompt_tok.tolist()[0], clean_full_tok.tolist()[0]
        lcp = 0
        while lcp < len(p_ids) and lcp < len(f_ids) and p_ids[lcp] == f_ids[lcp]:
            lcp += 1

        gold_ids_list = f_ids[lcp:] if lcp < len(f_ids) else self.model.to_tokens(example.target, prepend_bos=False).tolist()[0]
        target_ids = torch.tensor(gold_ids_list, device=device, dtype=torch.long)
        prompt_len = prompt_tok.shape[1]
        positions = torch.arange(prompt_len - 1, prompt_len - 1 + len(gold_ids_list), device=device, dtype=torch.long)
        metric = self._get_metric_fn(positions=positions, target_ids=target_ids)
        return clean_full_tok.to(device), metric, clean_full_tok.shape[1]

    def _extract_neuron_attr(self, example: Example, device: str, autocast_ctx) -> Dict[Component, float]:
        """Node-level attribution: activation_diff × gradient for neurons and heads.

        Following Arora et al. ("Language Model Circuits Are Sparse in the Neuron Basis"),
        scores each MLP neuron and attention head by the dot product of the
        activation difference (corrupted - clean) and the gradient of the metric
        w.r.t. that activation, summed over sequence positions.
        """
        cfg = self.model.cfg
        n_layers, n_heads, d_mlp = cfg.n_layers, cfg.n_heads, cfg.d_mlp

        # Tokenize
        clean_full = self.model.to_tokens(example.prompt + example.target, prepend_bos=True)
        corrupted_full = self.model.to_tokens(example.corrupted_prompt + example.corrupted_target, prepend_bos=True)
        prompt_tok = self.model.to_tokens(example.prompt, prepend_bos=True)

        p_ids, f_ids = prompt_tok.tolist()[0], clean_full.tolist()[0]
        lcp = 0
        for a, b in zip(p_ids, f_ids):
            if a == b:
                lcp += 1
            else:
                break
        gold_ids_list = f_ids[lcp:] if lcp < len(f_ids) else self.model.to_tokens(example.target, prepend_bos=False).tolist()[0]
        prompt_len = prompt_tok.shape[1]
        positions = torch.arange(prompt_len - 1, prompt_len - 1 + len(gold_ids_list), device=device, dtype=torch.long)
        target_ids = torch.tensor(gold_ids_list, device=device, dtype=torch.long)
        metric_fn = self._get_metric_fn(positions=positions, target_ids=target_ids)

        pad_id = self.model.tokenizer.pad_token_id if self.model.tokenizer.pad_token_id is not None else self.model.tokenizer.eos_token_id
        max_len = max(clean_full.shape[1], corrupted_full.shape[1])
        clean_tokens = F.pad(clean_full, (0, max_len - clean_full.shape[1]), "constant", pad_id).to(device)
        corrupted_tokens = F.pad(corrupted_full, (0, max_len - corrupted_full.shape[1]), "constant", pad_id).to(device)

        # Step 1: corrupted forward pass — save activations
        corrupted_acts: Dict[str, torch.Tensor] = {}

        def save_corrupted(name):
            def hook(act, hook=None):
                corrupted_acts[name] = act.detach()
                return act
            return hook

        corrupted_hooks = []
        for layer in range(n_layers):
            corrupted_hooks.append((f"blocks.{layer}.mlp.hook_post", save_corrupted(f"mlp.{layer}")))
            corrupted_hooks.append((f"blocks.{layer}.attn.hook_result", save_corrupted(f"attn.{layer}")))

        with torch.inference_mode():
            with self.model.hooks(fwd_hooks=corrupted_hooks):
                self.model(corrupted_tokens)
        self.model.reset_hooks()

        # Step 2+3: clean forward + backward — compute activation_diff and capture gradients
        clean_acts: Dict[str, torch.Tensor] = {}
        gradients: Dict[str, torch.Tensor] = {}

        def save_clean_and_grad(name):
            def fwd_hook(act, hook=None):
                act.retain_grad()
                clean_acts[name] = act
                return act
            return fwd_hook

        def save_grad(name):
            def bwd_hook(act, hook=None):
                gradients[name] = act.detach()
                return act
            return bwd_hook

        fwd_hooks = []
        bwd_hooks = []
        for layer in range(n_layers):
            fwd_hooks.append((f"blocks.{layer}.mlp.hook_post", save_clean_and_grad(f"mlp.{layer}")))
            fwd_hooks.append((f"blocks.{layer}.attn.hook_result", save_clean_and_grad(f"attn.{layer}")))
            bwd_hooks.append((f"blocks.{layer}.mlp.hook_post", save_grad(f"mlp.{layer}")))
            bwd_hooks.append((f"blocks.{layer}.attn.hook_result", save_grad(f"attn.{layer}")))

        with autocast_ctx:
            with self.model.hooks(fwd_hooks=fwd_hooks, bwd_hooks=bwd_hooks):
                logits = self.model(clean_tokens)
                loss = metric_fn(logits, None, None, None)
                loss.backward()

        self.model.zero_grad(set_to_none=True)
        self.model.reset_hooks()

        # Step 4+5: compute per-component scores
        component_scores: Dict[Component, float] = {}
        for layer in range(n_layers):
            mlp_key = f"mlp.{layer}"
            if mlp_key in clean_acts and mlp_key in corrupted_acts and mlp_key in gradients:
                act_diff = corrupted_acts[mlp_key] - clean_acts[mlp_key].detach()  # [batch, seq, d_mlp]
                grad = gradients[mlp_key]  # [batch, seq, d_mlp]
                # Per-neuron: sum over batch and positions
                neuron_scores = (act_diff * grad).sum(dim=(0, 1))  # [d_mlp]
                for i in range(d_mlp):
                    component_scores[Component(layer=layer, kind="neuron", index=i)] = float(neuron_scores[i].abs().item())

            attn_key = f"attn.{layer}"
            if attn_key in clean_acts and attn_key in corrupted_acts and attn_key in gradients:
                act_diff = corrupted_acts[attn_key] - clean_acts[attn_key].detach()  # [batch, seq, n_heads, d_head]
                grad = gradients[attn_key]
                # Per-head: sum over batch, positions, and d_head
                head_scores = (act_diff * grad).sum(dim=(0, 1, 3))  # [n_heads]
                for h in range(n_heads):
                    component_scores[Component(layer=layer, kind="head", index=h)] = float(head_scores[h].abs().item())

        return component_scores

    def extract_circuits_from_examples(
        self, examples: List[Example], task_name: str, amp: bool, device: str
    ) -> Tuple[List[Set[Component]], List[Dict[Component, float]]]:
        autocast_ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                        if amp and device.startswith("cuda") else nullcontext())

        circuits: List[Set[Component]] = []
        per_example_scores: List[Dict[Component, float]] = []

        if self.method == "neuron_attr":
            n_skipped = 0
            for idx, ex in enumerate(examples):
                try:
                    component_scores = self._extract_neuron_attr(ex, device, autocast_ctx)
                except torch.cuda.OutOfMemoryError:
                    n_skipped += 1
                    self.model.zero_grad(set_to_none=True)
                    self.model.reset_hooks()
                    gc.collect()
                    torch.cuda.empty_cache()
                    print(f"[OOM] Skipping example {idx}")
                    continue

                per_example_scores.append(component_scores)
                items = sorted(component_scores.items(), key=lambda x: x[1], reverse=True)
                circuits.append({c for c, _ in items})

                if (idx + 1) % 10 == 0 or (idx + 1) == len(examples):
                    print(f"[{task_name}] (neuron_attr) {idx + 1}/{len(examples)} examples processed")

                torch.cuda.empty_cache()

            if n_skipped:
                print(f"[WARN] {n_skipped}/{len(examples)} examples skipped due to OOM")
            return circuits, per_example_scores

        # Lazily-allocated work buffer — sized to current example, re-allocated
        # only when a longer example is seen.  This avoids a single outlier
        # sequence forcing an allocation that OOMs on memory-constrained GPUs.
        work_buf: torch.Tensor | None = None
        work_buf_seq_len = 0

        def _get_work_buf(seq_len: int) -> torch.Tensor:
            nonlocal work_buf, work_buf_seq_len
            if work_buf is None or seq_len > work_buf_seq_len:
                del work_buf
                torch.cuda.empty_cache()
                work_buf = torch.zeros(
                    (1, seq_len, self.graph.n_forward, self.model.cfg.d_model),
                    device=self.model.cfg.device, dtype=self.model.cfg.dtype,
                )
                work_buf_seq_len = seq_len
            return work_buf

        # Stream examples to GPU one by one
        n_skipped = 0
        for idx, ex in enumerate(examples):
          try:
            if self.method == "eap":
                prompt_tok = self.model.to_tokens(ex.prompt, prepend_bos=True)
                clean_full = self.model.to_tokens(ex.prompt + ex.target, prepend_bos=True)
                corrupted_full = self.model.to_tokens(ex.corrupted_prompt + ex.corrupted_target, prepend_bos=True)

                p_ids, f_ids = prompt_tok.tolist()[0], clean_full.tolist()[0]
                lcp = 0
                for a, b in zip(p_ids, f_ids):
                    if a == b: lcp += 1
                    else: break
                gold_ids_list = f_ids[lcp:] if lcp < len(f_ids) else self.model.to_tokens(ex.target, prepend_bos=False).tolist()[0]
                prompt_len = prompt_tok.shape[1]
                pos_cpu = torch.arange(prompt_len - 1, prompt_len - 1 + len(gold_ids_list), dtype=torch.long)

                pad_id = self.model.tokenizer.pad_token_id if self.model.tokenizer.pad_token_id is not None else self.model.tokenizer.eos_token_id
                ex_len = max(clean_full.shape[1], corrupted_full.shape[1])
                clean_pad = F.pad(clean_full, (0, ex_len - clean_full.shape[1]), "constant", pad_id)
                corrupt_pad = F.pad(corrupted_full, (0, ex_len - corrupted_full.shape[1]), "constant", pad_id)

                clean_tokens = clean_pad.to(device)
                corrupted_tokens = corrupt_pad.to(device)
                positions = pos_cpu.to(device)
                target_ids = torch.tensor(gold_ids_list, device=device, dtype=torch.long)
                metric = self._get_metric_fn(positions=positions, target_ids=target_ids)

                with autocast_ctx:
                    scores = attribute_single_example(
                        model=self.model, graph=self.graph, metric=metric,
                        clean_tokens=clean_tokens, corrupted_tokens=corrupted_tokens,
                        activation_difference=_get_work_buf(ex_len),
                    )

            else:  # gradient method
                clean_full = self.model.to_tokens(ex.prompt + ex.target, prepend_bos=True)
                prompt_tok = self.model.to_tokens(ex.prompt, prepend_bos=True)

                p_ids, f_ids = prompt_tok.tolist()[0], clean_full.tolist()[0]
                lcp = 0
                for a, b in zip(p_ids, f_ids):
                    if a == b: lcp += 1
                    else: break
                gold_ids_list = f_ids[lcp:] if lcp < len(f_ids) else self.model.to_tokens(ex.target, prepend_bos=False).tolist()[0]
                prompt_len = prompt_tok.shape[1]
                positions = torch.arange(prompt_len - 1, prompt_len - 1 + len(gold_ids_list), dtype=torch.long)
                target_ids = torch.tensor(gold_ids_list, dtype=torch.long)

                # per-example reusable tensors on GPU
                seq_len = int(clean_full.shape[1])
                scores = torch.zeros((self.graph.n_forward, self.graph.n_backward),
                                    device=self.model.cfg.device, dtype=self.model.cfg.dtype)
                buf = _get_work_buf(seq_len)
                buf.zero_()

                with autocast_ctx:
                    _, fwd_hooks_clean, bwd_hooks = make_hooks(self.model, self.graph, buf, scores)
                    metric = self._get_metric_fn(positions=positions.to(device), target_ids=target_ids.to(device))
                    with self.model.hooks(fwd_hooks=fwd_hooks_clean, bwd_hooks=bwd_hooks):
                        logits = self.model(clean_full.to(device))
                        metric(logits, None, None, None).backward()
                    self.model.zero_grad(set_to_none=True)
                    self.model.reset_hooks()
                    scores = scores.cpu()

          except torch.cuda.OutOfMemoryError:
            n_skipped += 1
            self.model.zero_grad(set_to_none=True)
            self.model.reset_hooks()
            # Shrink the work_buf back so the failed size doesn't persist
            work_buf = None
            work_buf_seq_len = 0
            gc.collect()
            torch.cuda.empty_cache()
            print(f"[OOM] Skipping example {idx} (seq_len too large for available VRAM)")
            continue

          component_scores = self._scores_to_components(scores)
          per_example_scores.append(component_scores)
          items = sorted(component_scores.items(), key=lambda x: x[1], reverse=True)
          comp_set = {c for c, _ in items}
          circuits.append(comp_set)

          if (idx + 1) % 10 == 0 or (idx + 1) == len(examples):
              print(f"[{task_name}] ({self.method}) {idx + 1}/{len(examples)} examples processed (last circuit size={len(comp_set)})")

          if self.method == "eap":
              del clean_tokens, corrupted_tokens, positions, target_ids, scores
          else:
              del clean_full, prompt_tok, positions, target_ids, scores
          torch.cuda.empty_cache()

        if n_skipped:
            print(f"[WARN] {n_skipped}/{len(examples)} examples skipped due to OOM")

        del work_buf
        gc.collect()
        torch.cuda.empty_cache()
        return circuits, per_example_scores

__all__ = ["Component", "CircuitExtractor"]
