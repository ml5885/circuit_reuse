from __future__ import annotations
from typing import Dict, List, Set, Any, Optional, Tuple, Callable, Iterable
from dataclasses import dataclass
import gc

import torch
from torch.nn import functional as F
from transformer_lens import HookedTransformer
from contextlib import nullcontext

from .graph import Graph, Granularity, attribute_single_example, attribute_single_example_ig
from .dataset import Example
from .lrp_patch import DEFAULT_LRP_RULES, enable_lrp, disable_lrp


@dataclass(frozen=True)
class Component:
    layer: int
    kind: str  # "head", "mlp", or "neuron"
    index: int

    def __hash__(self) -> int:
        return hash((self.layer, self.kind, self.index))

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.kind}[layer={self.layer}, index={self.index}]"


HookTarget = Tuple[str, Callable[[torch.Tensor], Dict[Component, float]]]


class CircuitExtractor:
    """
    Extract per-example attribution scores for components.

    Supported methods (`eap` and `eap_ig` use edge-level EAP machinery, `relp` is node-level):
    - ``eap``: Edge Attribution Patching (Syed et al. 2023). Score =
      ``(corrupted_act − clean_act) × clean_grad`` on edges, then aggregated per node.
    - ``eap_ig``: EAP with integrated gradients over the clean↔corrupted input
      embedding path (Hanna et al. 2024). Uses the same edge graph as ``eap``
      but averages gradients over ``ig_steps`` interpolation points.
    - ``relp``: Relevance Patching (Jafari et al. 2025, arXiv:2508.21258) and
      neuron-basis attribution (Arora et al. 2026, arXiv:2601.22594). Score =
      ``corrupted_grad × (clean_act − corrupted_act)`` at a single hook per component,
      with LRP-modified backward rules (``LN-rule``, ``AH-rule``, ``Half-rule``).
      Works at all granularities; at ``neuron`` granularity it scores MLP neurons
      only, matching Arora et al.'s neuron-basis circuits.
    """

    def __init__(
        self,
        model: HookedTransformer,
        method: str = "eap",
        granularity: Granularity = "head_mlp",
        task_metric: str = "logprob",
        use_lrp: Optional[bool] = None,
        lrp_rules: Optional[Iterable[str]] = None,
        ig_steps: int = 5,
    ) -> None:
        self.model = model
        self.method = method
        self.granularity = granularity
        self.task_metric = task_metric
        self.use_lrp = bool(method == "relp") if use_lrp is None else bool(use_lrp)
        self.lrp_rules = list(lrp_rules) if lrp_rules is not None else list(DEFAULT_LRP_RULES)
        self.ig_steps = int(ig_steps)
        if self.task_metric not in ("logprob", "kl"):
            raise ValueError(f"Unknown task_metric: {self.task_metric}")

        if method in ("eap", "eap_ig"):
            if granularity == "neuron":
                raise ValueError(
                    "granularity='neuron' is only supported with method='relp'. "
                    "Use --method relp for neuron-level circuits."
                )
            self.graph = Graph.from_model(model, granularity=granularity)
        elif method == "relp":
            self.graph = None
        else:
            raise ValueError(f"Unknown method: {method}")
        if self.method == "eap_ig" and self.ig_steps <= 0:
            raise ValueError(f"ig_steps must be positive for method='eap_ig' (got {self.ig_steps})")

        # Enable TL hooks needed for component-level scoring
        self.model.cfg.use_split_qkv_input = True
        self.model.cfg.use_attn_result = True
        self.model.cfg.use_hook_mlp_in = True

        if self.use_lrp:
            enable_lrp(model, rules=self.lrp_rules)
        else:
            disable_lrp(model)

    def _get_metric_fn(self, positions: torch.Tensor, target_ids: torch.Tensor):
        def metric(logits: torch.Tensor, corrupted_logits=None, input_lengths=None, label=None) -> torch.Tensor:
            logits_slice = logits[0, positions, :]
            if self.task_metric == "logprob":
                logprobs = logits_slice.log_softmax(dim=-1)
                selected = logprobs.gather(dim=1, index=target_ids.view(-1, 1))
                return selected.sum()
            if corrupted_logits is None:
                raise ValueError("task_metric='kl' requires reference logits")
            ref_slice = corrupted_logits[0, positions, :]
            log_probs = F.log_softmax(logits_slice, dim=-1)
            ref_probs = F.softmax(ref_slice, dim=-1)
            return F.kl_div(log_probs, ref_probs, reduction="batchmean")
        return metric

    def _prepare_paired_inputs(self, example: Example):
        """Tokenize clean and corrupted, pad to the same length, build metric fn."""
        device = self.model.cfg.device
        prompt_tok = self.model.to_tokens(example.prompt, prepend_bos=True)
        clean_full = self.model.to_tokens(example.prompt + example.target, prepend_bos=True)
        corrupted_full = self.model.to_tokens(
            example.corrupted_prompt + example.corrupted_target, prepend_bos=True
        )

        p_ids, f_ids = prompt_tok.tolist()[0], clean_full.tolist()[0]
        lcp = 0
        while lcp < len(p_ids) and lcp < len(f_ids) and p_ids[lcp] == f_ids[lcp]:
            lcp += 1
        gold_ids_list = (
            f_ids[lcp:] if lcp < len(f_ids)
            else self.model.to_tokens(example.target, prepend_bos=False).tolist()[0]
        )
        target_ids = torch.tensor(gold_ids_list, device=device, dtype=torch.long)
        prompt_len = prompt_tok.shape[1]
        positions = torch.arange(
            prompt_len - 1, prompt_len - 1 + len(gold_ids_list), device=device, dtype=torch.long
        )

        pad_id = (
            self.model.tokenizer.pad_token_id
            if self.model.tokenizer.pad_token_id is not None
            else self.model.tokenizer.eos_token_id
        )
        max_len = max(clean_full.shape[1], corrupted_full.shape[1])
        clean_len = clean_full.shape[1]
        corrupted_len = corrupted_full.shape[1]
        clean_tokens = F.pad(clean_full, (0, max_len - clean_full.shape[1]), "constant", pad_id).to(device)
        corrupted_tokens = F.pad(
            corrupted_full, (0, max_len - corrupted_full.shape[1]), "constant", pad_id
        ).to(device)
        clean_attention_mask = torch.zeros((1, max_len), device=device, dtype=torch.long)
        clean_attention_mask[:, :clean_len] = 1
        corrupted_attention_mask = torch.zeros((1, max_len), device=device, dtype=torch.long)
        corrupted_attention_mask[:, :corrupted_len] = 1

        metric = self._get_metric_fn(positions=positions, target_ids=target_ids)
        return (
            clean_tokens,
            corrupted_tokens,
            clean_attention_mask,
            corrupted_attention_mask,
            metric,
            max_len,
        )

    # --- EAP edge-level machinery ------------------------------------------

    def _scores_to_components(self, scores: torch.Tensor) -> Dict[Component, float]:
        from .graph import InputNode, MLPNode, AttentionNode
        component_scores: Dict[Component, float] = {}
        per_component_scores = scores.abs().sum(dim=1)
        for fwd_idx, score in enumerate(per_component_scores.tolist()):
            node = self.graph.idx_to_forward_node.get(fwd_idx)
            if node is None or isinstance(node, InputNode):
                continue
            if isinstance(node, AttentionNode):
                comp = Component(layer=node.layer, kind="head", index=node.head)
            elif isinstance(node, MLPNode):
                comp = Component(layer=node.layer, kind="mlp", index=0)
            else:
                continue
            component_scores[comp] = float(score)
        return component_scores

    # --- RelP node-level machinery ------------------------------------------

    def _hook_targets(self) -> List[HookTarget]:
        """Return a list of (hook_name, scorer) pairs for the current granularity.

        Each scorer takes an attribution tensor `grad * (clean - corrupted)` at that
        hook and returns a ``{Component: score}`` dict.
        """
        n_layers = self.model.cfg.n_layers
        n_heads = self.model.cfg.n_heads
        d_mlp = self.model.cfg.d_mlp

        targets: List[HookTarget] = []
        if self.granularity == "neuron":
            # Arora et al.: MLP neurons only, scored at hook_post
            for layer in range(n_layers):
                name = f"blocks.{layer}.mlp.hook_post"
                targets.append((name, _make_neuron_scorer(layer, d_mlp)))
        else:  # head_mlp
            for layer in range(n_layers):
                targets.append(
                    (f"blocks.{layer}.attn.hook_z", _make_head_scorer(layer, n_heads))
                )
                targets.append(
                    (f"blocks.{layer}.hook_mlp_out", _make_mlp_scorer(layer))
                )
        return targets

    def _extract_relp_example(
        self, example: Example, device: str, autocast_ctx, hook_targets: List[HookTarget]
    ) -> Dict[Component, float]:
        """Run two forwards + one backward with Jafari's RelP formula.

        Score = corrupted_grad × (clean_act − corrupted_act), aggregated per component.
        """
        (
            clean_tokens,
            corrupted_tokens,
            clean_attention_mask,
            corrupted_attention_mask,
            metric,
            _max_len,
        ) = self._prepare_paired_inputs(example)

        hook_names = [name for name, _ in hook_targets]

        def _make_fwd_cache_hook(store: Dict[str, torch.Tensor], name: str):
            def fwd_hook(act, hook=None):
                store[name] = act.detach()
            return fwd_hook

        def _make_bwd_cache_hook(store: Dict[str, torch.Tensor], name: str):
            def bwd_hook(grad, hook=None):
                store[name] = grad.detach()
            return bwd_hook

        # Clean forward — cache activations (no grad).
        clean_cache: Dict[str, torch.Tensor] = {}
        with torch.inference_mode():
            fwd = [(n, _make_fwd_cache_hook(clean_cache, n)) for n in hook_names]
            with self.model.hooks(fwd_hooks=fwd):
                clean_logits = self.model(clean_tokens, attention_mask=clean_attention_mask)
        self.model.reset_hooks()

        # Corrupted forward + backward — cache activations and gradients.
        corrupted_cache: Dict[str, torch.Tensor] = {}
        grad_cache: Dict[str, torch.Tensor] = {}

        fwd = [(n, _make_fwd_cache_hook(corrupted_cache, n)) for n in hook_names]
        bwd = [(n, _make_bwd_cache_hook(grad_cache, n)) for n in hook_names]

        with autocast_ctx:
            with self.model.hooks(fwd_hooks=fwd, bwd_hooks=bwd):
                logits = self.model(corrupted_tokens, attention_mask=corrupted_attention_mask)
                loss = metric(logits, clean_logits)
                loss.backward()
        self.model.zero_grad(set_to_none=True)
        self.model.reset_hooks()

        comp_scores: Dict[Component, float] = {}
        for name, scorer in hook_targets:
            if name not in grad_cache or name not in clean_cache or name not in corrupted_cache:
                continue
            clean_a = clean_cache[name]
            corr_a = corrupted_cache[name]
            grad = grad_cache[name]
            # Safety for rare shape mismatches across padded/unpadded paths.
            if clean_a.shape != corr_a.shape:
                min_len = min(clean_a.shape[1], corr_a.shape[1])
                clean_a = clean_a[:, :min_len]
                corr_a = corr_a[:, :min_len]
                grad = grad[:, :min_len]
            attr = grad * (clean_a - corr_a)
            comp_scores.update(scorer(attr))

        return comp_scores

    # --- Unified entry point ------------------------------------------------

    def extract_circuits_from_examples(
        self, examples: List[Example], task_name: str, amp: bool, device: str
    ) -> Tuple[List[Set[Component]], List[Dict[Component, float]]]:
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if amp and device.startswith("cuda")
            else nullcontext()
        )

        if self.method == "relp":
            return self._extract_relp(examples, task_name, device, autocast_ctx)
        return self._extract_edge_graph(examples, task_name, device, autocast_ctx)

    def _extract_relp(
        self, examples: List[Example], task_name: str, device: str, autocast_ctx
    ) -> Tuple[List[Set[Component]], List[Dict[Component, float]]]:
        circuits: List[Set[Component]] = []
        per_example_scores: List[Dict[Component, float]] = []
        hook_targets = self._hook_targets()
        n_skipped = 0

        for idx, ex in enumerate(examples):
            try:
                comp_scores = self._extract_relp_example(ex, device, autocast_ctx, hook_targets)
            except torch.cuda.OutOfMemoryError:
                n_skipped += 1
                self.model.zero_grad(set_to_none=True)
                self.model.reset_hooks()
                gc.collect()
                torch.cuda.empty_cache()
                print(f"[OOM] Skipping example {idx}")
                continue

            per_example_scores.append(comp_scores)
            items = sorted(comp_scores.items(), key=lambda x: x[1], reverse=True)
            circuits.append({c for c, _ in items})

            if (idx + 1) % 10 == 0 or (idx + 1) == len(examples):
                print(
                    f"[{task_name}] (relp/{self.granularity}) "
                    f"{idx + 1}/{len(examples)} examples processed"
                )
            torch.cuda.empty_cache()

        if n_skipped:
            print(f"[WARN] {n_skipped}/{len(examples)} examples skipped due to OOM")
        return circuits, per_example_scores

    def _extract_edge_graph(
        self, examples: List[Example], task_name: str, device: str, autocast_ctx
    ) -> Tuple[List[Set[Component]], List[Dict[Component, float]]]:
        circuits: List[Set[Component]] = []
        per_example_scores: List[Dict[Component, float]] = []

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

        n_skipped = 0
        for idx, ex in enumerate(examples):
            try:
                if self.method == "eap":
                    (
                        clean_tokens,
                        corrupted_tokens,
                        clean_attention_mask,
                        corrupted_attention_mask,
                        metric,
                        ex_len,
                    ) = self._prepare_paired_inputs(ex)
                    with autocast_ctx:
                        scores = attribute_single_example(
                            model=self.model, graph=self.graph, metric=metric,
                            clean_tokens=clean_tokens, corrupted_tokens=corrupted_tokens,
                            activation_difference=_get_work_buf(ex_len),
                            clean_attention_mask=clean_attention_mask,
                            corrupted_attention_mask=corrupted_attention_mask,
                        )
                elif self.method == "eap_ig":
                    (
                        clean_tokens,
                        corrupted_tokens,
                        clean_attention_mask,
                        corrupted_attention_mask,
                        metric,
                        ex_len,
                    ) = self._prepare_paired_inputs(ex)
                    with autocast_ctx:
                        scores = attribute_single_example_ig(
                            model=self.model,
                            graph=self.graph,
                            metric=metric,
                            clean_tokens=clean_tokens,
                            corrupted_tokens=corrupted_tokens,
                            activation_difference=_get_work_buf(ex_len),
                            steps=self.ig_steps,
                            clean_attention_mask=clean_attention_mask,
                            corrupted_attention_mask=corrupted_attention_mask,
                        )
                else:
                    raise ValueError(f"Unknown edge-graph method: {self.method}")
            except torch.cuda.OutOfMemoryError:
                n_skipped += 1
                self.model.zero_grad(set_to_none=True)
                self.model.reset_hooks()
                work_buf = None
                work_buf_seq_len = 0
                gc.collect()
                torch.cuda.empty_cache()
                print(f"[OOM] Skipping example {idx} (seq_len too large for available VRAM)")
                continue

            comp_scores = self._scores_to_components(scores)
            per_example_scores.append(comp_scores)
            items = sorted(comp_scores.items(), key=lambda x: x[1], reverse=True)
            comp_set = {c for c, _ in items}
            circuits.append(comp_set)

            if (idx + 1) % 10 == 0 or (idx + 1) == len(examples):
                print(
                    f"[{task_name}] ({self.method}) {idx + 1}/{len(examples)} "
                    f"examples processed (last circuit size={len(comp_set)})"
                )
            torch.cuda.empty_cache()

        if n_skipped:
            print(f"[WARN] {n_skipped}/{len(examples)} examples skipped due to OOM")

        del work_buf
        gc.collect()
        torch.cuda.empty_cache()
        return circuits, per_example_scores


# --- Per-granularity scorers --------------------------------------------------

def _make_neuron_scorer(layer: int, d_mlp: int):
    def scorer(attr: torch.Tensor) -> Dict[Component, float]:
        # attr: [batch, pos, d_mlp] — sum over batch, pos, keep sign
        per_neuron = attr.sum(dim=(0, 1))  # [d_mlp]
        out: Dict[Component, float] = {}
        vals = per_neuron.tolist()
        for i in range(d_mlp):
            out[Component(layer=layer, kind="neuron", index=i)] = float(vals[i])
        return out
    return scorer


def _make_head_scorer(layer: int, n_heads: int):
    def scorer(attr: torch.Tensor) -> Dict[Component, float]:
        # attr: [batch, pos, head, d_head]; sum over batch, pos, d_head
        per_head = attr.sum(dim=(0, 1, -1))  # [n_heads]
        out: Dict[Component, float] = {}
        vals = per_head.tolist()
        for h in range(n_heads):
            out[Component(layer=layer, kind="head", index=h)] = float(vals[h])
        return out
    return scorer


def _make_mlp_scorer(layer: int):
    def scorer(attr: torch.Tensor) -> Dict[Component, float]:
        # attr: [batch, pos, d_model]; one scalar per layer
        score = float(attr.sum().item())
        return {Component(layer=layer, kind="mlp", index=0): score}
    return scorer


__all__ = ["Component", "CircuitExtractor"]
