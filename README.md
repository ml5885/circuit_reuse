# Circuit Reuse Experiments

Measures per-example circuits via Edge Attribution Patching, quantifies how much they overlap across examples (reuse@p), and validates shared circuits causally via zero-ablation against size-matched random controls.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Quick start

```bash

# Single-task knockout experiment
python main_experiment.py \
  --model_name qwen3-0.6b --task ioi --num_examples 50 \
  --top_k_list 5,10 --reuse-thresholds 95,99,100 \
  --perm-trials 2000 --dtype bf16 --device cpu --debug

# Cross-task ablation confusion matrix (requires prior main experiment results)
python cross_task_experiment.py \
  --results-dir results/my_run --model_name qwen3-4b \
  --tasks "boolean,addition,ioi,mcqa" --K 100 --threshold 100 \
  --output-dir results/cross_task
```

## Key CLI args

- `--top_k_list`: Per-example top-K values as percentages (e.g., `50,75,100`).
- `--reuse-thresholds`: Thresholds p as percentages (e.g., `95,99,100`).
- `--perm-trials`: Trials for paired permutation test (shared vs control).
- `--ignore-type`: Sample control randomly regardless of head/MLP type.
- `--analysis`: Skip extraction, load cached attributions only.
- `--method`: Attribution method (`eap`, `eap_ig`, `relp`, or `neuron_attr`). `neuron_attr` is a deprecated alias for `--method relp --granularity neuron`.
- `--ig-steps`: Number of interpolation steps for `--method eap_ig` (default: `5`).
- `--task-metric`: Attribution objective (`logprob` by default, or `kl`).
- `--granularity`: `head_mlp` (default) or `neuron`. `neuron` requires `--method relp`.
- `--use-lrp` / `--no-use-lrp`: force LRP backward rules on/off. Default: on for `relp`, off otherwise.
- `--lrp-rules`: comma-separated LRP rules. Default: `LN-rule,AH-rule,Half-rule`.
- `--score-threshold`: Absolute score threshold (e.g., `0.005`). When set, selects components by score magnitude instead of top-K%.

## Analysis scripts

All in `analysis/`:

- `plot_accuracy_and_lift_bars.py` -- per-model accuracy and lift bar charts
- `plot_k_sweep.py` -- lift and reuse vs top-K line plots
- `plot_attribution_scores.py` -- attribution score distribution histograms
- `multiplot_lift_and_reuse.py` -- multi-panel lift and reuse bar charts
- `multiplot_pvalues.py` -- permutation p-value visualizations
- `generate_air_tables.py` -- LaTeX AIR tables (pretraining sweep)
- `cross_task_tables.py` -- cross-task ablation confusion matrices and heatmaps

## Attribution methods

| Method | Scoring path | Formula | Description |
|---|---|---|---|
| `eap` (default) | Edge-level graph | `(corrupted_act − clean_act) × clean_grad` on edges, aggregated per node | Edge Attribution Patching (Syed et al. 2023). Clean and corrupted sequences are padded to a shared length and run with explicit attention masks. |
| `eap_ig` | Edge-level graph | `mean_t[(corrupted_act − clean_act) × grad_t]` along the clean↔corrupted input-embedding path | EAP with integrated gradients over inputs, using the same paired-input padding and attention-mask handling as `eap`. |
| `relp` | Node-level | `corrupted_grad × (clean_act − corrupted_act)` at one hook per component | Relevance Patching (Jafari et al. 2025, arXiv:2508.21258). The backward pass uses LRP rules (`LN-rule`, `AH-rule`, `Half-rule`) so gradients are better-conditioned through LayerNorm/RMSNorm, attention softmax, and gated-MLP multiplications. At `--granularity neuron` this reproduces the MLP-neuron-basis circuits of Arora et al. 2026 (arXiv:2601.22594). |
| `neuron_attr` | Alias | — | Deprecated alias for `--method relp --granularity neuron`. |

## Task metric

`CircuitExtractor` optimizes a scalar objective over continuation positions during attribution.

- `logprob` (default): sum of gold-token log-probabilities over the clean continuation.
- `kl`: KL divergence to the paired reference logits. For the edge-graph methods (`eap`, `eap_ig`), gradients are taken on the clean/interpolated path against fixed corrupted logits. For `relp`, gradients are taken on the corrupted path against fixed clean logits.

## Node granularity

| Granularity | Components scored (at `--method relp`) | Notes |
|---|---|---|
| `head_mlp` (default) | Attention heads (at `attn.hook_z`) + MLP layer outputs (at `hook_mlp_out`) | Standard per-head, per-layer granularity. |
| `neuron` | MLP neurons only (at `mlp.hook_post`, no attention) | Arora et al.'s neuron basis. Only valid with `--method relp`. |

For the edge-graph methods `eap` and `eap_ig`, only `head_mlp` is supported; `neuron` requires `--method relp`.

```python
from circuit_reuse.circuit_extraction import CircuitExtractor

# RelP at head-level — scores attention heads and MLP layers with LRP backward
extractor = CircuitExtractor(model, method="relp", granularity="head_mlp")

# Arora's neuron-basis circuits — per-MLP-neuron RelP
extractor = CircuitExtractor(model, method="relp", granularity="neuron")

# Classic EAP edge-graph method (unchanged)
extractor = CircuitExtractor(model, method="eap", granularity="head_mlp")

# EAP-IG over interpolated input embeddings
extractor = CircuitExtractor(model, method="eap_ig", granularity="head_mlp", ig_steps=5)

# Use KL instead of summed gold-token log-prob
extractor = CircuitExtractor(model, method="eap", task_metric="kl")
```

**Computational cost:** `relp` uses two forward passes and one backward pass per example. `eap_ig` uses one corrupted forward pass, one clean forward pass, and `ig_steps` backward passes along the interpolation path.

## Score threshold

Use `--score-threshold` to select circuit components by absolute score magnitude instead of top-K percentage:

```bash
# Extract + threshold-based selection at the MLP-neuron level
python main_experiment.py \
  --model_name gpt2-small --task ioi --num_examples 50 \
  --top_k_list 5,10 --score-threshold 0.005 \
  --method relp --granularity neuron --device cpu

# Recompute from cached scores (no re-extraction)
python main_experiment.py \
  --model_name gpt2-small --task ioi --num_examples 50 \
  --top_k_list 5,10 --score-threshold 0.005 \
  --analysis --method eap --device cpu
```

A component is included if `|score| >= τ × Σ|all scores|` for that example. Results are stored in `by_threshold` alongside the usual `by_k`.

## Caching

Attribution scores are cached as JSONL in `cache/` (configurable via `--cache-dir`). Filenames encode model, revision, task, method, task metric when non-default, granularity when non-default, `ig_steps` for `eap_ig`, example count, digits, and seed. Use `--force-extract` to recompute.

## Output

Each run saves `metrics.json` with baseline accuracies, per-(K, p) shared circuit components, ablation/control accuracies, knockout_diff (AIR), and permutation test results on train/val splits.

Cross-task experiment saves a confusion matrix CSV and structured JSON with raw and baseline-normalized accuracy drops.

## References & acknowledgements

This project builds on:

- [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) — hook-based mechanistic-interpretability library; all forward passes and component hooks are TL primitives.
- [eap-ig](https://github.com/hannamw/eap-ig) — our `eap` and `eap_ig` edge-graph paths (`circuit_reuse/graph.py`) are derived from this implementation of Edge Attribution Patching (Syed et al. 2023, [arXiv:2310.10348](https://arxiv.org/abs/2310.10348)).
- [RelP (Jafari et al. 2025)](https://arxiv.org/abs/2508.21258) — the `relp` method and the LRP rules in `circuit_reuse/lrp_patch.py` are ported from the authors' TransformerLens fork at [FarnoushRJ/RelP](https://github.com/FarnoushRJ/RelP) (see `reference_code/RelP/`).
- [ADAG / Arora et al. 2026](https://arxiv.org/abs/2601.22594) — "Language Model Circuits Are Sparse in the Neuron Basis." `--method relp --granularity neuron` reproduces their MLP-neuron-basis circuit scoring. The `reference_code/circuits/` directory vendors the ADAG library (Transluce) for comparison.
- LRP propagation rules: LN-rule (Ali et al. 2022), AH-rule (Ali et al. 2022), Half-rule (Arras et al. 2019; Jafari et al. 2024).
