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
- `--method`: Attribution method (`eap`, `gradient`, or `neuron_attr`).
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

| Method | Granularity | Scoring | Description |
|---|---|---|---|
| `"eap"` (default) | Edge-level | `activation_diff × gradient` on edges | Edge Attribution Patching (Syed et al. 2023). Scores edges between components, aggregated to per-component scores |
| `"gradient"` | Edge-level | Clean-pass gradient magnitude | Same graph as EAP but uses only the clean gradient (no corrupted forward pass) |
| `"neuron_attr"` | Node-level | `activation_diff × gradient` per neuron/head | Following Arora et al. ("Language Model Circuits Are Sparse in the Neuron Basis"). Scores individual MLP neurons and attention heads directly — no edge graph needed |

`neuron_attr` automatically uses neuron-level granularity (the `--granularity` flag is ignored).

## Node granularity

The `CircuitExtractor` supports configurable node granularity via the `granularity` parameter:

| Granularity | Nodes | Description |
|---|---|---|
| `"head_mlp"` (default) | Attention heads + MLP blocks | Standard per-head, per-MLP-layer circuit analysis |
| `"block"` | Attention blocks + MLP blocks | Merges all heads in a layer into a single attention block node. Faster and lower memory, but loses head-level resolution |
| `"neuron"` | Same graph as `head_mlp` | Same computation; reserved for future per-neuron score decomposition |

```python
from circuit_reuse.circuit_extraction import CircuitExtractor

extractor = CircuitExtractor(model, method="eap", granularity="block")
```

**Impact on computation**: `"block"` reduces `n_forward` from `1 + n_layers*(n_heads+1)` to `1 + 2*n_layers`, proportionally shrinking the activation difference buffer and score matrix. For GPT-2 small this is 157 → 25.

## Score threshold

Use `--score-threshold` to select circuit components by absolute score magnitude instead of top-K percentage:

```bash
# Extract + threshold-based selection
python main_experiment.py \
  --model_name gpt2-small --task ioi --num_examples 50 \
  --top_k_list 5,10 --score-threshold 0.005 \
  --method neuron_attr --device cpu

# Recompute from cached scores (no re-extraction)
python main_experiment.py \
  --model_name gpt2-small --task ioi --num_examples 50 \
  --top_k_list 5,10 --score-threshold 0.005 \
  --analysis --method eap --device cpu
```

A component is included if `|score| >= τ × Σ|all scores|` for that example. Results are stored in `by_threshold` alongside the usual `by_k`.

## Caching

Attribution scores are cached as JSONL in `cache/` (configurable via `--cache-dir`). Filename encodes model, revision, task, method, N, digits, and seed. Use `--force-extract` to recompute.

## Output

Each run saves `metrics.json` with baseline accuracies, per-(K, p) shared circuit components, ablation/control accuracies, knockout_diff (AIR), and permutation test results on train/val splits.

Cross-task experiment saves a confusion matrix CSV and structured JSON with raw and baseline-normalized accuracy drops.

## References

- [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)
- [eap-ig](https://github.com/hannamw/eap-ig)
