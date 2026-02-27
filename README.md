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

## Analysis scripts

All in `analysis/`:

- `plot_accuracy_and_lift_bars.py` -- per-model accuracy and lift bar charts
- `plot_k_sweep.py` -- lift and reuse vs top-K line plots
- `plot_attribution_scores.py` -- attribution score distribution histograms
- `multiplot_lift_and_reuse.py` -- multi-panel lift and reuse bar charts
- `multiplot_pvalues.py` -- permutation p-value visualizations
- `generate_air_tables.py` -- LaTeX AIR tables (pretraining sweep)
- `cross_task_tables.py` -- cross-task ablation confusion matrices and heatmaps

## Caching

Attribution scores are cached as JSONL in `cache/` (configurable via `--cache-dir`). Filename encodes model, revision, task, method, N, digits, and seed. Use `--force-extract` to recompute.

## Output

Each run saves `metrics.json` with baseline accuracies, per-(K, p) shared circuit components, ablation/control accuracies, knockout_diff (AIR), and permutation test results on train/val splits.

Cross-task experiment saves a confusion matrix CSV and structured JSON with raw and baseline-normalized accuracy drops.

## References

- [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)
- [eap-ig](https://github.com/hannamw/eap-ig)
