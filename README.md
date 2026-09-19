# Circuit Reuse

Code for *How Much Do Circuits Tell Us? Measuring the Consistency and Specificity of Language Model Circuits*.

For each example in a task dataset, we extract a circuit, defined as the top-K% of components by attribution score. We then define the task's shared circuit, $S_P$, as the set of components that appear in at least P% of these per-example circuits. We evaluate $S_P$ on two criteria.

**Consistency** is a measure of how well one circuit describes the whole task. Reuse@P quantifies it as the fraction of each example's circuit that is contained in $S_P$, averaged over examples. To check that $S_P$ matters causally, we zero-ablate it and measure the accuracy drop relative to ablating a random set of components with the same size and the same head/MLP composition.

**Specificity** is a measure of how much the circuit belongs to its task rather than to the model in general. We ablate task A's shared circuit and measure the accuracy drop on task A, then ablate every other task's shared circuit and measure the drop on task A again. A specific circuit hurts its own task more than the others' circuits do.

We run this at two granularities, attention heads and MLP blocks (`head_mlp`) or individual MLP neurons (`neuron`), with three attribution methods (EAP, EAP-IG, RelP), on six tasks and five models: Gemma 2 2B and its instruction-tuned variant, Llama 3.2 3B and its Instruct variant, and Qwen3 4B.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -e .
huggingface-cli login   # gemma-2 and llama-3.2 are gated
```

## Layout

| Path | Contents |
|---|---|
| `main_experiment.py` | Runs attribution and the within-task evaluation for one model, task, method and granularity. Writes `metrics.json`. |
| `cross_task_experiment.py` | Builds the cross-task ablation matrix for every (K, P) setting. Writes one JSON file per setting. `--ablation mean` swaps zero ablation for mean ablation. |
| `cross_task_mean_ablation.py` | Older mean-ablation driver behind `results/cross_task_ablation_mean_k10` (EAP components only, schema v1). Superseded by `--ablation mean`. |
| `selective_ablation_experiment.py` | For a task pair (A, B), ablates the shared core $C_A \cap C_B$ and the residuals $C_A \setminus C_B$ and $C_B \setminus C_A$ separately. |
| `circuit_reuse/circuit_extraction.py` | `CircuitExtractor`, which implements EAP, EAP-IG and RelP at either granularity. |
| `circuit_reuse/graph.py` | The edge graph used by `eap` and `eap_ig`. |
| `circuit_reuse/lrp_patch.py` | LRP backward rules for `relp`. |
| `circuit_reuse/evaluate.py` | Accuracy evaluation under zero and mean ablation. |
| `circuit_reuse/dataset.py` | Tasks and counterfactuals. |
| `models/olmo_adapter.py` | `load_model_any`, which loads through TransformerLens and falls back to a hooked HF wrapper for OLMo checkpoints. |
| `analysis/` | Aggregation scripts and figure generators. |
| `scripts/`, `paper2/`, `results*/`, `cache*/` | SLURM and pod scripts, paper source, outputs, and attribution caches. Not tracked. |

## Tasks

| Task | Source | Counterfactual |
|---|---|---|
| `addition` | generated, `Compute: a + b = ` | different (a, b) |
| `boolean` | generated, `Evaluate: <expr> = ` | one literal flipped so the value changes; expressions with no such literal resampled |
| `ioi` | `mib-bench/ioi` | `s2_io_flip` (answer flips IO → S) |
| `mcqa` | `mib-bench/copycolors_mcqa` | `answerPosition` |
| `arc_easy`, `arc_challenge` | `mib-bench/arc_*` | `answerPosition` |

Each run generates or reads `--num_examples` examples, shuffles them with `--seed`, and holds out `--val-fraction` of them for validation. `google/gemma-2-2b-it` receives a few-shot prefix on `addition`.

## Pipeline

### 1. Extraction

```bash
python main_experiment.py \
  --model_name google/gemma-2-2b --task ioi --num_examples 1000 --digits 3 \
  --method eap_ig --granularity head_mlp --ig-steps 5 \
  --top_k_list 1,5,10,20,30 --reuse-thresholds 50,75,85,90,95,96,97,98,99,100 \
  --perm-trials 5000 --dtype bf16 --amp --device cuda \
  --run-name granularity_parity_eap_ig_head_mlp \
  --output-dir results/granularity_parity/granularity_parity_extraction \
  --cache-dir cache_granularity_parity/eap_ig_head_mlp
```

The output is `<output-dir>/<run-name>/<model>__main__<task>__<method>__.../metrics.json`, with the following structure:

```
baseline_{train,val}_accuracy
by_k[K].thresholds[P]: shared_components, shared_circuit_size, reuse_percent,
                       train/val: {ablation_accuracy, control_accuracy, permutation}
```

Per-example attribution scores are cached as JSONL files in `--cache-dir`, one line per example with every component's score. `--analysis` recomputes the metrics from the cache without loading a model; `--force-extract` ignores the cache and re-runs attribution.

### 2. Cross-task ablation

```bash
python cross_task_experiment.py \
  --results-dir results/granularity_parity/granularity_parity_extraction/granularity_parity_eap_ig_head_mlp \
  --model_name google/gemma-2-2b --method eap_ig --granularity head_mlp \
  --tasks addition,arc_challenge,arc_easy,boolean,ioi,mcqa \
  --K 1,5,10,20,30 --threshold 50,75,85,90,95,96,97,98,99,100 \
  --num-examples 100 --digits 3 --device cuda \
  --output-dir results/granularity_parity/granularity_parity_cross_task/eap_ig_head_mlp
```

The model is loaded once and every (K, P) matrix is evaluated. Each `cross_task_<model>_<method>_<granularity>_K<K>_p<P>.json` file holds `cells[donor][target]` with the baseline and ablated accuracies and the drop in percentage points, plus the baselines and circuit sizes. Matrices that already exist are skipped, so the command can be rerun after an interruption. `--refresh-tasks t1,t2` recomputes only the cells where the donor or the target is in the list and copies the other cells from the existing file.

### 3. Aggregation and figures

```bash
python -m analysis.granularity_parity --results-root results/granularity_parity \
    --output-dir results/granularity_parity_analysis --plots
```

This reads every `metrics.json` and cross-task JSON under `--results-root` and writes the tidy CSVs (`extraction_tidy.csv`, `cross_task_tidy.csv`, `overlap_pairs.csv`, `bootstrap_summary.csv`, `circuit_composition.csv`) and summary figures. The scripts below read those CSVs:

| Script | Output |
|---|---|
| `analysis/paper_candidates.py` | The paper's figures, written to `paper2/candidates/`; `sync_paper()` copies the chosen ones into the paper's `figures/`. |
| `analysis/paper2_figures.py` | Alternative renderings of each main-text claim. |
| `analysis/appendix_tables.py --method M --granularity G` | Appendix LaTeX tables. |
| `analysis/review_plots.py [A1 ...]` | The analyses requested by the reviews that need no GPU. Writes `paper2/review_plots.md`. |
| `analysis/pat_confound_checks.py` | Score-rule and top-K confound checks computed from the per-example caches. |
| `analysis/zero_vs_mean_ablation_report.py [--config M_G --K 10 --P 50]` | Zero-vs-mean robustness figures and statistics; `--config` selects a granularity-parity configuration. |
| `analysis/zero_vs_mean_ablation.py`, `analysis/selective_ablation_summary.py` | Summaries of the original mean-ablation and selective-ablation experiments. |

The older scripts in `analysis/` (`plot_k_sweep.py`, `multiplot_*.py`, ...) predate the tidy CSVs and read `metrics.json` files directly.

## `main_experiment.py` arguments

| Argument | Meaning |
|---|---|
| `--method` | Attribution method: `eap`, `eap_ig`, or `relp`. `neuron_attr` is a deprecated alias for `--method relp --granularity neuron`. |
| `--granularity` | `head_mlp` scores attention heads at `attn.hook_z` and MLP blocks at `hook_mlp_out`; `neuron` scores MLP neurons at `mlp.hook_post` and is supported by `eap_ig` and `relp`. |
| `--top_k_list` | Per-example circuit sizes, as percentages of all components. |
| `--reuse-thresholds` | The consensus thresholds P, as percentages. |
| `--perm-trials` | Number of trials for the paired permutation test of the shared circuit against the control. |
| `--ig-steps` | Number of integrated-gradients interpolation steps for `eap_ig` (default 5). |
| `--task-metric` | Attribution objective: `logprob`, the summed log-probability of the gold tokens (default), or `kl`. |
| `--ignore-type` | Sample the control without matching component types. |
| `--score-threshold` | Select components with `\|score\| ≥ τ Σ\|scores\|` instead of the top-K%. Results are stored under `by_threshold`. |
| `--use-lrp` / `--no-use-lrp`, `--lrp-rules` | Enable or disable the LRP backward rules for `relp`, and choose them (default `LN-rule,AH-rule,Half-rule`). |

## Attribution methods

| Method | Node score | Cost per example |
|---|---|---|
| `eap` | Edge Attribution Patching (Syed et al. 2023). Each edge is scored as (corrupted activation − clean activation) · clean gradient; a node's score is the sum of the absolute scores of its outgoing edges, so it is non-negative. | 2 forward passes, 1 backward pass |
| `eap_ig` | EAP with integrated gradients (Hanna et al. 2024): the same edge score with the gradient averaged over `ig_steps` points on the path from the clean to the corrupted input embedding. Non-negative at `head_mlp`; at `neuron` the score is signed. | 2 forward passes, `ig_steps` backward passes |
| `relp` | Relevance Patching (Jafari et al. 2025): corrupted gradient · (clean activation − corrupted activation) at one hook per component, with LRP backward rules. Signed. | 2 forward passes, 1 backward pass |

Top-K selection ranks components by the stored score. Where the score is signed, the circuit therefore consists of the most positive components.

## References & acknowledgements

This project builds on:

- [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) — hook-based mechanistic-interpretability library; all forward passes and component hooks are TL primitives.
- [eap-ig](https://github.com/hannamw/eap-ig) — our `eap` and `eap_ig` edge-graph paths (`circuit_reuse/graph.py`) are derived from this implementation of Edge Attribution Patching (Syed et al. 2023, [arXiv:2310.10348](https://arxiv.org/abs/2310.10348)).
- [RelP (Jafari et al. 2025)](https://arxiv.org/abs/2508.21258) — the `relp` method and the LRP rules in `circuit_reuse/lrp_patch.py` are ported from the authors' TransformerLens fork at [FarnoushRJ/RelP](https://github.com/FarnoushRJ/RelP) (see `reference_code/RelP/`).
- [ADAG / Arora et al. 2026](https://arxiv.org/abs/2601.22594) — "Language Model Circuits Are Sparse in the Neuron Basis." `--method relp --granularity neuron` reproduces their MLP-neuron-basis circuit scoring. The `reference_code/circuits/` directory vendors the ADAG library (Transluce) for comparison.
- LRP propagation rules: LN-rule (Ali et al. 2022), AH-rule (Ali et al. 2022), Half-rule (Arras et al. 2019; Jafari et al. 2024).
