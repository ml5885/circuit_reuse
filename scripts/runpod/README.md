# Boolean rerun on a rented GPU (RunPod)

One pod per model, 48 GB card (A40 / A6000 / L40S), RunPod PyTorch template, `/workspace` volume.

## On your laptop, once

```
git push                                   # the pod clones from GitHub
```

## On each pod

```
cd /workspace && git clone https://github.com/ml5885/circuit_reuse.git && cd circuit_reuse
cp scripts/runpod/env.example.sh scripts/runpod/env.sh   # fill in HF_TOKEN and NOTIFY_* values
source scripts/runpod/env.sh
bash scripts/runpod/setup.sh                # installs deps, prefetches models + datasets, emails "setup complete"
```

## On your laptop, per pod

```
scripts/runpod/push_inputs.sh root@<pod-ip> <ssh-port>   # ~110 MB: the other tasks' circuits + old cross-task matrices
```

## On each pod

```
source scripts/runpod/env.sh
tmux new -s boolean 'bash scripts/runpod/run_boolean.sh google/gemma-2-2b'
```

Models: `google/gemma-2-2b`, `google/gemma-2-2b-it`, `meta-llama/Llama-3.2-3B`, `meta-llama/Llama-3.2-3B-Instruct`, `qwen3-4b`.

The script runs 8 stages (extraction then cross-task refresh, for eap_ig/relp x head_mlp/neuron).
It emails on start, after every stage, on failure (with the last 60 log lines), if a log is silent
for 45 minutes, and at the end. Stages that finished are skipped on rerun (`logs/<stage>.done`),
so after a crash just rerun the same command.

## On your laptop, when a pod says ALL DONE

```
scripts/runpod/pull_results.sh root@<pod-ip> <ssh-port>
python -m analysis.granularity_parity          # rebuilds results/granularity_parity_analysis/*.csv
python -m analysis.review_plots                 # review_plots.md
```

Then stop the pod.

## What the rerun changes

`circuit_reuse/dataset.py`: `BooleanDataset` now flips a literal that changes the truth value
(expressions with no such literal are resampled). `cross_task_experiment.py --refresh-tasks boolean`
recomputes only the 11 cells per matrix where Boolean is donor or target and copies the other 25
from the existing file, which is why the old matrices have to be pushed to the pod first.
