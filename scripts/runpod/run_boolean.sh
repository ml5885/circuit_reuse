#!/bin/bash
# Re-extract Boolean with the fixed counterfactual and refresh the Boolean cells
# of the cross-task matrices, for one model and all four method x granularity
# configurations. Resumable: finished stages are skipped on rerun.
#
#   source scripts/runpod/env.sh
#   tmux new -s boolean 'bash scripts/runpod/run_boolean.sh google/gemma-2-2b'
#
# Inputs (rsync them first with push_inputs.sh from your laptop):
#   $WORK/results/granularity_parity/granularity_parity_extraction/granularity_parity_<method>_<gran>/*/metrics.json
#   $WORK/results/granularity_parity/granularity_parity_cross_task/<method>_<gran>_l40s/*.json
set -uo pipefail
: "${WORK:?source scripts/runpod/env.sh first}"
MODEL=${1:?usage: run_boolean.sh MODEL_NAME}
SLUG=${MODEL//\//_}
cd "$WORK"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOGS="$WORK/logs"; mkdir -p "$LOGS"
EXTRACT_ROOT="$WORK/results/granularity_parity/granularity_parity_extraction"
CROSS_ROOT="$WORK/results/granularity_parity/granularity_parity_cross_task"
CACHE_ROOT="$WORK/cache_granularity_parity"

notify() { python scripts/runpod/notify.py "$1" <<< "${2:-}"; }
fail() {  # $1 stage, $2 log
    notify "FAILED $1" "$(tail -n 60 "$2")"
    exit 1
}
run_stage() {  # $1 stage name, $2 log file, rest: command
    local stage=$1 log=$2; shift 2
    local done="$LOGS/$stage.done"
    if [[ -f $done ]]; then echo "[skip] $stage"; return; fi
    echo "[start] $stage -> $log"
    bash scripts/runpod/watchdog.sh "$log" "$stage" & local wd=$!
    "$@" > "$log" 2>&1; local rc=$?
    kill "$wd" 2> /dev/null
    [[ $rc -eq 0 ]] || fail "$stage" "$log"
    touch "$done"
    notify "done $stage" "$(tail -n 5 "$log")"
}

notify "started boolean rerun: $MODEL" "$(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader)"
for METHOD in eap_ig relp; do
  for GRAN in head_mlp neuron; do
    CFG="${METHOD}_${GRAN}"
    run_stage "extract_${SLUG}_${CFG}" "$LOGS/extract_${SLUG}_${CFG}.log" \
      python main_experiment.py --model_name "$MODEL" --task boolean --method "$METHOD" \
        --granularity "$GRAN" --ig-steps 5 --top_k_list 1,5,10,20,30 \
        --reuse-thresholds 50,75,85,90,95,96,97,98,99,100 --perm-trials 5000 \
        --digits 3 --num_examples 1000 --dtype bf16 --amp --device cuda \
        --run-name "granularity_parity_${CFG}" --output-dir "$EXTRACT_ROOT" \
        --cache-dir "$CACHE_ROOT/$CFG" --force-extract
    run_stage "cross_${SLUG}_${CFG}" "$LOGS/cross_${SLUG}_${CFG}.log" \
      python cross_task_experiment.py --results-dir "$EXTRACT_ROOT/granularity_parity_${CFG}" \
        --model_name "$MODEL" --method "$METHOD" --granularity "$GRAN" \
        --tasks addition,arc_challenge,arc_easy,boolean,ioi,mcqa \
        --K 1,5,10,20,30 --threshold 50,75,85,90,95,96,97,98,99,100 \
        --num-examples 100 --digits 3 --device cuda \
        --output-dir "$CROSS_ROOT/${CFG}_l40s" --refresh-tasks boolean
  done
done
notify "ALL DONE boolean rerun: $MODEL" "$(ls "$LOGS"/*.done | wc -l) stages complete"
