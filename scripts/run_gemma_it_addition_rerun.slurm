#!/bin/bash
#SBATCH --job-name=gemma_it_add
#SBATCH --output=/home/ml6/logs/sbatch/gemma_it_add_%j.out
#SBATCH --error=/home/ml6/logs/sbatch/gemma_it_add_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=96G
#SBATCH --mail-user=ml6@andrew.cmu.edu
#SBATCH --mail-type=START,END,FAIL
#SBATCH --partition=general

# Re-run gemma-2-2b-it addition with few-shot fix:
#   1) Extract circuits (addition only)
#   2) Cross-task ablation (all K values)
#   3) Selective ablation (addition vs all other tasks)

export HF_HOME=/data/user_data/ml6/.hf_cache
export HF_HUB_CACHE=/data/user_data/ml6/.hf_cache/hub
export HF_DATASETS_CACHE=/data/user_data/ml6/.hf_cache/datasets
export HF_HUB_OFFLINE=0
export HF_TOKEN_PATH=/home/ml6/.cache/huggingface/token
mkdir -p /home/ml6/logs/sbatch "$HF_HOME"

eval "$(conda shell.bash hook)"
conda activate circuit_reuse

cd /home/ml6/circuit_reuse
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA_DIR=/data/user_data/ml6/circuit_reuse
CACHE_DIR="$DATA_DIR/cache"
MODEL="google/gemma-2-2b-it"
DIGITS=3

# ── Step 1: Extract addition circuits ──────────────────────────────

echo "[STEP 1] Extract addition circuits for $MODEL"
python main_experiment.py \
    --model_name "$MODEL" \
    --task addition \
    --top_k_list "1,5,10,20,30" \
    --reuse-thresholds "95,96,97,98,99,100" \
    --perm-trials 5000 \
    --digits "$DIGITS" \
    --num_examples 1000 \
    --dtype bf16 \
    --amp \
    --device cuda \
    --run-name "cross_task" \
    --output-dir "$DATA_DIR/results" \
    --cache-dir "$CACHE_DIR" \
    --results-home /home/ml6/circuit_reuse/results \
    --force-extract

# ── Step 2: Cross-task ablation ────────────────────────────────────

RESULTS_DIR="$DATA_DIR/results/cross_task"
CROSS_TASK_BASE_DIR="/home/ml6/circuit_reuse/results"
tasks=("addition" "arc_challenge" "arc_easy" "boolean" "ioi" "mcqa" "mmlu")
K_LIST=(1 5 10 20 30)
TASKS_CSV=$(IFS=,; echo "${tasks[*]}")

echo "[STEP 2] Cross-task ablation for $MODEL"
for K in "${K_LIST[@]}"; do
    CROSS_TASK_DIR="$CROSS_TASK_BASE_DIR/cross_task_ablation_k${K}"
    mkdir -p "$CROSS_TASK_DIR"

    python cross_task_experiment.py \
        --results-dir "$RESULTS_DIR" \
        --model_name "$MODEL" \
        --tasks "$TASKS_CSV" \
        --K "$K" \
        --threshold 100 \
        --digits "$DIGITS" \
        --num-examples 100 \
        --device cuda \
        --output-dir "$CROSS_TASK_DIR"
done

# ── Step 3: Selective ablation ─────────────────────────────────────

OUTPUT_BASE="/home/ml6/circuit_reuse/results"

echo "[STEP 3] Selective ablation for $MODEL"
for K in "${K_LIST[@]}"; do
    OUTPUT_DIR="$OUTPUT_BASE/selective_ablation_k${K}"
    mkdir -p "$OUTPUT_DIR"

    for (( i=0; i<${#tasks[@]}; i++ )); do
        for (( j=0; j<${#tasks[@]}; j++ )); do
            [ "$i" -eq "$j" ] && continue
            TASK_A=${tasks[$i]}
            TASK_B=${tasks[$j]}

            python selective_ablation_experiment.py \
                --results-dir "$RESULTS_DIR" \
                --model_name "$MODEL" \
                --task-a "$TASK_A" \
                --task-b "$TASK_B" \
                --eval-tasks "$TASKS_CSV" \
                --K "$K" \
                --threshold 100 \
                --digits "$DIGITS" \
                --num-examples 100 \
                --device cuda \
                --output-dir "$OUTPUT_DIR"
        done
    done
done

echo "[DONE] All gemma-2-2b-it addition rerun steps complete."
