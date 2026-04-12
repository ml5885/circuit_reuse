#!/bin/bash
#SBATCH --job-name=q4b_exclude
#SBATCH --output=/home/ml6/logs/sbatch/q4b_exclude_%j.out
#SBATCH --error=/home/ml6/logs/sbatch/q4b_exclude_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=96G
#SBATCH --mail-user=ml6@andrew.cmu.edu
#SBATCH --mail-type=START,END,FAIL
#SBATCH --partition=general

# Fill in missing Qwen3-4B c3_exclude results for boolean and ioi tasks.

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

MODEL="qwen3-4b"
NUM_EXAMPLES=1000
top_k_list="1,5,10,20,30"
reuse_thresholds="95,96,97,98,99,100"
RUN_NAME="cross_task"

for TASK in boolean ioi; do
    echo "[EXTRACT] MODEL=$MODEL TASK=$TASK"
    python main_experiment.py \
        --model_name "$MODEL" \
        --task "$TASK" \
        --top_k_list "$top_k_list" \
        --reuse-thresholds "$reuse_thresholds" \
        --perm-trials 5000 \
        --num_examples "$NUM_EXAMPLES" \
        --dtype bf16 \
        --amp \
        --device cuda \
        --run-name "$RUN_NAME" \
        --output-dir "$DATA_DIR/results" \
        --cache-dir "$CACHE_DIR" \
        --results-home /home/ml6/circuit_reuse/results \
        --force-extract
done

echo "[DONE] Missing Qwen3-4B exclude tasks complete"
