#!/bin/bash
# One-time pod setup. Run from anywhere after `source scripts/runpod/env.sh`.
# Assumes a RunPod PyTorch image (torch + CUDA already installed).
set -euo pipefail
: "${WORK:?source scripts/runpod/env.sh first}"
: "${HF_TOKEN:?HF_TOKEN not set}"

apt-get update -qq && apt-get install -y -qq rsync tmux > /dev/null
mkdir -p "$HF_HOME" "$WORK/logs"
cd "$WORK"
pip install -q -r scripts/runpod/requirements.txt -e .
python -c "import huggingface_hub as h; h.login(token='$HF_TOKEN', add_to_git_credential=False)"

# Pull everything the runs will need so a missing gate or a bad token fails now, not two hours in.
python - <<'PY'
from datasets import load_dataset
from huggingface_hub import snapshot_download
for name in ["mib-bench/ioi", "mib-bench/copycolors_mcqa", "mib-bench/arc_easy", "mib-bench/arc_challenge"]:
    load_dataset(name, split="test", **({"name": "4_answer_choices"} if "copycolors" in name else {}))
for repo in ["google/gemma-2-2b", "google/gemma-2-2b-it", "meta-llama/Llama-3.2-3B",
             "meta-llama/Llama-3.2-3B-Instruct", "Qwen/Qwen3-4B"]:
    snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"])
print("prefetch ok")
PY
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0), torch.__version__)"
python scripts/runpod/notify.py "setup complete" <<< "$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
