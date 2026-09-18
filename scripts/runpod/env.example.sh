# Copy to env.sh on the pod, fill in, then `source scripts/runpod/env.sh`.
# env.sh is gitignored.
export HF_TOKEN=hf_...                     # needs access to google/gemma-2-2b and meta-llama/Llama-3.2-3B
export NOTIFY_EMAIL_TO=you@example.com
export NOTIFY_SMTP_USER=you@gmail.com
export NOTIFY_SMTP_PASS=xxxx-xxxx-xxxx-xxxx   # Gmail app password (Google account -> Security -> App passwords)
# export NOTIFY_NTFY_TOPIC=circuit-reuse-runs  # optional phone push via the ntfy app

export WORK=${WORK:-/workspace/circuit_reuse}   # repo checkout; /workspace persists across pod restarts on RunPod
export HF_HOME=${HF_HOME:-/workspace/hf_cache}
