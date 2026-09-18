#!/bin/bash
# Run on your laptop when a pod reports ALL DONE. Brings back the new Boolean
# extraction metrics, refreshed cross-task matrices, logs, and per-example caches.
#   scripts/runpod/pull_results.sh root@<pod-ip> <ssh-port>
set -euo pipefail
HOST=${1:?host}; PORT=${2:-22}; WORK=${WORK:-/workspace/circuit_reuse}
cd "$(dirname "$0")/../.."
rsync -avz --progress -e "ssh -p $PORT" \
    "$HOST:$WORK/results/granularity_parity/" results/granularity_parity/
rsync -avz --progress -e "ssh -p $PORT" "$HOST:$WORK/logs/" logs/runpod/
rsync -avz --progress -e "ssh -p $PORT" --include='*boolean*' --include='*/' --exclude='*' \
    "$HOST:$WORK/cache_granularity_parity/" cache_granularity_parity/
