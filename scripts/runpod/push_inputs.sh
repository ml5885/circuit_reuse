#!/bin/bash
# Run on your laptop. Copies the existing parity results (the other five tasks'
# extraction metrics and the cross-task matrices to be refreshed) to the pod.
#   scripts/runpod/push_inputs.sh root@<pod-ip> <ssh-port>
set -euo pipefail
HOST=${1:?host}; PORT=${2:-22}; WORK=${WORK:-/workspace/circuit_reuse}
cd "$(dirname "$0")/../.."
ssh -p "$PORT" "$HOST" "mkdir -p $WORK/results/granularity_parity"
rsync -avz --progress -e "ssh -p $PORT" \
    results/granularity_parity/granularity_parity_extraction \
    results/granularity_parity/granularity_parity_cross_task \
    "$HOST:$WORK/results/granularity_parity/"
