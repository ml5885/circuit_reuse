#!/bin/bash
# Emails if a log file stops growing for STALL_MIN minutes (default 45), then keeps watching.
LOG=$1; STAGE=$2; STALL_MIN=${STALL_MIN:-45}
last=-1; quiet=0
while sleep 300; do
    size=$(stat -c %s "$LOG" 2> /dev/null || echo 0)
    if [[ $size -eq $last ]]; then quiet=$((quiet + 5)); else quiet=0; fi
    last=$size
    if [[ $quiet -ge $STALL_MIN ]]; then
        python "$(dirname "$0")/notify.py" "STALLED $STAGE ($quiet min without output)" <<< "$(tail -n 30 "$LOG")"
        quiet=0
    fi
done
