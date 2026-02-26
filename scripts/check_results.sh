#!/bin/bash

DATA_DIR=/data/user_data/ml6/circuit_reuse/results
SEARCH_DIR="${1:-$DATA_DIR}"

echo "Searching: $SEARCH_DIR"
echo ""

find "$SEARCH_DIR" -name "metrics.json" -exec \
    python3 -c "
import json, sys
for path in sys.argv[1:]:
    with open(path) as f:
        d = json.load(f)
    print(f\"{d['model_name']:40s} {d['task']:20s} {path}\")
" {} + | sort

echo ""
echo "Total: $(find "$SEARCH_DIR" -name "metrics.json" | wc -l) metrics files"
