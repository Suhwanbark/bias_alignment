#!/bin/bash
# SimNPO training script for Qwen3-30B
#
# Usage:
#   bash simple_npo/scripts/run_train.sh                                    # default config
#   bash simple_npo/scripts/run_train.sh simple_npo/configs/qwen3_30b.yaml  # custom config
#   bash simple_npo/scripts/run_train.sh --max-steps 10                     # dry run

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR/.."

CONFIG="${1:-simple_npo/configs/qwen3_30b.yaml}"

# If first arg is a flag (starts with --), use default config
if [[ "$1" == --* ]]; then
    CONFIG="simple_npo/configs/qwen3_30b.yaml"
    python -m simple_npo.src.train --config "$CONFIG" "$@"
else
    python -m simple_npo.src.train --config "$CONFIG" "${@:2}"
fi
