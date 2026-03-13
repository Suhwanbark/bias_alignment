#!/bin/bash
# BiasUnlearn NPO training script for Qwen3-30B
#
# Usage:
#   bash new_npo/scripts/run_train.sh                                  # default config
#   bash new_npo/scripts/run_train.sh new_npo/configs/qwen3_30b.yaml   # custom config
#   bash new_npo/scripts/run_train.sh --max-steps 10                   # dry run

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR/.."

CONFIG="${1:-new_npo/configs/qwen3_30b.yaml}"

# If first arg is a flag (starts with --), use default config
if [[ "$1" == --* ]]; then
    CONFIG="new_npo/configs/qwen3_30b.yaml"
    python -m new_npo.src.train --config "$CONFIG" "$@"
else
    python -m new_npo.src.train --config "$CONFIG" "${@:2}"
fi
