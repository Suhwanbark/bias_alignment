#!/bin/bash
# NPO training script
#
# Usage:
#   bash npo/run_train.sh                  # forget + retain (default)
#   bash npo/run_train.sh --forget-only    # forget only (no retain loss)
#
# To adjust loss weights:
#   Edit --alpha-forget / --alpha-retain below (default: 1.0 each)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

FORGET_ONLY=false
if [[ "$1" == "--forget-only" ]]; then
    FORGET_ONLY=true
fi

COMMON_ARGS=(
    --model-name Qwen/Qwen3-30B-A3B-Instruct-2507
    --forget-data npo/data/qwen3-30b-a3b-instruct-2507/forget.jsonl
    --beta 0.1 --lr 2e-5 --max-steps 500 --save-steps 50
    --batch-size 4 --grad-accum 4 --lora-r 8 --seed 42
)

if [[ "$FORGET_ONLY" == true ]]; then
    python npo/train.py "${COMMON_ARGS[@]}" \
        --output-dir /home/jovyan/models/qwen-npo-forget
else
    python npo/train.py "${COMMON_ARGS[@]}" \
        --retain-data npo/data/qwen3-30b-a3b-instruct-2507/retain.jsonl \
        --output-dir /home/jovyan/models/qwen-npo-retain \
        --alpha-forget 1.0 --alpha-retain 1.0
fi
