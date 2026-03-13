#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT_DIR"

EXP_DIR="experiments/npo_scenario_4b_v2"
DATA_DIR="${EXP_DIR}/data"
ADAPTER_DIR="models/adapters/npo_scenario_4b_v2"
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/npo_scenario_4b_v2_train_$(date '+%Y%m%d_%H%M%S').log"

mkdir -p "$LOG_DIR"

if [[ ! -f "${DATA_DIR}/forget.jsonl" ]]; then
    echo "[ERROR] forget.jsonl not found."
    exit 1
fi

FORGET_SAMPLES=$(wc -l < "${DATA_DIR}/forget.jsonl")
echo "══════════════════════════════════════════════════════════"
echo "NPO Scenario 4B V2 Training"
echo "  Forget:   ${DATA_DIR}/forget.jsonl (${FORGET_SAMPLES} samples)"
echo "  Adapter:  ${ADAPTER_DIR}/"
echo "  Log:      ${LOG_FILE}"
echo "══════════════════════════════════════════════════════════"

python npo/train.py \
    --model-name Qwen/Qwen3-4B-Instruct-2507 \
    --forget-data "${DATA_DIR}/forget.jsonl" \
    --output-dir "$ADAPTER_DIR" \
    --beta 0.1 --lr 2e-5 --max-steps 300 --save-steps 50 \
    --batch-size 8 --grad-accum 4 --lora-r 8 --seed 42 \
    "$@" \
    2>&1 | tee "$LOG_FILE"
