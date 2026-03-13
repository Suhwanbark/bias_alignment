#!/bin/bash
# ══════════════════════════════════════════════════════════════
# NPO training script
#
# Usage:
#   bash npo/scripts/run_train.sh qwen30b                  # Qwen3-30B (default)
#   bash npo/scripts/run_train.sh qwen4b                   # Qwen3-4B
#   bash npo/scripts/run_train.sh qwen4b --forget-only     # forget only (no retain)
#   bash npo/scripts/run_train.sh qwen30b --flat KL        # FLAT mode
#
# Auto-derived paths:
#   Data:     npo/data/<model_short>/forget.jsonl
#   Adapter:  models/adapters/npo_<model_short>/
#   Log:      logs/npo_<model_short>_train.log
# ══════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

# ─── Model configs ────────────────────────────────────────────
declare -A MODEL_NAMES=(
    [qwen30b]="Qwen/Qwen3-30B-A3B-Instruct-2507"
    [qwen4b]="Qwen/Qwen3-4B-Instruct-2507"
    [gemma27b]="google/gemma-3-27b-it"
    [mistral24b]="mistralai/Mistral-Small-3.1-24B-Instruct-2503"
)

declare -A MODEL_BATCH=(
    [qwen30b]=4
    [qwen4b]=8
    [gemma27b]=4
    [mistral24b]=4
)

# ─── Parse args ───────────────────────────────────────────────
MODEL_SHORT="${1:-qwen30b}"
shift || true

if [[ -z "${MODEL_NAMES[$MODEL_SHORT]+x}" ]]; then
    echo "Unknown model: $MODEL_SHORT"
    echo "Available: ${!MODEL_NAMES[*]}"
    exit 1
fi

FORGET_ONLY=false
FLAT_MODE=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --forget-only)
            FORGET_ONLY=true
            shift
            ;;
        --flat)
            FLAT_MODE="${2:-KL}"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# ─── Auto-derive paths ───────────────────────────────────────
MODEL_NAME="${MODEL_NAMES[$MODEL_SHORT]}"
BATCH_SIZE="${MODEL_BATCH[$MODEL_SHORT]}"
DATA_DIR="npo/data/${MODEL_SHORT}"
ADAPTER_DIR="models/adapters/npo_${MODEL_SHORT}"
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/npo_${MODEL_SHORT}_train.log"

mkdir -p "$LOG_DIR"

# Validate data exists
if [[ ! -f "${DATA_DIR}/forget.jsonl" ]]; then
    echo "[ERROR] forget.jsonl not found: ${DATA_DIR}/forget.jsonl"
    echo ""
    echo "Generate it first:"
    echo "  python npo/prepare_dataset.py \\"
    echo "    --csv <phase0_combined.csv> \\"
    echo "    --output-dir ${DATA_DIR}"
    exit 1
fi

# ─── Build command ────────────────────────────────────────────
COMMON_ARGS=(
    --model-name "$MODEL_NAME"
    --forget-data "${DATA_DIR}/forget.jsonl"
    --output-dir "$ADAPTER_DIR"
    --beta 0.1 --lr 2e-5 --max-steps 300 --save-steps 50
    --batch-size "$BATCH_SIZE" --grad-accum 4 --lora-r 8 --seed 42
)

if [[ -n "$FLAT_MODE" ]]; then
    # FLAT mode
    echo "Mode: FLAT (${FLAT_MODE})"
    COMMON_ARGS+=(--loss-type flat --flat-div "$FLAT_MODE")
    ADAPTER_DIR="models/adapters/npo_${MODEL_SHORT}_flat_${FLAT_MODE,,}"
    COMMON_ARGS=(
        --model-name "$MODEL_NAME"
        --forget-data "${DATA_DIR}/forget.jsonl"
        --output-dir "$ADAPTER_DIR"
        --loss-type flat --flat-div "$FLAT_MODE"
        --beta 0.1 --lr 2e-5 --max-steps 300 --save-steps 50
        --batch-size "$BATCH_SIZE" --grad-accum 4 --lora-r 8 --seed 42
    )
elif [[ "$FORGET_ONLY" == true ]]; then
    echo "Mode: forget-only"
else
    # Forget + Retain
    if [[ -f "${DATA_DIR}/retain.jsonl" ]]; then
        echo "Mode: forget + retain"
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        ADAPTER_DIR="models/adapters/npo_${MODEL_SHORT}_retain_${TIMESTAMP}"
        LOG_FILE="${LOG_DIR}/npo_${MODEL_SHORT}_retain_${TIMESTAMP}.log"
        COMMON_ARGS=(
            --model-name "$MODEL_NAME"
            --forget-data "${DATA_DIR}/forget.jsonl"
            --retain-data "${DATA_DIR}/retain.jsonl"
            --output-dir "$ADAPTER_DIR"
            --alpha-forget 1.0 --alpha-retain 1.0
            --beta 0.1 --lr 2e-5 --max-steps 300 --save-steps 50
            --batch-size "$BATCH_SIZE" --grad-accum 4 --lora-r 8 --seed 42
        )
    else
        echo "[WARN] retain.jsonl not found, falling back to forget-only"
        FORGET_ONLY=true
    fi
fi

# ─── Run ──────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════"
echo "NPO Training"
echo "  Model:   ${MODEL_SHORT} (${MODEL_NAME})"
echo "  Data:    ${DATA_DIR}/"
echo "  Adapter: ${ADAPTER_DIR}/"
echo "  Log:     ${LOG_FILE}"
echo "══════════════════════════════════════════════════════════"

python npo/train.py "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
