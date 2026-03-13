#!/bin/bash
# ══════════════════════════════════════════════════════════════
# LlamaFactory 훈련 래퍼 스크립트
#
# eval_steps와 save_steps를 데이터셋 크기에 맞춰 동적으로 계산:
#   eval_steps = save_steps = max(steps_per_epoch // 2, 1)
#
# 사용법:
#   bash run_train.sh configs/sft_v7.yaml
#   bash run_train.sh configs/dpo_qwen.yaml
# ══════════════════════════════════════════════════════════════
set -euo pipefail

CONFIG="$1"
if [ ! -f "$CONFIG" ]; then
    echo "[ERROR] Config file not found: $CONFIG"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ─── Parse YAML values ──────────────────────────────────────
get_yaml_value() {
    local key="$1"
    local file="$2"
    grep "^${key}:" "$file" | head -1 | sed "s/^${key}:[[:space:]]*//" | sed 's/[[:space:]]*$//'
}

DATASET_NAME=$(get_yaml_value "dataset" "$CONFIG")
DATASET_DIR=$(get_yaml_value "dataset_dir" "$CONFIG")
EPOCHS=$(get_yaml_value "num_train_epochs" "$CONFIG")
BATCH_SIZE=$(get_yaml_value "per_device_train_batch_size" "$CONFIG")
GRAD_ACCUM=$(get_yaml_value "gradient_accumulation_steps" "$CONFIG")
VAL_SIZE=$(get_yaml_value "val_size" "$CONFIG")

# Default values
DATASET_DIR="${DATASET_DIR:-$SCRIPT_DIR}"
EPOCHS="${EPOCHS:-10.0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
VAL_SIZE="${VAL_SIZE:-0.1}"

# ─── Get data file path from dataset_info.json ──────────────
DATASET_INFO="${DATASET_DIR}/dataset_info.json"
if [ ! -f "$DATASET_INFO" ]; then
    echo "[ERROR] dataset_info.json not found: $DATASET_INFO"
    exit 1
fi

DATA_FILE=$(python3 -c "
import json, sys, os
with open(sys.argv[1]) as f:
    info = json.load(f)
ds = sys.argv[2]
ds_dir = sys.argv[3]
if ds not in info:
    print(f'ERROR: dataset \"{ds}\" not found in dataset_info.json', file=sys.stderr)
    sys.exit(1)
path = info[ds]['file_name']
if not os.path.isabs(path):
    path = os.path.join(ds_dir, path)
print(path)
" "$DATASET_INFO" "$DATASET_NAME" "$DATASET_DIR")

if [ ! -f "$DATA_FILE" ]; then
    echo "[ERROR] Data file not found: $DATA_FILE"
    exit 1
fi

# ─── Calculate eval_steps ───────────────────────────────────
TOTAL_SAMPLES=$(wc -l < "$DATA_FILE")
TRAIN_SAMPLES=$(python3 -c "import math; print(math.ceil($TOTAL_SAMPLES * (1 - $VAL_SIZE)))")

# Detect number of GPUs
NUM_GPUS=$(python3 -c "
import torch
n = torch.cuda.device_count() if torch.cuda.is_available() else 1
print(n)
" 2>/dev/null || echo "1")

EFFECTIVE_BATCH=$((BATCH_SIZE * GRAD_ACCUM * NUM_GPUS))
STEPS_PER_EPOCH=$((TRAIN_SAMPLES / EFFECTIVE_BATCH))
if [ "$STEPS_PER_EPOCH" -lt 1 ]; then
    STEPS_PER_EPOCH=1
fi

# Integer part of epochs
EPOCHS_INT=$(python3 -c "print(int($EPOCHS))")

EVAL_STEPS=$((STEPS_PER_EPOCH / 2))
if [ "$EVAL_STEPS" -lt 1 ]; then
    EVAL_STEPS=1
fi

echo "══════════════════════════════════════════════════════════════"
echo "LlamaFactory Training Wrapper"
echo "══════════════════════════════════════════════════════════════"
echo "  Config:          $CONFIG"
echo "  Dataset:         $DATASET_NAME"
echo "  Data file:       $DATA_FILE"
echo "  Total samples:   $TOTAL_SAMPLES"
echo "  Train samples:   $TRAIN_SAMPLES"
echo "  GPUs:            $NUM_GPUS"
echo "  Effective batch: $EFFECTIVE_BATCH"
echo "  Steps/epoch:     $STEPS_PER_EPOCH"
echo "  Epochs:          $EPOCHS"
echo "  eval_steps:      $EVAL_STEPS"
echo "══════════════════════════════════════════════════════════════"

# ─── Create temporary config with computed values ────────────
TMP_CONFIG=$(mktemp /tmp/lf_train_XXXXXX.yaml)
cp "$CONFIG" "$TMP_CONFIG"

# Replace save_steps and eval_steps
sed -i "s/^save_steps:.*/save_steps: ${EVAL_STEPS}/" "$TMP_CONFIG"
sed -i "s/^eval_steps:.*/eval_steps: ${EVAL_STEPS}/" "$TMP_CONFIG"

echo ""
echo "Starting training..."
echo ""

# ─── Run training ───────────────────────────────────────────
DISABLE_VERSION_CHECK=1 llamafactory-cli train "$TMP_CONFIG"
STATUS=$?

rm -f "$TMP_CONFIG"

if [ $STATUS -ne 0 ]; then
    echo "[ERROR] Training failed (exit: $STATUS)"
    exit $STATUS
fi

echo ""
echo "[OK] Training completed successfully"
