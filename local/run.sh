#!/bin/bash

# ══════════════════════════════════════════════════════════════
# vLLM 기반 Bias Experiment Runner
# ══════════════════════════════════════════════════════════════

# ────────────── Configuration ──────────────
# Available Models (uncomment one):
# MODEL_ID="Qwen/Qwen3-30B-A3B-Instruct-2507"
# MODEL_ID="zai-org/GLM-4.7-Flash"
MODEL_ID="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

TEMPERATURE=0.6          # 기존 OpenRouter 세팅과 동일
MAX_TOKENS=2048
TOP_P=0.8
VLLM_PORT=8000
VLLM_URL="http://localhost:${VLLM_PORT}/v1"
OUTPUT_DIR="./result"
NUM_SETS=3
NUM_TRIALS=10
MAX_WORKERS=300

# ════════════════════════════════════════════
# Step 1: vLLM 서버 실행 (별도 터미널에서)
# ════════════════════════════════════════════
#
# vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 \
#     --tensor-parallel-size 4 \
#     --port 8000 \
#     --max-model-len 4096
#
# ════════════════════════════════════════════

# ────────────── Run Experiment ──────────────
echo "Starting vLLM bias experiment..."
echo "Model: ${MODEL_ID}"
echo "Temperature: ${TEMPERATURE}"
echo "vLLM URL: ${VLLM_URL}"
echo "Output directory: ${OUTPUT_DIR}"

python bias_attribute.py \
    --model-id "${MODEL_ID}" \
    --temperature ${TEMPERATURE} \
    --max-tokens ${MAX_TOKENS} \
    --top-p ${TOP_P} \
    --vllm-url "${VLLM_URL}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-sets ${NUM_SETS} \
    --num-trials ${NUM_TRIALS} \
    --max-workers ${MAX_WORKERS}

# ────────────── Aggregate Results ──────────────
echo ""
echo "Aggregating results..."

python result_attribute.py \
    --model-id "${MODEL_ID}" \
    --output-dir "${OUTPUT_DIR}"

echo ""
echo "Done!"
