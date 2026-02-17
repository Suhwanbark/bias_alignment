#!/bin/bash
# ══════════════════════════════════════════════════════════════
# Bias Index 측정 스크립트 (vLLM 로컬 서버용)
# ══════════════════════════════════════════════════════════════
#
# vLLM 서버에 올린 모델의 Bias Index를 측정한다.
# bias_attribute.py → result_attribute.py 순서로 실행.
#
# 사용법:
#   ./run_bias.sh "openai/gpt-oss-20b"
#   ./run_bias.sh "Qwen/Qwen3-30B-A3B-Instruct-2507"
#   ./run_bias.sh "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
#   ./run_bias.sh "zai-org/GLM-4.7-Flash"
#   ./run_bias.sh "mistralai/Ministral-3-14B-Instruct-2512"
#   ./run_bias.sh "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
#   ./run_bias.sh "google/gemma-3-27b-it"
#   ./run_bias.sh "microsoft/phi-4"
#
# 옵션:
#   ./run_bias.sh "모델ID" [NUM_SETS] [NUM_TRIALS] [MAX_WORKERS]
#
# 예시:
#   ./run_bias.sh "openai/gpt-oss-20b"              # 기본값: 3 sets, 10 trials, 200 workers
#   ./run_bias.sh "openai/gpt-oss-20b" 3 20 300     # 3 sets, 20 trials, 300 workers
#
# ══════════════════════════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── 인자 파싱 ──
MODEL_ID="${1:?Usage: ./run_bias.sh \"model-id\" [num_sets] [num_trials] [max_workers]}"
NUM_SETS="${2:-3}"
NUM_TRIALS="${3:-10}"
MAX_WORKERS="${4:-200}"
VLLM_URL="http://localhost:8000/v1"
OUTPUT_DIR="./result"

MODEL_SHORT=$(echo "$MODEL_ID" | awk -F'/' '{print $NF}')

echo "══════════════════════════════════════════════════════════"
echo "  Bias Index Measurement"
echo "══════════════════════════════════════════════════════════"
echo "  Model:       $MODEL_ID"
echo "  Short name:  $MODEL_SHORT"
echo "  Sets:        $NUM_SETS"
echo "  Trials:      $NUM_TRIALS"
echo "  Workers:     $MAX_WORKERS"
echo "  vLLM URL:    $VLLM_URL"
echo "  Output:      $OUTPUT_DIR"
echo "══════════════════════════════════════════════════════════"
echo ""

# ── vLLM 서버 확인 ──
echo "[1/3] Checking vLLM server..."
if curl -s "$VLLM_URL/../health" > /dev/null 2>&1 || curl -s "http://localhost:8000/health" > /dev/null 2>&1; then
    echo "  ✓ vLLM server is running"
    SERVING_MODEL=$(curl -s http://localhost:8000/v1/models | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null || echo "unknown")
    echo "  Serving model: $SERVING_MODEL"
else
    echo "  ✗ vLLM server is not running!"
    echo "  Start it first: cd ../debias && ./vllm <model>"
    exit 1
fi
echo ""

# ── Step 1: bias_attribute.py 실행 ──
echo "[2/3] Running bias_attribute.py..."
echo "  Generating $NUM_SETS sets × $NUM_TRIALS trials per ticker..."
echo ""

python3 bias_attribute.py \
    --model-id "$MODEL_ID" \
    --num-sets "$NUM_SETS" \
    --num-trials "$NUM_TRIALS" \
    --max-workers "$MAX_WORKERS" \
    --vllm-url "$VLLM_URL" \
    --output-dir "$OUTPUT_DIR"

echo ""

# ── Step 2: result_attribute.py 실행 ──
echo "[3/3] Running result_attribute.py..."
echo "  Computing Bias Index..."
echo ""

python3 result_attribute.py \
    --model-id "$MODEL_ID" \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  Done! Results:"
echo "  - CSV:  $OUTPUT_DIR/${MODEL_SHORT}_att_combined.csv"
echo "  - JSON: $OUTPUT_DIR/${MODEL_SHORT}_att_result.json"
echo "══════════════════════════════════════════════════════════"
