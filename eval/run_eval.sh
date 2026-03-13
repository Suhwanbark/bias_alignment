#!/bin/bash
# ══════════════════════════════════════════════════════════════
# Unified Bias Evaluation Script
#
# Usage:
#   bash eval/run_eval.sh --base qwen          # Base model bias score
#   bash eval/run_eval.sh --base gemma         # Gemma-3-27B base
#   bash eval/run_eval.sh --model /path/to/model  # Already merged model
#   bash eval/run_eval.sh --adapter /path/to/checkpoint --save-merged  # LoRA -> merge -> eval, keep merged
#   bash eval/run_eval.sh --adapter /path/to/checkpoint  # LoRA -> merge -> eval, delete merged
#
# Options:
#   --output DIR     Result output directory (default: auto from method)
#   --num-sets N     Number of sets (default: 3)
#   --num-trials N   Number of trials per ticker (default: 10)
#   --max-workers N  Concurrent workers (default: 200)
#   --tag TAG        Tag for result files
#   --tp N           Tensor parallel size (default: 2)
#
# ══════════════════════════════════════════════════════════════
set -euo pipefail

# ─── Path Setup ────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_DIR="/home/jovyan/models"
MERGE_DIR="${MODEL_DIR}/_tmp_merged"
MERGED_SAVE_DIR="${ROOT_DIR}/models/merged"

VLLM_PORT=8000

LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/eval_$(date '+%Y%m%d_%H%M%S').log"

# ─── Model Alias Mapping ──────────────────────────────────────
select_model() {
    case "$1" in
        gpt|gp|gpt-oss|openai)
            echo "openai/gpt-oss-20b"
            ;;
        qwen|Qwen|QWEN)
            echo "Qwen/Qwen3-30B-A3B-Instruct-2507"
            ;;
        nemotron|Nemotron|NEMOTRON|nvidia)
            echo "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
            ;;
        glm|GLM|glm4)
            echo "zai-org/GLM-4.7-Flash"
            ;;
        ministral|Ministral|ministral-14b)
            echo "mistralai/Ministral-3-14B-Instruct-2512"
            ;;
        mistral-small|mistral-small-3.1|ms)
            echo "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
            ;;
        gemma|gemma3|Gemma)
            echo "google/gemma-3-27b-it"
            ;;
        phi|phi4|phi-4)
            echo "microsoft/phi-4"
            ;;
        qwen-4b|qwen4b)
            echo "Qwen/Qwen3-4B-Instruct-2507"
            ;;
        *)
            echo ""
            ;;
    esac
}

# ─── Utilities ─────────────────────────────────────────────────
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg"
    echo "$msg" >> "$LOG_FILE"
}

kill_gpu_processes() {
    log "Cleaning up GPU processes..."
    pkill -f "vllm.entrypoints" 2>/dev/null || true
    sleep 5
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ -n "$used" ] && [ "$used" -gt 5000 ] 2>/dev/null; then
        log "  GPU memory ${used}MiB in use, force cleaning..."
        nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | \
            xargs -r kill -9 2>/dev/null || true
        sleep 10
    fi
    log "  GPU cleanup done"
}

wait_vllm_ready() {
    local port="$1"
    local max_wait="${2:-300}"
    local pid="${3:-}"
    local elapsed=0
    while [ "$elapsed" -lt "$max_wait" ]; do
        if curl -s "http://localhost:${port}/v1/models" > /dev/null 2>&1; then
            return 0
        fi
        if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
            return 1
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done
    return 1
}

wait_gpu_free() {
    local gpu_id="$1"
    local max_wait=30
    local elapsed=0
    while [ $elapsed -lt $max_wait ]; do
        local used
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null)
        if [ -n "$used" ] && [ "$used" -lt 5000 ] 2>/dev/null; then
            return 0
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done
    log "  [WARN] GPU ${gpu_id} memory release timeout"
    return 1
}

stop_vllm() {
    local pid="${1:-}"
    if [ -n "$pid" ]; then
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi
    pkill -f "vllm.entrypoints.*${VLLM_PORT}" 2>/dev/null || true
    sleep 5
    wait_gpu_free 0 || true
    if [ "$TP" -ge 2 ]; then
        wait_gpu_free 1 || true
    fi
}

usage() {
    echo "Usage:"
    echo "  bash eval/run_eval.sh --base <alias>                 # Base model evaluation"
    echo "  bash eval/run_eval.sh --model /path/to/model         # Pre-merged model evaluation"
    echo "  bash eval/run_eval.sh --adapter /path/to/checkpoint  # LoRA adapter evaluation"
    echo ""
    echo "Model aliases: qwen, gemma, phi, gpt, nemotron, glm, ministral, mistral-small, qwen-4b"
    echo ""
    echo "Options:"
    echo "  --output DIR       Result output directory (default: auto)"
    echo "  --num-sets N       Number of sets (default: 3)"
    echo "  --num-trials N     Number of trials per ticker (default: 10)"
    echo "  --max-workers N    Concurrent workers (default: 200)"
    echo "  --tag TAG          Tag for result files (default: auto)"
    echo "  --tp N             Tensor parallel size (default: 2)"
    echo "  --save-merged      Keep merged model (only with --adapter)"
    echo "  --merge-template F Path to merge YAML template (default: sft/configs/merge.yaml)"
    exit 1
}

# ─── Argument Parsing ─────────────────────────────────────────
MODE=""
BASE_ALIAS=""
MODEL_PATH=""
ADAPTER_PATH=""
OUTPUT_DIR=""
NUM_SETS=3
NUM_TRIALS=10
MAX_WORKERS=200
TAG=""
TP=2
SAVE_MERGED=false
MERGE_TEMPLATE="${ROOT_DIR}/sft/configs/merge.yaml"

while [ $# -gt 0 ]; do
    case "$1" in
        --base)
            MODE="base"
            BASE_ALIAS="$2"
            shift 2
            ;;
        --model)
            MODE="model"
            MODEL_PATH="$2"
            shift 2
            ;;
        --adapter)
            MODE="adapter"
            ADAPTER_PATH="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --num-sets)
            NUM_SETS="$2"
            shift 2
            ;;
        --num-trials)
            NUM_TRIALS="$2"
            shift 2
            ;;
        --max-workers)
            MAX_WORKERS="$2"
            shift 2
            ;;
        --tag)
            TAG="$2"
            shift 2
            ;;
        --tp)
            TP="$2"
            shift 2
            ;;
        --save-merged)
            SAVE_MERGED=true
            shift
            ;;
        --merge-template)
            MERGE_TEMPLATE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "[ERROR] Unknown option: $1"
            usage
            ;;
    esac
done

if [ -z "$MODE" ]; then
    echo "[ERROR] Must specify one of: --base, --model, --adapter"
    usage
fi

# ─── Resolve Model ID and Paths ──────────────────────────────
VLLM_MODEL_PATH=""
MODEL_ID=""
NEEDS_MERGE=false
NEEDS_CLEANUP=false

case "$MODE" in
    base)
        MODEL_ID=$(select_model "$BASE_ALIAS")
        if [ -z "$MODEL_ID" ]; then
            echo "[ERROR] Unknown model alias: ${BASE_ALIAS}"
            echo "Available: qwen, gemma, phi, gpt, nemotron, glm, ministral, mistral-small, qwen-4b"
            exit 1
        fi
        VLLM_MODEL_PATH="$MODEL_ID"
        [ -z "$TAG" ] && TAG="baseline"
        [ -z "$OUTPUT_DIR" ] && OUTPUT_DIR="${ROOT_DIR}/eval/results/${BASE_ALIAS}_baseline"
        # Auto-detect TP for small models
        if echo "$MODEL_ID" | grep -qi "4B\|14B\|phi-4"; then
            TP=1
        fi
        ;;
    model)
        if [ ! -d "$MODEL_PATH" ] && [ ! -f "${MODEL_PATH}/config.json" ]; then
            # Could be a HuggingFace model ID
            MODEL_ID="$MODEL_PATH"
            VLLM_MODEL_PATH="$MODEL_PATH"
        else
            MODEL_ID="$MODEL_PATH"
            VLLM_MODEL_PATH="$MODEL_PATH"
        fi
        [ -z "$TAG" ] && TAG="merged"
        if [ -z "$OUTPUT_DIR" ]; then
            local_name=$(basename "$MODEL_PATH")
            OUTPUT_DIR="${ROOT_DIR}/eval/results/${local_name}"
        fi
        ;;
    adapter)
        if [ ! -d "$ADAPTER_PATH" ]; then
            echo "[ERROR] Adapter path not found: ${ADAPTER_PATH}"
            exit 1
        fi
        if [ ! -f "${ADAPTER_PATH}/adapter_model.safetensors" ] && [ ! -f "${ADAPTER_PATH}/adapter_model.bin" ]; then
            echo "[ERROR] No adapter file found in: ${ADAPTER_PATH}"
            exit 1
        fi
        NEEDS_MERGE=true
        # Detect base model from merge template
        MODEL_ID=$(grep "^model_name_or_path:" "$MERGE_TEMPLATE" | head -1 | sed 's/^model_name_or_path:[[:space:]]*//' | sed 's/[[:space:]]*$//')
        VLLM_MODEL_PATH="$MERGE_DIR"
        if [ "$SAVE_MERGED" = true ]; then
            NEEDS_CLEANUP=false
        else
            NEEDS_CLEANUP=true
        fi
        [ -z "$TAG" ] && TAG="adapter"
        if [ -z "$OUTPUT_DIR" ]; then
            local_name=$(basename "$ADAPTER_PATH")
            parent_name=$(basename "$(dirname "$ADAPTER_PATH")")
            OUTPUT_DIR="${ROOT_DIR}/eval/results/${parent_name}/${local_name}"
        fi
        # Auto-detect TP for 4B models
        if echo "$MODEL_ID" | grep -qi "4B"; then
            TP=1
        fi
        ;;
esac

mkdir -p "$OUTPUT_DIR"

# ─── Print Configuration ─────────────────────────────────────
log "══════════════════════════════════════════════════════════════"
log "Bias Evaluation"
log "  Mode:        ${MODE}"
log "  Model ID:    ${MODEL_ID}"
log "  TP:          ${TP}"
log "  Num sets:    ${NUM_SETS}"
log "  Num trials:  ${NUM_TRIALS}"
log "  Max workers: ${MAX_WORKERS}"
log "  Tag:         ${TAG}"
log "  Output:      ${OUTPUT_DIR}"
log "  Log:         ${LOG_FILE}"
if [ "$MODE" = "adapter" ]; then
    log "  Adapter:     ${ADAPTER_PATH}"
    log "  Merge tmpl:  ${MERGE_TEMPLATE}"
    log "  Save merged: ${SAVE_MERGED}"
fi
log "══════════════════════════════════════════════════════════════"

# ═══════════════════════════════════════════════════════════════
# Step 1: LoRA Merge (adapter mode only)
# ═══════════════════════════════════════════════════════════════
if [ "$NEEDS_MERGE" = true ]; then
    log "[1/4] Merging LoRA adapter..."
    rm -rf "$MERGE_DIR"

    merge_config=$(mktemp /tmp/lf_merge_XXXXXX.yaml)
    sed -e "s|ADAPTER_PATH|${ADAPTER_PATH}|g" \
        -e "s|EXPORT_DIR|${MERGE_DIR}|g" \
        "$MERGE_TEMPLATE" > "$merge_config"

    DISABLE_VERSION_CHECK=1 llamafactory-cli export "$merge_config" 2>&1 | tee -a "$LOG_FILE"
    rm -f "$merge_config"

    if [ ! -d "$MERGE_DIR" ] || [ ! -f "${MERGE_DIR}/config.json" ]; then
        log "[ERROR] LoRA merge failed"
        exit 1
    fi
    log "  Merge complete: ${MERGE_DIR}"
else
    log "[1/4] Skip merge (not adapter mode)"
fi

# ═══════════════════════════════════════════════════════════════
# Step 2: Start vLLM Server
# ═══════════════════════════════════════════════════════════════
log "[2/4] Starting vLLM server (TP=${TP}, port ${VLLM_PORT})..."
kill_gpu_processes

VLLM_ARGS=(
    --model "$VLLM_MODEL_PATH"
    --port "$VLLM_PORT"
    --tensor-parallel-size "$TP"
    --trust-remote-code
    --dtype bfloat16
    --gpu-memory-utilization 0.90
)

# For HuggingFace models that need downloading, add download dir
if [ "$MODE" = "base" ]; then
    VLLM_ARGS+=(--download-dir "$MODEL_DIR")
fi

python -m vllm.entrypoints.openai.api_server "${VLLM_ARGS[@]}" \
    > /tmp/vllm_eval.log 2>&1 &
VLLM_PID=$!

if ! wait_vllm_ready "$VLLM_PORT" 300 "$VLLM_PID"; then
    log "[ERROR] vLLM failed to start (5 min timeout)"
    log "  Check /tmp/vllm_eval.log for details"
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
    if [ "$NEEDS_CLEANUP" = true ]; then
        rm -rf "$MERGE_DIR"
    fi
    exit 1
fi
log "  vLLM ready (PID: ${VLLM_PID})"

# ═══════════════════════════════════════════════════════════════
# Step 3: Run Bias Evaluation
# ═══════════════════════════════════════════════════════════════
log "[3/4] Running bias evaluation (${NUM_SETS} sets x ${NUM_TRIALS} trials)..."

# Use the model ID that vLLM is serving. For local models, pass the path.
EVAL_MODEL_ID="$VLLM_MODEL_PATH"

cd "$SCRIPT_DIR"
python bias_attribute.py \
    --model-id "$EVAL_MODEL_ID" \
    --vllm-url "http://localhost:${VLLM_PORT}/v1" \
    --temperature 0.6 \
    --seed 42 \
    --num-sets "$NUM_SETS" \
    --num-trials "$NUM_TRIALS" \
    --max-workers "$MAX_WORKERS" \
    --output-dir "$OUTPUT_DIR" \
    --tag "$TAG" 2>&1 | tee -a "$LOG_FILE"
cd "$ROOT_DIR"

# ═══════════════════════════════════════════════════════════════
# Step 4: Cleanup and Report
# ═══════════════════════════════════════════════════════════════
log "[4/4] Cleaning up..."

# Stop vLLM
stop_vllm "$VLLM_PID"

# Handle merged model
if [ "$NEEDS_MERGE" = true ]; then
    if [ "$SAVE_MERGED" = true ]; then
        local_name=$(basename "$ADAPTER_PATH")
        parent_name=$(basename "$(dirname "$ADAPTER_PATH")")
        save_path="${MERGED_SAVE_DIR}/${parent_name}_${local_name}"
        mkdir -p "$MERGED_SAVE_DIR"
        if [ -d "$MERGE_DIR" ]; then
            mv "$MERGE_DIR" "$save_path"
            log "  Merged model saved to: ${save_path}"
        fi
    elif [ "$NEEDS_CLEANUP" = true ]; then
        rm -rf "$MERGE_DIR"
        log "  Temporary merged model deleted"
    fi
fi

# ─── Extract and Print Bias Index ─────────────────────────────
log ""
log "══════════════════════════════════════════════════════════════"
log "Results"
log "══════════════════════════════════════════════════════════════"

# Find the result JSON in the output directory (may be in a subdirectory)
RESULT_JSON=$(find "$OUTPUT_DIR" -name "*_att_result.json" -type f 2>/dev/null | sort -r | head -1)

if [ -n "$RESULT_JSON" ] && [ -f "$RESULT_JSON" ]; then
    BIAS_INDEX=$(python3 -c "
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    bi = d['bias_index']
    print(f'Bias Index: {bi}')
    print()
    print('Sector Stats:')
    for s, v in d.get('sector_stats', {}).items():
        print(f'  {s:30s}  mean={v[\"bias_mean\"]:6.0f}  std={v[\"bias_std\"]:5.0f}')
    print()
    print('Size Stats:')
    for s, v in d.get('size_stats', {}).items():
        print(f'  {s:30s}  mean={v[\"bias_mean\"]:6.0f}  std={v[\"bias_std\"]:5.0f}')
    if 't_test_results' in d:
        print()
        print('T-test Results:')
        for k, v in d['t_test_results'].items():
            print(f'  {k}: {v[\"high_bias_group\"]} vs {v[\"low_bias_group\"]}  '
                  f'diff={v[\"mean_diff\"]:.2f}  t={v[\"t_statistic\"]:.2f}  p={v[\"p_value\"]:.4e}')
except Exception as e:
    print(f'Error parsing result: {e}')
" "$RESULT_JSON" 2>&1)
    log "$BIAS_INDEX"
    log ""
    log "Result JSON: ${RESULT_JSON}"
else
    log "[WARN] No result JSON found in ${OUTPUT_DIR}"
fi

log "Output dir: ${OUTPUT_DIR}"
log "Log file:   ${LOG_FILE}"
log ""
log "Done: $(date)"
