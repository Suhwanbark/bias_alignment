#!/bin/bash
# ══════════════════════════════════════════════════════════════
# NPO 4B 체크포인트 평가 스크립트
#
# Usage:
#   bash npo/scripts/run_eval_4b.sh models/adapters/npo_qwen4b_v2 npo_qwen4b_v2
# ══════════════════════════════════════════════════════════════
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
EVAL_DIR="${ROOT_DIR}/eval"

MODEL_DIR="${ROOT_DIR}/${1:?Usage: $0 <adapter_dir> <result_name>}"
RESULT_NAME="${2:?Usage: $0 <adapter_dir> <result_name>}"
MERGE_TEMPLATE="${ROOT_DIR}/sft/configs/merge_4b.yaml"
MERGE_DIR="/tmp/_tmp_merged_4b"
RESULT_DIR="${ROOT_DIR}/npo/results/${RESULT_NAME}"
VLLM_PORT=8001
VLLM_TP=1
GPU_ID=1

LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${RESULT_NAME}_eval_$(date '+%Y%m%d_%H%M%S').log"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg"
    echo "$msg" >> "$LOG_FILE"
}

wait_vllm_ready() {
    local port="$1" max_wait="${2:-300}" pid="$3" elapsed=0
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

cleanup_vllm() {
    pkill -f "vllm.entrypoints.*${VLLM_PORT}" 2>/dev/null || true
    sleep 5
}

eval_checkpoint() {
    local ckpt_path="$1" step="$2"
    local summary_file="${RESULT_DIR}/summary.log"
    local step_dir="${RESULT_DIR}/step_${step}"

    # Skip if already done
    if [ -d "$step_dir" ]; then
        local existing_json
        existing_json=$(ls "$step_dir"/*_att_result.json 2>/dev/null | head -1)
        if [ -n "$existing_json" ] && [ -f "$existing_json" ]; then
            local bi
            bi=$(python3 -c "
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    print(d['bias_index'])
except: print('', end='')
" "$existing_json" 2>/dev/null)
            if [ -n "$bi" ]; then
                log "    step ${step}: already done (bias_index=${bi}), skip"
                echo "step=${step}  bias_index=${bi}  status=cached" >> "$summary_file"
                return 0
            fi
        fi
    fi

    log "    ── step ${step} ──"

    # 1. LoRA merge (PEFT merge_and_unload)
    log "    [1/4] LoRA merge..."
    rm -rf "$MERGE_DIR"

    python3 -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = 'Qwen/Qwen3-4B-Instruct-2507'
adapter = '${ckpt_path}'
output = '${MERGE_DIR}'

tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map='cpu', trust_remote_code=True)
model = PeftModel.from_pretrained(model, adapter)
model = model.merge_and_unload()
model.save_pretrained(output, max_shard_size='5GB')
tokenizer.save_pretrained(output)
print('Merge complete:', output)
" 2>&1 | tail -3

    if [ ! -f "${MERGE_DIR}/config.json" ]; then
        log "    [ERROR] Merge failed"
        echo "step=${step}  bias_index=MERGE_ERROR  status=error" >> "$summary_file"
        rm -rf "$MERGE_DIR"
        return 1
    fi
    log "    Merge done"

    # 2. vLLM serve (GPU 1 only)
    log "    [2/4] vLLM serve (TP=${VLLM_TP}, GPU=${GPU_ID}, port=${VLLM_PORT})..."
    cleanup_vllm

    CONDA_LIB="$(python3 -c 'import sys,os; print(os.path.join(sys.prefix,"lib","libstdc++.so.6"))')"
    CUDA_VISIBLE_DEVICES=$GPU_ID LD_PRELOAD="$CONDA_LIB" python -m vllm.entrypoints.openai.api_server \
        --model "$MERGE_DIR" \
        --port "$VLLM_PORT" \
        --tensor-parallel-size "$VLLM_TP" \
        --trust-remote-code \
        --dtype bfloat16 \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.90 \
        > /tmp/vllm_eval_4b.log 2>&1 &
    local vllm_pid=$!

    if ! wait_vllm_ready "$VLLM_PORT" 300 "$vllm_pid"; then
        log "    [ERROR] vLLM start failed"
        kill "$vllm_pid" 2>/dev/null || true
        cleanup_vllm
        rm -rf "$MERGE_DIR"
        echo "step=${step}  bias_index=VLLM_ERROR  status=error" >> "$summary_file"
        return 1
    fi
    log "    vLLM ready (PID: ${vllm_pid})"

    # 3. Bias measurement
    log "    [3/4] Bias measurement (3 sets × 10 trials)..."
    mkdir -p "$step_dir"

    pushd "$EVAL_DIR" > /dev/null
    python bias_attribute.py \
        --model-id "$MERGE_DIR" \
        --vllm-url "http://localhost:${VLLM_PORT}/v1" \
        --temperature 0.6 \
        --seed 42 \
        --num-sets 3 \
        --num-trials 10 \
        --max-workers 500 \
        --output-dir "$step_dir" \
        --tag "step_${step}" 2>&1 | tail -5
    popd > /dev/null

    # Flatten subdirectory if created
    local inner_dir
    inner_dir=$(find "$step_dir" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -1)
    if [ -n "$inner_dir" ] && [ -d "$inner_dir" ]; then
        mv "$inner_dir"/* "$step_dir"/ 2>/dev/null || true
        rmdir "$inner_dir" 2>/dev/null || true
    fi

    # Extract bias_index
    local result_json bias_index="ERROR"
    result_json=$(ls "$step_dir"/*_att_result.json 2>/dev/null | head -1)
    if [ -n "$result_json" ] && [ -f "$result_json" ]; then
        bias_index=$(python3 -c "
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    print(d['bias_index'])
except: print('PARSE_ERROR')
" "$result_json" 2>/dev/null)
        log "    ★ Step ${step} → Bias Index: ${bias_index}"
    else
        log "    [ERROR] No result file"
    fi
    echo "step=${step}  bias_index=${bias_index}  status=ok" >> "$summary_file"

    # 4. Cleanup
    log "    [4/4] Cleanup..."
    kill "$vllm_pid" 2>/dev/null || true; wait "$vllm_pid" 2>/dev/null || true
    cleanup_vllm
    rm -rf "$MERGE_DIR"

    log "    step ${step} done"
}

# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════
if [ ! -d "$MODEL_DIR" ]; then
    log "[ERROR] Model dir not found: ${MODEL_DIR}"
    exit 1
fi

CHECKPOINTS=()
for ckpt_dir in "$MODEL_DIR"/checkpoint-*; do
    [ -d "$ckpt_dir" ] || continue
    [ -f "${ckpt_dir}/adapter_model.safetensors" ] || continue
    step="${ckpt_dir##*checkpoint-}"
    CHECKPOINTS+=("$step")
done
IFS=$'\n' CHECKPOINTS=($(sort -n <<<"${CHECKPOINTS[*]}")); unset IFS

if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
    log "[ERROR] No checkpoints in: ${MODEL_DIR}"
    exit 1
fi

log "══════════════════════════════════════════════════════════════"
log "NPO 4B Checkpoint Evaluation"
log "  Adapter: ${MODEL_DIR}"
log "  Checkpoints: ${CHECKPOINTS[*]}"
log "  Results: ${RESULT_DIR}"
log "  GPU: ${GPU_ID}, TP: ${VLLM_TP}"
log "══════════════════════════════════════════════════════════════"

mkdir -p "$RESULT_DIR"
> "${RESULT_DIR}/summary.log"

cleanup_vllm

total=${#CHECKPOINTS[@]}
current=0
for step in "${CHECKPOINTS[@]}"; do
    current=$((current + 1))
    ckpt="${MODEL_DIR}/checkpoint-${step}"
    log ""
    log "[${current}/${total}] checkpoint-${step}"
    eval_checkpoint "$ckpt" "$step"
done

log ""
log "══════════════════════════════════════════════════════════════"
log "Results Summary"
log "══════════════════════════════════════════════════════════════"
cat "${RESULT_DIR}/summary.log" | tee -a "$LOG_FILE"
log ""
log "Done: $(date)"
