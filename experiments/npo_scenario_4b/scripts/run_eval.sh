#!/bin/bash
# ══════════════════════════════════════════════════════════════
# NPO Scenario 4B: parallel evaluation on 2 GPUs
#
# Evaluates baseline + checkpoint-50/100/150 (2 at a time)
#
# Usage:
#   bash experiments/npo_scenario_4b/scripts/run_eval.sh \
#       models/adapters/npo_scenario_4b npo_scenario_4b
# ══════════════════════════════════════════════════════════════
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
EXP_DIR="${ROOT_DIR}/experiments/npo_scenario_4b"

MODEL_DIR="${ROOT_DIR}/${1:?Usage: $0 <adapter_dir> <result_name>}"
RESULT_NAME="${2:?Usage: $0 <adapter_dir> <result_name>}"
RESULT_DIR="${EXP_DIR}/results/${RESULT_NAME}"
MAX_STEP="${3:-150}"

LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "$LOG_DIR" "$RESULT_DIR"
LOG_FILE="${LOG_DIR}/${RESULT_NAME}_eval_$(date '+%Y%m%d_%H%M%S').log"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg"
    echo "$msg" >> "$LOG_FILE"
}

CONDA_LIB="$(python3 -c 'import sys,os; print(os.path.join(sys.prefix,"lib","libstdc++.so.6"))')"

wait_vllm_ready() {
    local port="$1" max_wait="${2:-300}" elapsed=0
    while [ "$elapsed" -lt "$max_wait" ]; do
        if curl -s "http://localhost:${port}/v1/models" > /dev/null 2>&1; then
            return 0
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done
    return 1
}

cleanup_all_vllm() {
    pkill -f "vllm.entrypoints" 2>/dev/null || true
    sleep 5
}

# Merge LoRA adapter into base model
merge_checkpoint() {
    local ckpt_path="$1" output_dir="$2"
    rm -rf "$output_dir"
    python3 -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = 'Qwen/Qwen3-4B-Instruct-2507'
tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map='cpu', trust_remote_code=True)
model = PeftModel.from_pretrained(model, '${ckpt_path}')
model = model.merge_and_unload()
model.save_pretrained('${output_dir}', max_shard_size='5GB')
tokenizer.save_pretrained('${output_dir}')
print('Merge complete')
" 2>&1 | tail -3
}

# Start vLLM on a specific GPU/port
start_vllm() {
    local model_path="$1" gpu_id="$2" port="$3"
    CUDA_VISIBLE_DEVICES=$gpu_id LD_PRELOAD="$CONDA_LIB" python -m vllm.entrypoints.openai.api_server \
        --model "$model_path" \
        --port "$port" \
        --tensor-parallel-size 1 \
        --trust-remote-code \
        --dtype bfloat16 \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.90 \
        > "/tmp/vllm_eval_gpu${gpu_id}.log" 2>&1 &
    echo $!
}

# Run eval_bias.py
run_eval() {
    local port="$1" output_dir="$2" label="$3"
    mkdir -p "$output_dir"
    python3 "${EXP_DIR}/scripts/eval_bias.py" \
        --vllm-url "http://localhost:${port}/v1" \
        --seed 42 \
        --num-trials 10 \
        --max-workers 200 \
        --output-dir "$output_dir" \
        --target-tickers "${EXP_DIR}/data/target_tickers.json" \
        2>&1 | tail -5

    if [ -f "${output_dir}/result.json" ]; then
        local bi
        bi=$(python3 -c "import json; print(json.load(open('${output_dir}/result.json'))['bias_index'])" 2>/dev/null)
        log "    ★ ${label} → Bias Index: ${bi}"
        echo "${label}  bias_index=${bi}  status=ok" >> "${RESULT_DIR}/summary.log"
    else
        log "    [ERROR] ${label}: No result file"
        echo "${label}  bias_index=ERROR  status=error" >> "${RESULT_DIR}/summary.log"
    fi
}

# Evaluate one pair of checkpoints on GPU 0 + GPU 1
eval_pair() {
    local step_a="$1" model_a="$2" step_b="${3:-}" model_b="${4:-}"

    log "── Evaluating pair: ${step_a} + ${step_b:-none} ──"

    # Merge
    local merge_a="/tmp/_merge_scenario_4b_a"
    local merge_b="/tmp/_merge_scenario_4b_b"

    if [ "$step_a" = "baseline" ]; then
        merge_a="Qwen/Qwen3-4B-Instruct-2507"
    else
        log "  Merging ${step_a}..."
        merge_checkpoint "$model_a" "$merge_a"
    fi

    if [ -n "$step_b" ]; then
        log "  Merging ${step_b}..."
        merge_checkpoint "$model_b" "$merge_b"
    fi

    # Start vLLM servers
    cleanup_all_vllm
    local pid_a pid_b

    log "  Starting vLLM: GPU 0 (port 8000) = ${step_a}"
    pid_a=$(start_vllm "$merge_a" 0 8000)
    if [ -n "$step_b" ]; then
        log "  Starting vLLM: GPU 1 (port 8001) = ${step_b}"
        pid_b=$(start_vllm "$merge_b" 1 8001)
    fi

    # Wait for servers
    if ! wait_vllm_ready 8000 300; then
        log "  [ERROR] vLLM GPU 0 failed"
        cleanup_all_vllm
        return 1
    fi
    log "  vLLM GPU 0 ready"

    if [ -n "$step_b" ]; then
        if ! wait_vllm_ready 8001 300; then
            log "  [ERROR] vLLM GPU 1 failed"
            cleanup_all_vllm
            return 1
        fi
        log "  vLLM GPU 1 ready"
    fi

    # Run evals in parallel
    log "  Running evaluations..."
    run_eval 8000 "${RESULT_DIR}/${step_a}" "${step_a}" &
    local eval_pid_a=$!

    if [ -n "$step_b" ]; then
        run_eval 8001 "${RESULT_DIR}/${step_b}" "${step_b}" &
        local eval_pid_b=$!
        wait "$eval_pid_b"
    fi
    wait "$eval_pid_a"

    # Cleanup
    cleanup_all_vllm
    [ "$step_a" != "baseline" ] && rm -rf "$merge_a"
    [ -n "$step_b" ] && rm -rf "$merge_b"

    log "  Pair done: ${step_a} + ${step_b:-none}"
}


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════
log "══════════════════════════════════════════════════════════════"
log "NPO Scenario 4B — Parallel Evaluation (2 GPUs)"
log "  Adapter: ${MODEL_DIR}"
log "  Max step: ${MAX_STEP}"
log "  Results: ${RESULT_DIR}"
log "══════════════════════════════════════════════════════════════"

> "${RESULT_DIR}/summary.log"
cleanup_all_vllm

# Collect checkpoints up to MAX_STEP
STEPS=()
for ckpt_dir in "$MODEL_DIR"/checkpoint-*; do
    [ -d "$ckpt_dir" ] || continue
    step="${ckpt_dir##*checkpoint-}"
    [ "$step" -le "$MAX_STEP" ] 2>/dev/null && STEPS+=("$step")
done
IFS=$'\n' STEPS=($(sort -n <<<"${STEPS[*]}")); unset IFS

log "Checkpoints: ${STEPS[*]}"
log "Plan: baseline + step_50 (pair 1), step_100 + step_150 (pair 2)"

# Pair 1: baseline + step_50
eval_pair "baseline" "" "step_50" "${MODEL_DIR}/checkpoint-50"

# Pair 2: step_100 + step_150
eval_pair "step_100" "${MODEL_DIR}/checkpoint-100" "step_150" "${MODEL_DIR}/checkpoint-150"

log ""
log "══════════════════════════════════════════════════════════════"
log "Summary"
log "══════════════════════════════════════════════════════════════"
cat "${RESULT_DIR}/summary.log" | tee -a "$LOG_FILE"
log "Done: $(date)"
