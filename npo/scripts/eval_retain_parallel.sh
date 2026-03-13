#!/bin/bash
# ══════════════════════════════════════════════════════════════
# NPO retain 체크포인트 병렬 평가 (4B)
# GPU 0: steps 50,100,150  |  GPU 1: steps 200,250,300
# n_evidence=1, num_sets=1, num_trials=10
# ══════════════════════════════════════════════════════════════
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

ADAPTER_DIR="${1:?Usage: bash npo/scripts/eval_retain_parallel.sh <adapter_dir> [result_dir]}"
RESULT_DIR="${2:-npo/results/npo_qwen4b_retain_sellonly}"
BASE_MODEL="Qwen/Qwen3-4B-Instruct-2507"
MAX_WORKERS=500

mkdir -p "$RESULT_DIR"

log() { echo "[$(date '+%H:%M:%S')] $1"; }

# ── Merge script ─────────────────────────────────────────────
cat > /tmp/_merge_lora.py << 'PYEOF'
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

adapter_path, output_path, base_model = sys.argv[1], sys.argv[2], sys.argv[3]
print(f"Loading base: {base_model}")
model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
print(f"Loading adapter: {adapter_path}")
model = PeftModel.from_pretrained(model, adapter_path)
print("Merging...")
model = model.merge_and_unload()
print(f"Saving to: {output_path}")
model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)
print("Done")
PYEOF

# ── Eval function (single GPU) ──────────────────────────────
eval_checkpoint() {
    local ckpt_path="$1"
    local step="$2"
    local gpu_id="$3"
    local port="$4"
    local merge_dir="/tmp/_npo_merged_4b_gpu${gpu_id}"
    local step_dir="${ROOT_DIR}/${RESULT_DIR}/step_${step}"

    # Skip if already done
    if find "$step_dir" -name "*_att_result.json" 2>/dev/null | head -1 | grep -q .; then
        log "[GPU${gpu_id}] step ${step}: already done, skipping"
        return 0
    fi

    log "[GPU${gpu_id}] ══ Step ${step} ══"

    # 1. Merge
    log "[GPU${gpu_id}] [1/4] LoRA merge..."
    rm -rf "$merge_dir"
    CUDA_VISIBLE_DEVICES=$gpu_id python /tmp/_merge_lora.py "$ckpt_path" "$merge_dir" "$BASE_MODEL"

    # 2. vLLM serve
    log "[GPU${gpu_id}] [2/4] vLLM serve (port ${port})..."
    LD_PRELOAD=/opt/conda/lib/libstdc++.so.6 CUDA_VISIBLE_DEVICES=$gpu_id vllm serve "$merge_dir" \
        --port "$port" \
        --tensor-parallel-size 1 \
        --trust-remote-code \
        --dtype bfloat16 \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.90 \
        > /tmp/vllm_eval_gpu${gpu_id}.log 2>&1 &
    local vllm_pid=$!

    # Wait for vLLM ready
    local elapsed=0
    while [ $elapsed -lt 300 ]; do
        if curl -s "http://localhost:${port}/v1/models" > /dev/null 2>&1; then
            break
        fi
        if ! kill -0 "$vllm_pid" 2>/dev/null; then
            log "[GPU${gpu_id}] [ERROR] vLLM crashed"; tail -20 /tmp/vllm_eval_gpu${gpu_id}.log; return 1
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done
    if [ $elapsed -ge 300 ]; then
        log "[GPU${gpu_id}] [ERROR] vLLM timeout"; kill "$vllm_pid" 2>/dev/null; return 1
    fi
    log "[GPU${gpu_id}] vLLM ready (PID: ${vllm_pid})"

    # 3. Bias eval
    log "[GPU${gpu_id}] [3/4] Bias evaluation..."
    mkdir -p "$step_dir"
    cd "$ROOT_DIR/eval"
    python bias_attribute.py \
        --model-id "$merge_dir" \
        --vllm-url "http://localhost:${port}/v1" \
        --temperature 0.6 \
        --seed 42 \
        --num-sets 1 \
        --num-trials 10 \
        --n-evidence 1 \
        --max-workers "$MAX_WORKERS" \
        --output-dir "$step_dir" \
        --tag "step_${step}" 2>&1 | tee /tmp/bias_eval_gpu${gpu_id}_step_${step}.log | tail -10
    cd "$ROOT_DIR"

    # Flatten
    local inner_dir
    inner_dir=$(find "$step_dir" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -1) || true
    if [ -n "$inner_dir" ] && [ -d "$inner_dir" ]; then
        log "[GPU${gpu_id}] Flattening..."
        find "$inner_dir" -maxdepth 1 -type f -exec mv {} "$step_dir/" \; 2>/dev/null || true
        rmdir "$inner_dir" 2>/dev/null || true
    fi

    # Extract bias_index
    local result_json
    result_json=$(find "$step_dir" -name "*_att_result.json" 2>/dev/null | head -1)
    if [ -n "$result_json" ] && [ -f "$result_json" ]; then
        local bi
        bi=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['bias_index'])" "$result_json" 2>/dev/null)
        log "[GPU${gpu_id}] ★ Step ${step} → Bias Index: ${bi}"
        echo "step=${step}  bias_index=${bi}" >> "${RESULT_DIR}/summary.log"
    else
        log "[GPU${gpu_id}] [WARN] No result json found"
        echo "step=${step}  bias_index=ERROR" >> "${RESULT_DIR}/summary.log"
    fi

    # 4. Cleanup
    log "[GPU${gpu_id}] [4/4] Cleanup..."
    kill "$vllm_pid" 2>/dev/null || true
    wait "$vllm_pid" 2>/dev/null || true
    sleep 3
    rm -rf "$merge_dir"

    log "[GPU${gpu_id}] Step ${step} done"
    echo ""
}

# ── Main ─────────────────────────────────────────────────────
pkill -f "vllm.entrypoints" 2>/dev/null || true
sleep 3

> "${RESULT_DIR}/summary.log"

log "══════════════════════════════════════════════════════════"
log "NPO Retain Parallel Evaluation (4B)"
log "  Adapter: ${ADAPTER_DIR}"
log "  GPU 0 (port 8000): steps 50, 100, 150"
log "  GPU 1 (port 8001): steps 200, 250, 300"
log "  n_evidence=1, num_sets=1, num_trials=10"
log "══════════════════════════════════════════════════════════"

# GPU 0: steps 50, 100, 150
(
    for step in 50 100 150; do
        eval_checkpoint "${ADAPTER_DIR}/checkpoint-${step}" "$step" 0 8000
    done
) &
pid_gpu0=$!

# GPU 1: steps 200, 250, 300
(
    for step in 200 250 300; do
        eval_checkpoint "${ADAPTER_DIR}/checkpoint-${step}" "$step" 1 8001
    done
) &
pid_gpu1=$!

# Wait for both
wait $pid_gpu0
wait $pid_gpu1

log ""
log "══ Summary ══"
sort -t= -k1 -n "${RESULT_DIR}/summary.log"
log "Done: $(date)"
