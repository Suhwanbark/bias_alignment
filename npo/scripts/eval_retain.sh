#!/bin/bash
# ══════════════════════════════════════════════════════════════
# NPO retain 체크포인트 순차 평가 (4B, GPU 0 only)
#
# LoRA merge (PEFT) → vLLM serve (TP=1) → bias_attribute.py → cleanup
# ══════════════════════════════════════════════════════════════
set -eo pipefail

export CUDA_VISIBLE_DEVICES=0

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

ADAPTER_DIR="${1:?Usage: bash npo/scripts/eval_retain.sh <adapter_dir>}"
RESULT_DIR="${2:-npo/results/npo_qwen4b_retain}"
BASE_MODEL="Qwen/Qwen3-4B-Instruct-2507"
MERGE_DIR="/tmp/_npo_merged_4b"
VLLM_PORT=8000
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

# ── Eval function ────────────────────────────────────────────
eval_checkpoint() {
    local ckpt_path="$1"
    local step="$2"
    local step_dir="${ROOT_DIR}/${RESULT_DIR}/step_${step}"

    # Skip if already done (search recursively)
    if find "$step_dir" -name "*_att_result.json" 2>/dev/null | head -1 | grep -q .; then
        log "step ${step}: already done, skipping"
        return 0
    fi

    log "══ Step ${step} ══"

    # 1. Merge
    log "[1/4] LoRA merge..."
    rm -rf "$MERGE_DIR"
    python /tmp/_merge_lora.py "$ckpt_path" "$MERGE_DIR" "$BASE_MODEL"

    # 2. vLLM serve
    log "[2/4] vLLM serve (TP=1, GPU 0)..."
    LD_PRELOAD=/opt/conda/lib/libstdc++.so.6 CUDA_VISIBLE_DEVICES=0 vllm serve "$MERGE_DIR" \
        --port "$VLLM_PORT" \
        --tensor-parallel-size 1 \
        --trust-remote-code \
        --dtype bfloat16 \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.90 \
        > /tmp/vllm_eval.log 2>&1 &
    local vllm_pid=$!

    # Wait for vLLM ready
    local elapsed=0
    while [ $elapsed -lt 300 ]; do
        if curl -s "http://localhost:${VLLM_PORT}/v1/models" > /dev/null 2>&1; then
            break
        fi
        if ! kill -0 "$vllm_pid" 2>/dev/null; then
            log "[ERROR] vLLM crashed"; cat /tmp/vllm_eval.log | tail -20; return 1
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done
    if [ $elapsed -ge 300 ]; then
        log "[ERROR] vLLM timeout"; kill "$vllm_pid" 2>/dev/null; return 1
    fi
    log "vLLM ready (PID: ${vllm_pid})"

    # 3. Bias eval
    log "[3/4] Bias evaluation (${MAX_WORKERS} workers)..."
    mkdir -p "$step_dir"
    cd "$ROOT_DIR/eval"
    python bias_attribute.py \
        --model-id "$MERGE_DIR" \
        --vllm-url "http://localhost:${VLLM_PORT}/v1" \
        --temperature 0.6 \
        --seed 42 \
        --num-sets 1 \
        --num-trials 10 \
        --max-workers "$MAX_WORKERS" \
        --output-dir "$step_dir" \
        --tag "step_${step}" 2>&1 | tee /tmp/bias_eval_step_${step}.log | tail -10
    cd "$ROOT_DIR"

    # Flatten: move files from any nested subdirectory up to step_dir
    local inner_dir
    inner_dir=$(find "$step_dir" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -1) || true
    if [ -n "$inner_dir" ] && [ -d "$inner_dir" ]; then
        log "Flattening: $inner_dir → $step_dir"
        find "$inner_dir" -maxdepth 1 -type f -exec mv {} "$step_dir/" \; 2>/dev/null || true
        rmdir "$inner_dir" 2>/dev/null || true
    fi

    # Extract bias_index (search recursively just in case)
    local result_json
    result_json=$(find "$step_dir" -name "*_att_result.json" 2>/dev/null | head -1)
    if [ -n "$result_json" ] && [ -f "$result_json" ]; then
        local bi
        bi=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['bias_index'])" "$result_json" 2>/dev/null)
        log "★ Step ${step} → Bias Index: ${bi}"
        echo "step=${step}  bias_index=${bi}" >> "${RESULT_DIR}/summary.log"
    else
        log "[WARN] No result json found"
        echo "step=${step}  bias_index=ERROR" >> "${RESULT_DIR}/summary.log"
    fi

    # 4. Cleanup
    log "[4/4] Cleanup..."
    kill "$vllm_pid" 2>/dev/null || true
    wait "$vllm_pid" 2>/dev/null || true
    sleep 5
    rm -rf "$MERGE_DIR"

    log "Step ${step} done"
    echo ""
}

# ── Main ─────────────────────────────────────────────────────
# Kill any existing vLLM
pkill -f "vllm.entrypoints" 2>/dev/null || true
sleep 3

# Find checkpoints
CHECKPOINTS=()
for ckpt_dir in "$ADAPTER_DIR"/checkpoint-*; do
    [ -d "$ckpt_dir" ] || continue
    step="${ckpt_dir##*checkpoint-}"
    CHECKPOINTS+=("$step")
done
IFS=$'\n' CHECKPOINTS=($(sort -n <<<"${CHECKPOINTS[*]}")); unset IFS

log "══════════════════════════════════════════════════════════"
log "NPO Retain Evaluation (4B, GPU 0)"
log "  Adapter: ${ADAPTER_DIR}"
log "  Checkpoints: ${CHECKPOINTS[*]}"
log "  Results: ${RESULT_DIR}"
log "  Workers: ${MAX_WORKERS}"
log "══════════════════════════════════════════════════════════"

> "${RESULT_DIR}/summary.log"

total=${#CHECKPOINTS[@]}
current=0
for step in "${CHECKPOINTS[@]}"; do
    current=$((current + 1))
    log "[${current}/${total}] checkpoint-${step}"
    eval_checkpoint "${ADAPTER_DIR}/checkpoint-${step}" "$step"
done

log ""
log "══ Summary ══"
cat "${RESULT_DIR}/summary.log"
log "Done: $(date)"
