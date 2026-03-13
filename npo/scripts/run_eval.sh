#!/bin/bash
# ══════════════════════════════════════════════════════════════
# NPO 전체 체크포인트 평가 스크립트
#
# 모든 체크포인트 자동 감지 후 순차 평가:
#   LoRA merge → vLLM serve → bias_attribute.py → cleanup
#
# 사용법:
#   bash npo/run_eval.sh                                    # 기본값 (forget-only)
#   bash npo/run_eval.sh qwen-npo-retain-a05r05 npo_retain_a05r05  # alpha sweep
#
# 결과:
#   npo/results/<result_name>/step_{50,100,...}/
#   npo/results/<result_name>/summary.log
# ══════════════════════════════════════════════════════════════
set -euo pipefail

# ─── 경로 설정 ──────────────────────────────────────────────
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
EVAL_DIR="${ROOT_DIR}/eval"
MODELS_DIR="/home/jovyan/models"

MODEL_NAME="${1:-qwen-npo-forget}"
RESULT_NAME="${2:-npo_forget}"
MODEL_DIR="${MODELS_DIR}/${MODEL_NAME}"
MERGE_TEMPLATE="${ROOT_DIR}/sft/configs/merge.yaml"
MERGE_DIR="${MODELS_DIR}/_tmp_merged"
RESULT_DIR="${ROOT_DIR}/npo/results/${RESULT_NAME}"
VLLM_PORT=8000
VLLM_TP=2

LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${RESULT_NAME}_eval_$(date '+%Y%m%d_%H%M%S').log"

# ─── Utilities ───────────────────────────────────────────────
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg"
    echo "$msg" >> "$LOG_FILE"
}

kill_gpu_processes() {
    log "GPU 프로세스 정리 중..."
    pkill -f "vllm.entrypoints" 2>/dev/null || true
    sleep 5
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null)
    if [ -n "$used" ] && [ "$used" -gt 5000 ] 2>/dev/null; then
        log "  GPU 메모리 ${used}MiB 사용 중, 강제 정리..."
        nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | \
            xargs -r kill -9 2>/dev/null || true
        sleep 10
    fi
    log "  GPU 정리 완료"
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
    log "  [WARN] GPU ${gpu_id} 메모리 해제 대기 타임아웃"
    return 1
}

wait_vllm_ready() {
    local port="$1"
    local max_wait="${2:-300}"
    local pid="$3"
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

# ─── 단일 체크포인트 평가 ────────────────────────────────────
eval_checkpoint() {
    local ckpt_path="$1"
    local step="$2"
    local summary_file="${RESULT_DIR}/summary.log"
    local step_dir="${RESULT_DIR}/step_${step}"

    # 이미 완료된 step은 skip
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
                log "    step ${step}: 이미 완료 (bias_index=${bi}), 스킵"
                echo "step=${step}  bias_index=${bi}  checkpoint=checkpoint-${step}  status=cached" >> "$summary_file"
                return 0
            fi
        fi
    fi

    log "    ── step ${step} 평가 시작 ──"

    # 1. LoRA merge
    log "    [1/4] LoRA merge..."
    rm -rf "$MERGE_DIR"
    local merge_config
    merge_config=$(mktemp /tmp/lf_merge_XXXXXX.yaml)
    sed -e "s|ADAPTER_PATH|${ckpt_path}|g" \
        -e "s|EXPORT_DIR|${MERGE_DIR}|g" \
        "$MERGE_TEMPLATE" > "$merge_config"

    DISABLE_VERSION_CHECK=1 llamafactory-cli export "$merge_config" 2>&1 | tail -5
    rm -f "$merge_config"

    if [ ! -d "$MERGE_DIR" ] || [ ! -f "${MERGE_DIR}/config.json" ]; then
        log "    [ERROR] Merge 실패"
        echo "step=${step}  bias_index=MERGE_ERROR  checkpoint=checkpoint-${step}  status=error" >> "$summary_file"
        rm -rf "$MERGE_DIR"
        return 1
    fi
    log "    Merge 완료"

    # 2. vLLM serve
    log "    [2/4] vLLM 서빙 (TP=${VLLM_TP}, port ${VLLM_PORT})..."
    python -m vllm.entrypoints.openai.api_server \
        --model "$MERGE_DIR" \
        --port "$VLLM_PORT" \
        --tensor-parallel-size "$VLLM_TP" \
        --trust-remote-code \
        --dtype bfloat16 \
        --gpu-memory-utilization 0.90 \
        > /tmp/vllm_eval.log 2>&1 &
    local vllm_pid=$!

    if ! wait_vllm_ready "$VLLM_PORT" 300 "$vllm_pid"; then
        log "    [ERROR] vLLM 시작 실패 (5분 타임아웃)"
        kill "$vllm_pid" 2>/dev/null || true; wait "$vllm_pid" 2>/dev/null || true
        pkill -f "vllm.entrypoints.*${VLLM_PORT}" 2>/dev/null || true
        sleep 5
        rm -rf "$MERGE_DIR"
        echo "step=${step}  bias_index=VLLM_ERROR  checkpoint=checkpoint-${step}  status=error" >> "$summary_file"
        return 1
    fi
    log "    vLLM 준비 완료 (PID: ${vllm_pid})"

    # 3. Bias measurement
    log "    [3/4] Bias index 측정 (3 sets × 10 trials)..."
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

    # 하위 폴더 정리 (bias_attribute.py가 subdirectory를 만드는 경우)
    local inner_dir
    inner_dir=$(find "$step_dir" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -1)
    if [ -n "$inner_dir" ] && [ -d "$inner_dir" ]; then
        mv "$inner_dir"/* "$step_dir"/ 2>/dev/null || true
        rmdir "$inner_dir" 2>/dev/null || true
    fi

    # bias_index 추출
    local result_json
    result_json=$(ls "$step_dir"/*_att_result.json 2>/dev/null | head -1)
    local bias_index="ERROR"
    if [ -n "$result_json" ] && [ -f "$result_json" ]; then
        bias_index=$(python3 -c "
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    print(d['bias_index'])
except: print('PARSE_ERROR')
" "$result_json" 2>/dev/null)
        bias_index="${bias_index:-PARSE_ERROR}"
        log "    ★ Step ${step} → Bias Index: ${bias_index}"
    else
        log "    [ERROR] 결과 파일 없음"
    fi

    echo "step=${step}  bias_index=${bias_index}  checkpoint=checkpoint-${step}  status=ok" >> "$summary_file"

    # 4. Cleanup
    log "    [4/4] 정리..."
    kill "$vllm_pid" 2>/dev/null || true
    wait "$vllm_pid" 2>/dev/null || true
    pkill -f "vllm.entrypoints.*${VLLM_PORT}" 2>/dev/null || true
    sleep 5
    rm -rf "$MERGE_DIR"
    wait_gpu_free 0
    wait_gpu_free 1

    log "    step ${step} 평가 완료"
}


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════
# 체크포인트 존재 확인
if [ ! -d "$MODEL_DIR" ]; then
    log "[ERROR] 모델 디렉토리 없음: ${MODEL_DIR}"
    exit 1
fi

# 체크포인트 자동 감지 (숫자 순 정렬)
CHECKPOINTS=()
for ckpt_dir in "$MODEL_DIR"/checkpoint-*; do
    [ -d "$ckpt_dir" ] || continue
    [ -f "${ckpt_dir}/adapter_model.safetensors" ] || continue
    step="${ckpt_dir##*checkpoint-}"
    CHECKPOINTS+=("$step")
done
IFS=$'\n' CHECKPOINTS=($(sort -n <<<"${CHECKPOINTS[*]}")); unset IFS

if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
    log "[ERROR] 체크포인트 없음: ${MODEL_DIR}"
    exit 1
fi

log "══════════════════════════════════════════════════════════════"
log "NPO 전체 체크포인트 평가"
log "  모델: ${MODEL_DIR}"
log "  체크포인트: ${CHECKPOINTS[*]}"
log "  결과: ${RESULT_DIR}"
log "  로그: ${LOG_FILE}"
log "══════════════════════════════════════════════════════════════"

mkdir -p "$RESULT_DIR"
> "${RESULT_DIR}/summary.log"

kill_gpu_processes

total=${#CHECKPOINTS[@]}
current=0
for step in "${CHECKPOINTS[@]}"; do
    current=$((current + 1))
    ckpt="${MODEL_DIR}/checkpoint-${step}"
    log ""
    log "[${current}/${total}] checkpoint-${step}"
    eval_checkpoint "$ckpt" "$step"
done

# 결과 요약
log ""
log "══════════════════════════════════════════════════════════════"
log "NPO 평가 결과 요약"
log "══════════════════════════════════════════════════════════════"
cat "${RESULT_DIR}/summary.log" | tee -a "$LOG_FILE"

log ""
log "완료: $(date)"
log "로그: ${LOG_FILE}"
