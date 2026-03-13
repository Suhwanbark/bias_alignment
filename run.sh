#!/bin/bash
# ══════════════════════════════════════════════════════════════
# 통합 실행 스크립트: 훈련 + (옵션) 평가
#
# 사용법:
#   bash run.sh sft_v7                  # 훈련만
#   bash run.sh sft_v7_4b               # 4B 모델 훈련
#   bash run.sh sft_v7 --eval           # 훈련 + 마지막 체크포인트 평가
#   bash run.sh sft_v7 --eval-only      # 훈련 스킵, 평가만
#
# 평가 파이프라인 (--eval 또는 --eval-only):
#   마지막 체크포인트:
#     1. LoRA merge (llamafactory-cli export)
#     2. vLLM serve (모델에 따라 TP=1 또는 TP=2)
#     3. bias_attribute.py (3 sets × 10 trials)
#     4. bias_index 추출 + 정리
# ══════════════════════════════════════════════════════════════
set -euo pipefail

# ─── 인자 파싱 ───────────────────────────────────────────────
if [ $# -lt 1 ]; then
    echo "사용법: bash run.sh <config_name> [--eval|--eval-only]"
    echo ""
    echo "  config_name:  sft_v1~v8, dpo_v1~v13, sft_v7_4b, ..."
    echo "  --eval:       훈련 후 마지막 체크포인트 평가"
    echo "  --eval-only:  훈련 스킵, 마지막 체크포인트 평가만"
    echo ""
    echo "예시:"
    echo "  bash run.sh sft_v7"
    echo "  bash run.sh sft_v7_4b --eval"
    echo "  bash run.sh dpo_v1 --eval-only"
    exit 1
fi

CONFIG_NAME="$1"
MODE="${2:-train-only}"

# ─── 경로 설정 ──────────────────────────────────────────────
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
EVAL_DIR="${ROOT_DIR}/eval"
MODELS_DIR="/home/jovyan/models"

# Auto-detect SFT vs DPO from CONFIG_NAME prefix
if [[ "$CONFIG_NAME" == sft_* ]]; then
    TRAIN_DIR="${ROOT_DIR}/sft"
elif [[ "$CONFIG_NAME" == dpo_* ]]; then
    TRAIN_DIR="${ROOT_DIR}/dpo"
else
    echo "[ERROR] CONFIG_NAME must start with 'sft_' or 'dpo_': ${CONFIG_NAME}"
    exit 1
fi

CONFIG_FILE="${TRAIN_DIR}/configs/${CONFIG_NAME}.yaml"
MERGE_DIR="${MODELS_DIR}/_tmp_merged"
VLLM_PORT=8000

LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${CONFIG_NAME}_$(date '+%Y%m%d_%H%M%S').log"

# ─── YAML 파싱 ────────────────────────────────────────────
get_yaml_value() {
    grep "^${1}:" "$2" | head -1 | sed "s/^${1}:[[:space:]]*//" | sed 's/[[:space:]]*$//'
}

if [ ! -f "$CONFIG_FILE" ]; then
    echo "[ERROR] Config 없음: ${CONFIG_FILE}"
    echo ""
    echo "사용 가능한 설정:"
    ls "${TRAIN_DIR}/configs/" | grep -v merge | sed 's/.yaml$//' | sed 's/^/  /'
    exit 1
fi

MODEL_DIR=$(get_yaml_value "output_dir" "$CONFIG_FILE")
BASE_MODEL=$(get_yaml_value "model_name_or_path" "$CONFIG_FILE")
RESULT_DIR="${TRAIN_DIR}/results/${CONFIG_NAME}/result"

# ─── 모델별 설정 (4B vs 30B) ──────────────────────────────
if echo "$BASE_MODEL" | grep -qi "4B"; then
    VLLM_TP=1
    MERGE_TEMPLATE="${TRAIN_DIR}/configs/merge_4b.yaml"
else
    VLLM_TP=2
    MERGE_TEMPLATE="${TRAIN_DIR}/configs/merge.yaml"
fi


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
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
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

extract_step() {
    echo "$1" | sed 's/.*checkpoint-//' | grep -oP '^\d+'
}


# ══════════════════════════════════════════════════════════════
# 훈련
# ══════════════════════════════════════════════════════════════
do_train() {
    log "══════════════════════════════════════════════════════════════"
    log "[${CONFIG_NAME}] 훈련 시작"
    log "  Config:     ${CONFIG_FILE}"
    log "  Base model: ${BASE_MODEL}"
    log "  Output:     ${MODEL_DIR}"
    log "══════════════════════════════════════════════════════════════"

    if [ -d "$MODEL_DIR" ] && ls "${MODEL_DIR}"/checkpoint-* 1>/dev/null 2>&1; then
        local n
        n=$(ls -d "${MODEL_DIR}"/checkpoint-* 2>/dev/null | wc -l)
        log "[SKIP] 이미 학습됨 (${n}개 체크포인트)"
        return 0
    fi

    kill_gpu_processes

    bash "${TRAIN_DIR}/train.sh" "$CONFIG_FILE" 2>&1 | tee -a "$LOG_FILE"
    local status=${PIPESTATUS[0]}

    if [ $status -ne 0 ]; then
        log "[ERROR] 훈련 실패 (exit: ${status})"
        exit $status
    fi

    local n
    n=$(ls -d "${MODEL_DIR}"/checkpoint-* 2>/dev/null | wc -l)
    log "[OK] 훈련 완료 (${n}개 체크포인트)"
}


# ══════════════════════════════════════════════════════════════
# 평가: 단일 체크포인트
# ══════════════════════════════════════════════════════════════
eval_checkpoint() {
    local ckpt_path="$1"
    local summary_file="${RESULT_DIR}/summary.log"

    local ckpt_name
    ckpt_name=$(basename "$ckpt_path")
    local step
    step=$(extract_step "$ckpt_name")
    if [ -z "$step" ]; then
        log "    [ERROR] step 번호 추출 실패: ${ckpt_name}"
        return 1
    fi
    local step_dir="${RESULT_DIR}/step_${step}"

    # 이미 완료?
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
                echo "step=${step}  bias_index=${bi}  checkpoint=${ckpt_name}  status=cached" >> "$summary_file"
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
        echo "step=${step}  bias_index=MERGE_ERROR  checkpoint=${ckpt_name}  status=error" >> "$summary_file"
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
        kill "$vllm_pid" 2>/dev/null; wait "$vllm_pid" 2>/dev/null
        pkill -f "vllm.entrypoints.*${VLLM_PORT}" 2>/dev/null
        sleep 5
        rm -rf "$MERGE_DIR"
        echo "step=${step}  bias_index=VLLM_ERROR  checkpoint=${ckpt_name}  status=error" >> "$summary_file"
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

    # 하위 폴더 정리
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

    echo "step=${step}  bias_index=${bias_index}  checkpoint=${ckpt_name}  status=ok" >> "$summary_file"

    # 4. Cleanup
    log "    [4/4] 정리..."
    kill "$vllm_pid" 2>/dev/null
    wait "$vllm_pid" 2>/dev/null
    pkill -f "vllm.entrypoints.*${VLLM_PORT}" 2>/dev/null
    sleep 5
    rm -rf "$MERGE_DIR"
    wait_gpu_free 0
    if [ "$VLLM_TP" -ge 2 ]; then
        wait_gpu_free 1
    fi

    log "    step ${step} 평가 완료"
}


# ══════════════════════════════════════════════════════════════
# 평가: 마지막 체크포인트
# ══════════════════════════════════════════════════════════════
do_eval() {
    log "══════════════════════════════════════════════════════════════"
    log "[${CONFIG_NAME}] 평가 시작"
    log "  체크포인트: ${MODEL_DIR}"
    log "  결과 저장: ${RESULT_DIR}"
    log "  Base model: ${BASE_MODEL} (TP=${VLLM_TP})"
    log "══════════════════════════════════════════════════════════════"

    if [ ! -d "$MODEL_DIR" ]; then
        log "[ERROR] 체크포인트 디렉토리 없음: ${MODEL_DIR}"
        return 1
    fi

    # 마지막 체크포인트 선택
    local last_ckpt
    last_ckpt=$(ls -d "${MODEL_DIR}"/checkpoint-* 2>/dev/null | sort -t'-' -k2 -n | tail -1)
    if [ -z "$last_ckpt" ]; then
        log "[ERROR] 체크포인트 없음"
        return 1
    fi

    local total
    total=$(ls -d "${MODEL_DIR}"/checkpoint-* 2>/dev/null | wc -l)
    log "${total}개 체크포인트 중 마지막만 평가: $(basename "$last_ckpt")"

    mkdir -p "$RESULT_DIR"
    > "${RESULT_DIR}/summary.log"

    kill_gpu_processes

    if [ ! -f "${last_ckpt}/adapter_model.safetensors" ]; then
        log "[ERROR] adapter 없음: $(basename "$last_ckpt")"
        return 1
    fi

    log "[1/1] $(basename "$last_ckpt")"
    eval_checkpoint "$last_ckpt"

    # 결과 출력
    log ""
    log "══════════════════════════════════════════════════════════════"
    log "[${CONFIG_NAME}] 평가 결과"
    log "══════════════════════════════════════════════════════════════"
    cat "${RESULT_DIR}/summary.log" | tee -a "$LOG_FILE"
}


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════
log "══════════════════════════════════════════════════════════════"
log "실행: ${CONFIG_NAME} (mode: ${MODE})"
log "  Base model: ${BASE_MODEL} (TP=${VLLM_TP})"
log "  로그: ${LOG_FILE}"
log "══════════════════════════════════════════════════════════════"

case "$MODE" in
    train-only)
        do_train
        ;;
    --eval)
        do_train
        do_eval
        ;;
    --eval-only)
        do_eval
        ;;
    *)
        echo "[ERROR] 알 수 없는 옵션: ${MODE}"
        echo "사용법: bash run.sh <config_name> [--eval|--eval-only]"
        exit 1
        ;;
esac

log ""
log "완료: $(date)"
log "로그: ${LOG_FILE}"
