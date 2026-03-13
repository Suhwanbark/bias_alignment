#!/bin/bash
# ══════════════════════════════════════════════════════════════
# SimNPO 전체 체크포인트 평가 (vLLM offline)
#
# run_eval_local.py를 호출하여 LoRA를 직접 로드하고 평가.
# vLLM 서버 없이 offline batch inference로 동작.
#
# 사용법:
#   bash simple_npo/scripts/run_eval.sh <model_dir> <result_name>
#   bash simple_npo/scripts/run_eval.sh /home/jovyan/models/qwen-simnpo simnpo
#
# 결과:
#   simple_npo/results/<result_name>/step_{50,100,...}/
#   simple_npo/results/<result_name>/summary.log
# ══════════════════════════════════════════════════════════════
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
MODELS_DIR="/home/jovyan/models"

MODEL_DIR="${1:-${MODELS_DIR}/qwen-simnpo}"
RESULT_NAME="${2:-simnpo}"

LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${RESULT_NAME}_eval_$(date '+%Y%m%d_%H%M%S').log"

echo "══════════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
echo "SimNPO 체크포인트 평가 (vLLM offline)" | tee -a "$LOG_FILE"
echo "  모델: ${MODEL_DIR}" | tee -a "$LOG_FILE"
echo "  결과: simple_npo/results/${RESULT_NAME}/" | tee -a "$LOG_FILE"
echo "  로그: ${LOG_FILE}" | tee -a "$LOG_FILE"
echo "══════════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"

cd "$ROOT_DIR"

python simple_npo/scripts/run_eval_local.py \
    --model-dir "$MODEL_DIR" \
    --result-name "$RESULT_NAME" \
    2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "완료: $(date)" | tee -a "$LOG_FILE"
echo "로그: ${LOG_FILE}" | tee -a "$LOG_FILE"
