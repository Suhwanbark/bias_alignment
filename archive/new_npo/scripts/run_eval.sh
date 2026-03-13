#!/bin/bash
# ══════════════════════════════════════════════════════════════
# BiasUnlearn NPO 전체 체크포인트 평가 (vLLM offline)
#
# run_eval_local.py를 호출하여 LoRA를 직접 로드하고 평가.
# vLLM 서버 없이 offline batch inference로 동작.
#
# 사용법:
#   bash new_npo/scripts/run_eval.sh <model_dir> <result_name>
#   bash new_npo/scripts/run_eval.sh /home/jovyan/models/qwen-npo-biasunlearn npo_biasunlearn
#
# 결과:
#   local/<result_name>/result/step_{50,100,...}/
#   local/<result_name>/result/summary.log
# ══════════════════════════════════════════════════════════════
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
MODELS_DIR="/home/jovyan/models"

MODEL_DIR="${1:-${MODELS_DIR}/qwen-npo-biasunlearn}"
RESULT_NAME="${2:-npo_biasunlearn}"

LOG_DIR="${ROOT_DIR}/debias/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${RESULT_NAME}_eval_$(date '+%Y%m%d_%H%M%S').log"

echo "══════════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
echo "BiasUnlearn NPO 체크포인트 평가 (vLLM offline)" | tee -a "$LOG_FILE"
echo "  모델: ${MODEL_DIR}" | tee -a "$LOG_FILE"
echo "  결과: local/${RESULT_NAME}/result/" | tee -a "$LOG_FILE"
echo "  로그: ${LOG_FILE}" | tee -a "$LOG_FILE"
echo "══════════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"

cd "$ROOT_DIR"

python new_npo/scripts/run_eval_local.py \
    --model-dir "$MODEL_DIR" \
    --result-name "$RESULT_NAME" \
    2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "완료: $(date)" | tee -a "$LOG_FILE"
echo "로그: ${LOG_FILE}" | tee -a "$LOG_FILE"
