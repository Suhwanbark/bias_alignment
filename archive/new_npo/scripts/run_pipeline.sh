#!/bin/bash
# ══════════════════════════════════════════════════════════════
# BiasUnlearn 전체 파이프라인
#
# 1. Counterfactual retain 생성 (vLLM offline)
# 2. Retain 데이터셋 재구성 (counterfactual 병합)
# 3. 3가지 loss weight로 학습 sweep
# 4. 각 실험 전체 체크포인트 평가 (vLLM offline)
#
# Usage:
#   bash new_npo/scripts/run_pipeline.sh
# ══════════════════════════════════════════════════════════════
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

NEW_NPO="new_npo"
DATA_DIR="${NEW_NPO}/data/qwen3-30b-a3b-instruct-2507"
MODELS_DIR="/home/jovyan/models"
CONFIG="${NEW_NPO}/configs/qwen3_30b.yaml"
LOG_DIR="${ROOT_DIR}/debias/logs"
mkdir -p "$LOG_DIR"
PIPELINE_LOG="${LOG_DIR}/pipeline_$(date '+%Y%m%d_%H%M%S').log"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg"
    echo "$msg" >> "$PIPELINE_LOG"
}

log "══════════════════════════════════════════════════════════════"
log "BiasUnlearn 전체 파이프라인 시작"
log "  로그: ${PIPELINE_LOG}"
log "══════════════════════════════════════════════════════════════"

# ─── Step 1: Counterfactual Retain 생성 ──────────────────────
log ""
log "═══ Step 1/4: Counterfactual Retain 생성 (vLLM offline) ═══"

CF_PATH="${DATA_DIR}/retain_counterfactual.jsonl"

if [ -f "$CF_PATH" ] && [ "$(wc -l < "$CF_PATH")" -gt 10 ]; then
    log "  이미 존재: ${CF_PATH} ($(wc -l < "$CF_PATH") samples), 스킵"
else
    python -m new_npo.prepare_counterfactual \
        --forget-path "${DATA_DIR}/forget.jsonl" \
        --sp500-path data/sp500_final.csv \
        --model-id Qwen/Qwen3-30B-A3B-Instruct-2507 \
        --output-path "$CF_PATH" \
        --temperature 1.0 \
        --num-samples 10 \
        --max-tokens 150 \
        --seed 42 2>&1 | tee -a "$PIPELINE_LOG"
fi

log "  Counterfactual 완료: $(wc -l < "$CF_PATH") samples"

# ─── Step 2: Retain 데이터셋 재구성 ──────────────────────────
log ""
log "═══ Step 2/4: Retain 데이터셋 재구성 (counterfactual 병합) ═══"

# Backup original retain
cp "${DATA_DIR}/retain.jsonl" "${DATA_DIR}/retain_original.jsonl" 2>/dev/null || true

python -m new_npo.prepare_dataset \
    --csv "${NEW_NPO}/bias_profiling/qwen3-30b-a3b-instruct-2507_att_combined.csv" \
    --threshold 80 \
    --neutral-threshold 20 \
    --output-dir "$DATA_DIR" \
    --counterfactual "$CF_PATH" \
    --seed 42 2>&1 | tee -a "$PIPELINE_LOG"

log "  Retain 재구성 완료: $(wc -l < "${DATA_DIR}/retain.jsonl") samples"

# ─── Step 3: 3가지 loss weight로 학습 ────────────────────────
log ""
log "═══ Step 3/4: Training Sweep (3 experiments) ═══"

# alpha_forget:alpha_retain:model_suffix:result_name
EXPERIMENTS=(
    "0.75:0.25:qwen-biasunlearn-a75r25:biasunlearn_a75r25"
    "0.5:0.5:qwen-biasunlearn-a50r50:biasunlearn_a50r50"
    "0.25:0.75:qwen-biasunlearn-a25r75:biasunlearn_a25r75"
)

for i in "${!EXPERIMENTS[@]}"; do
    IFS=':' read -r af ar model_suffix result_name <<< "${EXPERIMENTS[$i]}"
    output_dir="${MODELS_DIR}/${model_suffix}"
    exp_num=$((i + 1))

    log ""
    log "  ── Experiment ${exp_num}/3: alpha_f=${af}, alpha_r=${ar} ──"
    log "  Output: ${output_dir}"

    if [ -d "$output_dir" ] && ls "$output_dir"/checkpoint-*/adapter_model.safetensors >/dev/null 2>&1; then
        n_ckpts=$(ls -d "$output_dir"/checkpoint-* 2>/dev/null | wc -l)
        log "  이미 학습 완료 (${n_ckpts} checkpoints), 스킵"
        continue
    fi

    # Create temp config with modified loss weights and output dir
    tmp_config=$(mktemp /tmp/train_config_XXXXXX.yaml)
    sed -e "s/alpha_forget: .*/alpha_forget: ${af}/" \
        -e "s/alpha_retain: .*/alpha_retain: ${ar}/" \
        -e "s|output_dir: .*|output_dir: ${output_dir}|" \
        "$CONFIG" > "$tmp_config"

    python -m new_npo.src.train \
        --config "$tmp_config" 2>&1 | tee -a "$PIPELINE_LOG"

    rm -f "$tmp_config"
    log "  Experiment ${exp_num} 학습 완료"
done

# ─── Step 4: 전체 체크포인트 평가 (vLLM offline) ─────────────
log ""
log "═══ Step 4/4: Evaluation Sweep (vLLM offline) ═══"

for i in "${!EXPERIMENTS[@]}"; do
    IFS=':' read -r af ar model_suffix result_name <<< "${EXPERIMENTS[$i]}"
    output_dir="${MODELS_DIR}/${model_suffix}"
    exp_num=$((i + 1))

    log ""
    log "  ── Eval ${exp_num}/3: ${result_name} ──"

    if [ ! -d "$output_dir" ]; then
        log "  [SKIP] 모델 디렉토리 없음: ${output_dir}"
        continue
    fi

    # Check if eval already done
    eval_summary="local/${result_name}/result/summary.log"
    if [ -f "$eval_summary" ] && grep -q "status=ok" "$eval_summary"; then
        log "  이미 평가 완료, 스킵"
        cat "$eval_summary" >> "$PIPELINE_LOG"
        continue
    fi

    python new_npo/scripts/run_eval_local.py \
        --model-dir "$output_dir" \
        --result-name "$result_name" \
        2>&1 | tee -a "$PIPELINE_LOG"

    log "  Eval ${exp_num} 완료"
done

# ─── 최종 결과 요약 ──────────────────────────────────────────
log ""
log "══════════════════════════════════════════════════════════════"
log "파이프라인 완료 — 결과 요약"
log "══════════════════════════════════════════════════════════════"

for i in "${!EXPERIMENTS[@]}"; do
    IFS=':' read -r af ar model_suffix result_name <<< "${EXPERIMENTS[$i]}"
    log ""
    log "── alpha_f=${af}, alpha_r=${ar} (${result_name}) ──"
    if [ -f "local/${result_name}/result/summary.log" ]; then
        cat "local/${result_name}/result/summary.log" | tee -a "$PIPELINE_LOG"
    else
        log "  (결과 없음)"
    fi
done

log ""
log "══════════════════════════════════════════════════════════════"
log "완료: $(date)"
log "로그: ${PIPELINE_LOG}"
log "══════════════════════════════════════════════════════════════"
