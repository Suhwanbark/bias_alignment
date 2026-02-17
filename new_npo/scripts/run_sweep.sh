#!/bin/bash
# ══════════════════════════════════════════════════════════════
# BiasUnlearn NPO loss weight ablation sweep
#
# 각 실험: train → eval all checkpoints (vLLM offline)
#
# 사용법:
#   bash new_npo/scripts/run_sweep.sh
#
# 실험:
#   A: alpha_forget=0.4,  alpha_retain=0.4,  alpha_kl=0.2  (default)
#   B: alpha_forget=0.5,  alpha_retain=0.3,  alpha_kl=0.2
#   C: alpha_forget=0.3,  alpha_retain=0.5,  alpha_kl=0.2
# ══════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR/.."

MODELS_DIR="/home/jovyan/models"
CONFIG="new_npo/configs/qwen3_30b.yaml"

# ─── 실험 정의 (alpha_f:alpha_r:alpha_kl:model_name:result_name) ───
EXPERIMENTS=(
    "0.4:0.4:0.2:qwen-biasunlearn-a04r04:biasunlearn_a04r04"
    "0.5:0.3:0.2:qwen-biasunlearn-a05r03:biasunlearn_a05r03"
    "0.3:0.5:0.2:qwen-biasunlearn-a03r05:biasunlearn_a03r05"
)

echo "══════════════════════════════════════════════════════════════"
echo "BiasUnlearn NPO Loss Weight Sweep: ${#EXPERIMENTS[@]} experiments"
echo "══════════════════════════════════════════════════════════════"

for exp in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r af ar akl model_name result_name <<< "$exp"
    output_dir="${MODELS_DIR}/${model_name}"

    echo ""
    echo "════════════════════════════════════════"
    echo "Training: alpha_f=${af}, alpha_r=${ar}, alpha_kl=${akl}"
    echo "  Model: ${output_dir}"
    echo "  Result: local/${result_name}/result/"
    echo "════════════════════════════════════════"

    # Create temporary config with modified loss weights
    tmp_config=$(mktemp /tmp/sweep_config_XXXXXX.yaml)
    sed -e "s/alpha_forget: .*/alpha_forget: ${af}/" \
        -e "s/alpha_retain: .*/alpha_retain: ${ar}/" \
        -e "s/alpha_kl: .*/alpha_kl: ${akl}/" \
        -e "s|output_dir: .*|output_dir: ${output_dir}|" \
        "$CONFIG" > "$tmp_config"

    # Train
    python -m new_npo.src.train --config "$tmp_config"
    rm -f "$tmp_config"

    # Eval all checkpoints (vLLM offline)
    python new_npo/scripts/run_eval_local.py \
        --model-dir "$output_dir" \
        --result-name "$result_name"
done

# ─── 결과 요약 ────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "All experiments complete. Results:"
echo "══════════════════════════════════════════════════════════════"
for exp in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r af ar akl model_name result_name <<< "$exp"
    echo "── alpha_f=${af}, alpha_r=${ar}, alpha_kl=${akl} ──"
    cat "local/${result_name}/result/summary.log" 2>/dev/null || echo "  (no results)"
done
