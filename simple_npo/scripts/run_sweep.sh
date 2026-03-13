#!/bin/bash
# ══════════════════════════════════════════════════════════════
# SimNPO loss weight & gamma ablation sweep
#
# 각 실험: train → eval all checkpoints (vLLM offline)
#
# 사용법:
#   bash simple_npo/scripts/run_sweep.sh
#
# 실험:
#   A: alpha_f=0.4, alpha_r=0.4, gamma=0.0  (default)
#   B: alpha_f=0.5, alpha_r=0.3, gamma=0.0
#   C: alpha_f=0.3, alpha_r=0.5, gamma=0.0
#   D: alpha_f=0.4, alpha_r=0.4, gamma=1.0  (with margin)
# ══════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR/.."

MODELS_DIR="/home/jovyan/models"
CONFIG="simple_npo/configs/qwen3_30b.yaml"

# ─── 실험 정의 (alpha_f:alpha_r:alpha_kl:gamma:model_name:result_name) ───
EXPERIMENTS=(
    "0.4:0.4:0.2:0.0:qwen-simnpo-a04r04:simnpo_a04r04"
    "0.5:0.3:0.2:0.0:qwen-simnpo-a05r03:simnpo_a05r03"
    "0.3:0.5:0.2:0.0:qwen-simnpo-a03r05:simnpo_a03r05"
    "0.4:0.4:0.2:1.0:qwen-simnpo-a04r04-g1:simnpo_a04r04_g1"
)

echo "══════════════════════════════════════════════════════════════"
echo "SimNPO Sweep: ${#EXPERIMENTS[@]} experiments"
echo "══════════════════════════════════════════════════════════════"

for exp in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r af ar akl gm model_name result_name <<< "$exp"
    output_dir="${MODELS_DIR}/${model_name}"

    echo ""
    echo "════════════════════════════════════════"
    echo "Training: alpha_f=${af}, alpha_r=${ar}, gamma=${gm}"
    echo "  Model: ${output_dir}"
    echo "  Result: simple_npo/results/${result_name}/"
    echo "════════════════════════════════════════"

    # Create temporary config with modified parameters
    tmp_config=$(mktemp /tmp/sweep_config_XXXXXX.yaml)
    sed -e "s/alpha_forget: .*/alpha_forget: ${af}/" \
        -e "s/alpha_retain: .*/alpha_retain: ${ar}/" \
        -e "s/alpha_kl: .*/alpha_kl: ${akl}/" \
        -e "s/gamma: .*/gamma: ${gm}/" \
        -e "s|output_dir: .*|output_dir: ${output_dir}|" \
        "$CONFIG" > "$tmp_config"

    # Train
    python -m simple_npo.src.train --config "$tmp_config"
    rm -f "$tmp_config"

    # Eval all checkpoints (vLLM offline)
    python simple_npo/scripts/run_eval_local.py \
        --model-dir "$output_dir" \
        --result-name "$result_name"
done

# ─── 결과 요약 ────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "All experiments complete. Results:"
echo "══════════════════════════════════════════════════════════════"
for exp in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r af ar akl gm model_name result_name <<< "$exp"
    echo "── alpha_f=${af}, alpha_r=${ar}, gamma=${gm} ──"
    cat "simple_npo/results/${result_name}/summary.log" 2>/dev/null || echo "  (no results)"
done
