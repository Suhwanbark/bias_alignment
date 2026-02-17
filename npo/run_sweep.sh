#!/bin/bash
# ══════════════════════════════════════════════════════════════
# NPO alpha sweep: 3 forget+retain experiments sequentially
#
# 각 실험: train (npo/train.py) → eval (npo/run_eval.sh)
#
# 사용법:
#   bash npo/run_sweep.sh
#
# 실험:
#   A: alpha_forget=0.5,  alpha_retain=0.5
#   B: alpha_forget=0.75, alpha_retain=0.25
#   C: alpha_forget=0.25, alpha_retain=0.75
# ══════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

# ─── 실험 정의 (alpha_forget:alpha_retain:model_name:result_name) ───
EXPERIMENTS=(
    "0.5:0.5:qwen-npo-retain-a05r05:npo_retain_a05r05"
    "0.75:0.25:qwen-npo-retain-a75r25:npo_retain_a75r25"
    "0.25:0.75:qwen-npo-retain-a25r75:npo_retain_a25r75"
)

COMMON_ARGS=(
    --model-name Qwen/Qwen3-30B-A3B-Instruct-2507
    --forget-data npo/data/qwen3-30b-a3b-instruct-2507/forget.jsonl
    --retain-data npo/data/qwen3-30b-a3b-instruct-2507/retain.jsonl
    --beta 0.1 --lr 2e-5 --max-steps 500 --save-steps 50
    --batch-size 4 --grad-accum 4 --lora-r 8 --seed 42
)

echo "══════════════════════════════════════════════════════════════"
echo "NPO Alpha Sweep: ${#EXPERIMENTS[@]} experiments"
echo "══════════════════════════════════════════════════════════════"

for exp in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r af ar model_name result_name <<< "$exp"
    echo ""
    echo "════════════════════════════════════════"
    echo "Training: alpha_forget=${af}, alpha_retain=${ar}"
    echo "  Model: /home/jovyan/models/${model_name}"
    echo "  Result: local/${result_name}/result/"
    echo "════════════════════════════════════════"

    # Train
    python npo/train.py "${COMMON_ARGS[@]}" \
        --output-dir "/home/jovyan/models/${model_name}" \
        --alpha-forget "$af" --alpha-retain "$ar"

    # Eval all checkpoints
    bash npo/run_eval.sh "$model_name" "$result_name"
done

# ─── 결과 요약 ────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "All experiments complete. Results:"
echo "══════════════════════════════════════════════════════════════"
for exp in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r af ar model_name result_name <<< "$exp"
    echo "── alpha_f=${af}, alpha_r=${ar} ──"
    cat "local/${result_name}/result/summary.log" 2>/dev/null || echo "  (no results)"
done
