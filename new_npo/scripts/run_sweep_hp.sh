#!/bin/bash
# ══════════════════════════════════════════════════════════════
# NPO Hyperparameter Sweep
#
# Sweeps lr × alpha × batch_ratio combinations
# All use: SUM loss, β=0.1, grad_accum=1, 200 steps
#
# Usage:
#   bash new_npo/scripts/run_sweep_hp.sh
# ══════════════════════════════════════════════════════════════
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/data/llm-bias-in-finance

ROOT_DIR="/data/llm-bias-in-finance"
cd "$ROOT_DIR"

CONFIG="new_npo/configs/qwen3_30b.yaml"
RESULTS_BASE="new_npo/results"
LOG_FILE="${RESULTS_BASE}/sweep_$(date '+%Y%m%d_%H%M%S').log"
mkdir -p "$RESULTS_BASE"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg"
    echo "$msg" >> "$LOG_FILE"
}

log "══════════════════════════════════════════════════════════════"
log "NPO Hyperparameter Sweep"
log "  Base config: ${CONFIG}"
log "  Log: ${LOG_FILE}"
log "══════════════════════════════════════════════════════════════"

# Experiment format: name:lr:alpha_forget:alpha_retain:forget_bs:retain_bs
EXPERIMENTS=(
    # --- batch 1:2 (forget=4, retain=8) ---
    "lr2e5_af25_ar75_b1to2:2e-5:0.25:0.75:4:8"
    "lr1e5_af25_ar75_b1to2:1e-5:0.25:0.75:4:8"
    "lr1e5_af50_ar50_b1to2:1e-5:0.5:0.5:4:8"
    "lr2e5_af50_ar50_b1to2:2e-5:0.5:0.5:4:8"
    # --- batch 1:3 (forget=4, retain=12) ---
    "lr2e5_af25_ar75_b1to3:2e-5:0.25:0.75:4:12"
    "lr1e5_af25_ar75_b1to3:1e-5:0.25:0.75:4:12"
    "lr1e5_af50_ar50_b1to3:1e-5:0.5:0.5:4:12"
    "lr2e5_af50_ar50_b1to3:2e-5:0.5:0.5:4:12"
)

TOTAL=${#EXPERIMENTS[@]}

for i in "${!EXPERIMENTS[@]}"; do
    IFS=':' read -r name lr af ar fbs rbs <<< "${EXPERIMENTS[$i]}"
    exp_num=$((i + 1))
    output_dir="${RESULTS_BASE}/${name}"

    log ""
    log "══ Experiment ${exp_num}/${TOTAL}: ${name} ══"
    log "  lr=${lr}, alpha_f=${af}, alpha_r=${ar}, forget_bs=${fbs}, retain_bs=${rbs}"
    log "  output: ${output_dir}"

    # Skip if already done
    if [ -f "${output_dir}/train_config.json" ] && ls "${output_dir}"/checkpoint-*/adapter_model.safetensors >/dev/null 2>&1; then
        n_ckpts=$(ls -d "${output_dir}"/checkpoint-* 2>/dev/null | wc -l)
        log "  이미 완료 (${n_ckpts} checkpoints), 스킵"
        continue
    fi

    # Create temp config with modified params
    tmp_config=$(mktemp /tmp/npo_sweep_XXXXXX.yaml)
    cp "$CONFIG" "$tmp_config"
    # Use python for reliable YAML field replacement
    python3 -c "
import sys
lines = open('$tmp_config').readlines()
out = []
in_batch = False
in_loss = False
for line in lines:
    stripped = line.strip()
    if stripped.startswith('batch_sizes:'):
        in_batch = True; in_loss = False
    elif stripped.startswith('loss_weights:'):
        in_loss = True; in_batch = False
    elif stripped.startswith('training:') or stripped.startswith('data:') or stripped.startswith('model:') or stripped.startswith('lora:'):
        in_batch = False; in_loss = False

    if stripped.startswith('lr:'):
        line = line.replace(stripped, 'lr: ${lr}')
    elif stripped.startswith('alpha_forget:') and in_loss:
        line = line.replace(stripped, 'alpha_forget: ${af}')
    elif stripped.startswith('alpha_retain:') and in_loss:
        line = line.replace(stripped, 'alpha_retain: ${ar}')
    elif stripped.startswith('forget:') and in_batch:
        line = line.replace(stripped, 'forget: ${fbs}')
    elif stripped.startswith('retain:') and in_batch:
        line = line.replace(stripped, 'retain: ${rbs}')
    elif stripped.startswith('output_dir:'):
        line = line.replace(stripped, 'output_dir: ${output_dir}')
    out.append(line)
open('$tmp_config', 'w').writelines(out)
"

    log "  Config:"
    grep -E "lr:|alpha_forget:|alpha_retain:|beta:|max_steps:|forget:|retain:" "$tmp_config" | while read line; do
        log "    $line"
    done

    # Run training
    python -m new_npo.src.train --config "$tmp_config" 2>&1 | tee -a "$LOG_FILE"
    exit_code=$?

    rm -f "$tmp_config"

    # Extract checkpoint results
    log ""
    log "  ── Results ──"
    if ls "${output_dir}"/checkpoint-*/adapter_model.safetensors >/dev/null 2>&1; then
        log "  Checkpoints saved: $(ls -d "${output_dir}"/checkpoint-* | wc -l)"
    else
        log "  [WARN] No checkpoints found"
    fi
    log "  Experiment ${exp_num}/${TOTAL} 완료 (exit=${exit_code})"
done

# ─── Summary ──────────────────────────────────────────────────
log ""
log "══════════════════════════════════════════════════════════════"
log "SWEEP COMPLETE — Summary"
log "══════════════════════════════════════════════════════════════"

for i in "${!EXPERIMENTS[@]}"; do
    IFS=':' read -r name lr af ar fbs rbs <<< "${EXPERIMENTS[$i]}"
    output_dir="${RESULTS_BASE}/${name}"
    log ""
    log "── ${name} (lr=${lr}, α_f=${af}, α_r=${ar}, bs=${fbs}:${rbs}) ──"
    if [ -d "${output_dir}" ]; then
        # Print checkpoint losses
        grep "Saved checkpoint.*${name}" "$LOG_FILE" 2>/dev/null | sed 's/.*Saved/  Saved/' | while read line; do
            log "$line"
        done
        # Bias reversal count
        reversal_count=$(grep -c "Bias reversal" "${output_dir}/train_config.json" 2>/dev/null || grep -A1 "${name}" "$LOG_FILE" 2>/dev/null | grep -c "Bias reversal" || echo 0)
        # Eval tickers at last checkpoint
        last_eval=$(grep "Step.*eval.*${name}" "$LOG_FILE" 2>/dev/null | tail -1 || echo "")
        if [ -n "$last_eval" ]; then
            log "  Last eval: $last_eval"
        fi
    else
        log "  (결과 없음)"
    fi
done

log ""
log "══════════════════════════════════════════════════════════════"
log "완료: $(date)"
log "로그: ${LOG_FILE}"
log "══════════════════════════════════════════════════════════════"
