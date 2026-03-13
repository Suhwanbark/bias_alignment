#!/bin/bash
# ══════════════════════════════════════════════════════════════
# NPO Gemma 전체 평가: step 50, 100, 150 순차 실행
#
# 각 step: LoRA merge → vLLM serve → bias_attribute.py → cleanup
# 마지막에 시각화
#
# 사용법:
#   bash run_npo_eval.sh
# ══════════════════════════════════════════════════════════════
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

ADAPTER_DIR="models/adapters/npo_gemma_retain"
MERGED_DIR="models/merged/_tmp_merged"
RESULT_BASE="npo/results/npo_gemma_retain"
BASE_MODEL="google/gemma-3-27b-it"
STEPS=(50 100 150)
PORT=8000

# ─── Utilities ───
kill_vllm() {
    pkill -9 -f "vllm" 2>/dev/null || true
    sleep 10
}

wait_vllm() {
    echo "  vLLM 대기 중..."
    for i in $(seq 1 120); do
        if python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')" 2>/dev/null; then
            echo "  vLLM 준비 완료 (${i}*5초)"
            return 0
        fi
        sleep 5
    done
    echo "  [ERROR] vLLM 타임아웃"
    return 1
}

merge_lora() {
    local adapter_path="$1"
    local output_path="$2"
    echo "  LoRA merge: $adapter_path → $output_path"
    python3 -c "
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import os, shutil

base = '${BASE_MODEL}'
tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map='cpu', trust_remote_code=True)
model = PeftModel.from_pretrained(model, '${adapter_path}')
model = model.merge_and_unload()
os.makedirs('${output_path}', exist_ok=True)
model.save_pretrained('${output_path}', max_shard_size='5GB')
tokenizer.save_pretrained('${output_path}')

# gemma multimodal config 복사
from huggingface_hub import hf_hub_download
for f in ['preprocessor_config.json', 'processor_config.json']:
    try:
        src = hf_hub_download(base, f)
        shutil.copy(src, '${output_path}/')
    except: pass
print('Merge 완료')
"
}

# ─── Main ───
echo "══════════════════════════════════════════════════════════════"
echo "NPO Gemma 평가: steps ${STEPS[*]}"
echo "══════════════════════════════════════════════════════════════"

kill_vllm

for STEP in "${STEPS[@]}"; do
    CKPT="${ADAPTER_DIR}/checkpoint-${STEP}"
    STEP_RESULT="${RESULT_BASE}/step_${STEP}"

    echo ""
    echo "════════════════════════════════════════"
    echo "Step ${STEP} 평가 시작"
    echo "════════════════════════════════════════"

    # 이미 완료된 경우 스킵
    if ls "${STEP_RESULT}"/*_att_combined.csv 2>/dev/null | head -1 | grep -q .; then
        echo "  [SKIP] 이미 완료됨: ${STEP_RESULT}"
        continue
    fi

    # 1. LoRA merge
    rm -rf "$MERGED_DIR"
    merge_lora "$CKPT" "$MERGED_DIR"

    # 2. vLLM serve
    echo "  vLLM 시작..."
    kill_vllm
    nohup vllm serve "$MERGED_DIR" \
        --tensor-parallel-size 2 \
        --port $PORT \
        --trust-remote-code \
        --max-model-len 4096 \
        > /tmp/vllm_npo_step${STEP}.log 2>&1 &
    VLLM_PID=$!

    if ! wait_vllm; then
        echo "  [ERROR] vLLM 시작 실패, 스킵"
        kill_vllm
        rm -rf "$MERGED_DIR"
        continue
    fi

    # 3. Bias 평가
    echo "  Bias 평가 시작 (3 sets × 10 trials)..."
    mkdir -p "$STEP_RESULT"
    cd eval
    python3 bias_attribute.py \
        --model-id "$MERGED_DIR" \
        --vllm-url "http://localhost:${PORT}/v1" \
        --temperature 0.6 --seed 42 \
        --num-sets 3 --num-trials 10 \
        --max-workers 50 \
        --output-dir "../${STEP_RESULT}" \
        --tag "npo_gemma_step${STEP}"
    cd "$ROOT_DIR"

    # 하위 폴더 정리 (bias_attribute.py가 만드는 중첩 폴더)
    INNER=$(find "$STEP_RESULT" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -1)
    if [ -n "$INNER" ]; then
        mv "$INNER"/* "$STEP_RESULT"/ 2>/dev/null || true
        rmdir "$INNER" 2>/dev/null || true
    fi

    # bias_index 계산
    cd eval
    python3 -c "
from bias_attribute import compute_bias_index
import json
result = compute_bias_index('../${STEP_RESULT}', 'npo_gemma_step${STEP}')
if result:
    print(f\"  ★ Step ${STEP} → Bias Index: {result.get('bias_index', 'N/A')}\")
"
    cd "$ROOT_DIR"

    # 4. 정리
    echo "  vLLM 종료..."
    kill_vllm
    rm -rf "$MERGED_DIR"

    echo "  Step ${STEP} 완료"
done

# ─── 시각화 ───
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "시각화 생성"
echo "══════════════════════════════════════════════════════════════"

cd eval
python3 visualize.py \
    "../${RESULT_BASE}/step_50" \
    "../${RESULT_BASE}/step_100" \
    "../${RESULT_BASE}/step_150" \
    --type compare \
    --output "../${RESULT_BASE}/compare.png"

python3 visualize.py \
    "../${RESULT_BASE}/step_150" \
    --type sector \
    --output "../${RESULT_BASE}/sector_step150.png"

python3 visualize.py \
    "../${RESULT_BASE}/step_150" \
    --type ticker-dist \
    --output "../${RESULT_BASE}/ticker_dist_step150.png"
cd "$ROOT_DIR"

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "완료! 결과:"
echo "  ${RESULT_BASE}/step_50/"
echo "  ${RESULT_BASE}/step_100/"
echo "  ${RESULT_BASE}/step_150/"
echo "  ${RESULT_BASE}/compare.png"
echo "══════════════════════════════════════════════════════════════"
