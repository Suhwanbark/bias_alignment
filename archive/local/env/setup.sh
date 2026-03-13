#!/bin/bash
# ══════════════════════════════════════════════════════════════
# LLM Bias Local Inference - Environment Setup Script
# ══════════════════════════════════════════════════════════════
#
# 사용법:
#   bash env/setup.sh
#
# 이 스크립트는 다음을 수행합니다:
#   1. Conda 환경 생성 (또는 pip 설치)
#   2. 필요한 패키지 설치
#   3. 환경 검증
#
# ══════════════════════════════════════════════════════════════

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     LLM Bias Local Inference - Environment Setup            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ────────────── Check for conda or pip ──────────────
USE_CONDA=false
if command -v conda &> /dev/null; then
    echo "✓ Conda detected"
    USE_CONDA=true
else
    echo "✓ Conda not found, using pip"
fi

# ────────────── Setup with Conda ──────────────
if [ "$USE_CONDA" = true ]; then
    ENV_NAME="llm-bias-local"

    # Check if environment exists
    if conda env list | grep -q "^${ENV_NAME} "; then
        echo "→ Environment '${ENV_NAME}' already exists"
        echo "→ Activating existing environment..."
    else
        echo "→ Creating conda environment '${ENV_NAME}'..."
        conda env create -f env/environment.yaml
    fi

    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "환경 활성화:"
    echo "  conda activate ${ENV_NAME}"
    echo "════════════════════════════════════════════════════════════"

# ────────────── Setup with pip ──────────────
else
    echo "→ Installing packages with pip..."
    pip install -r requirements.txt

    echo ""
    echo "→ Installing vLLM..."
    pip install vllm
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo "✓ Setup complete!"
echo ""
echo "다음 단계:"
echo "  1. vLLM 서버 실행:"
echo "     vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 \\"
echo "         --tensor-parallel-size 4 --port 8000"
echo ""
echo "  2. 실험 실행:"
echo "     bash run.sh"
echo "════════════════════════════════════════════════════════════"
