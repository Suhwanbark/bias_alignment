#!/bin/bash
# ══════════════════════════════════════════════════════════════
# Quick Install Script (pip only)
# ══════════════════════════════════════════════════════════════
#
# 쿠버네티스/Docker 환경에서 빠르게 설치하기 위한 스크립트
#
# 사용법:
#   bash env/install.sh
#
# ══════════════════════════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "Installing dependencies..."

# Core packages
pip install --quiet openai pandas scipy numpy tqdm

# vLLM (includes PyTorch, transformers, etc.)
pip install --quiet vllm

# Setup model cache symlink if models exist in project
MODEL_DIR="${PROJECT_DIR}/models/models--Qwen--Qwen3-30B-A3B-Instruct-2507"
if [ -d "$MODEL_DIR" ]; then
    echo "Setting up model cache symlink..."
    mkdir -p ~/.cache/huggingface/hub
    ln -sf "$MODEL_DIR" ~/.cache/huggingface/hub/
    echo "✓ Model cache linked: $MODEL_DIR"
fi

echo ""
echo "✓ Installation complete!"
echo ""
echo "Verify installation:"
echo "  python -c \"import vllm; print(f'vLLM {vllm.__version__}')\""
