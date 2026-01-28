# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research repository investigating biases in Large Language Models when making financial investment decisions. Published as "Your AI, Not Your View: The Bias of LLMs in Investment Analysis" at ICAIF 2025 (arXiv: 2507.20957).

**Leaderboard:** https://linqalpha.com/leaderboard

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set OpenRouter API key
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY
```

## Quick Reproduce (Debiasing 전체 파이프라인)

```bash
# 1. 레포 클론
git clone https://github.com/Suhwanbark/bias_alignment.git
cd bias_alignment

# 2. 의존성 설치
pip install -r requirements.txt

# 3. 환경변수 설정
cp .env.example .env
# .env 파일에 OPENROUTER_API_KEY 입력

# 4. vLLM 서버 시작 (GPU 필요, H200/A100 권장)
cd debias
./vllm gp   # gpt-oss-20b 자동 다운로드 → ./models/에 저장

# 5. DPO 데이터 생성
python generate_events.py --target nvidia --output data/events_nvidia.json
python generate_events.py --target qwen --output data/events_qwen.json

python generate_dpo_dataset.py --events data/events_nvidia.json --num-samples 1000 --output data/dpo_nvidia.jsonl
python generate_dpo_dataset.py --events data/events_qwen.json --num-samples 1000 --output data/dpo_qwen.jsonl

# 6. DPO 훈련 (trl/axolotl 사용 - 별도)

# 7. Bias 재측정
cd ..
python bias_attribute.py --model-id "훈련된모델" --output-dir ./result
python result_attribute.py --model-id "훈련된모델" --output-dir ./result
```

**필요 환경:** GPU (H200/A100), Python 3.10+, CUDA 12.x

## Running Experiments

### Main Experiments
```bash
# Full experiment suite (attribute + strategy bias)
bash run.sh

# Individual experiments
python bias_attribute.py --model-id "openai/gpt-4.1" --temperature 0.6 --num-sets 2 --num-trials 10
python bias_strategy.py --model-id "openai/gpt-4.1" --temperature 0.6 --num-sets 2
```

### Result Aggregation
```bash
python result_attribute.py --model-id "openai/gpt-4.1" --output-dir ./exp_result
python result_strategy.py --model-id "openai/gpt-4.1" --output-dir ./exp_result
```

### Verification
```bash
bash run_verification.sh
python bias_verification_score.py --model-id "openai/gpt-4.1" --json-path ./mixed_result/gpt-4.1_att_result.json
```

### Visualization
```bash
python visualize_bias.py      # Generate bias heatmaps
python plot_flip_rate.py      # Decision flip rate charts
```

## Architecture

### Core Components

**LLM Client (`llm_clients.py`)**: Unified client for all LLM providers via OpenRouter API. Handles retry logic, rate limiting, cost tracking, and supports both standard and reasoning models (o1, o3, gpt-5).

**Experiment Scripts**:
- `bias_attribute.py` - Tests bias for stock attributes (sector, market cap)
- `bias_strategy.py` - Tests bias for investment strategies (momentum vs contrarian)
- Both use `ThreadPoolExecutor` for parallel inference (default 40 workers)

**Result Aggregation**:
- `result_attribute.py` - Combines CSVs, runs t-tests for sector/size bias
- `result_strategy.py` - Combines CSVs, runs chi-squared test for strategy preference

**Verification**:
- `bias_verification_score.py` - Tests if models flip decisions when evidence is manipulated

### Data Flow

```
S&P 500 Data + Evidence Corpus
    ↓
Prompt Generation → Parallel LLM Inference → Raw CSV Results
    ↓
Statistical Analysis (t-test, chi-squared) → JSON Summaries
    ↓
Visualizations (heatmaps, flip rate charts)
```

### Key Files

| File | Purpose |
|------|---------|
| `llm_clients.py` | OpenRouter API client with retry, cost tracking |
| `data/sp500_final.csv` | S&P 500 stock metadata |
| `data/evidence_corpus_*.csv` | Investment evidence for prompts |
| `run.sh` | Configure MODEL_ID, TEMPERATURE, MAX_WORKERS, NUM_TRIALS |

### Output Formats

- Raw results: `{model}_{experiment}_set{n}.csv` - Individual trial data
- Aggregated: `{model}_att_result.json`, `{model}_str_result.json` - Stats and p-values
- Metrics: `{model}_att_metrics.json` - Cost, tokens, latency
- Visualizations: `bias_heatmap_sector.png`, `bias_heatmap_size.png`

## Key Patterns

- Model short IDs extracted from OpenRouter paths (e.g., "openai/gpt-4.1" → "gpt-4.1")
- Reasoning effort parameter affects API endpoint and timeout values
- Bias index is composite: `(sector_composite + size_composite) / 2`
- LLM output parsing uses JSON extraction with regex fallback

## Debiasing Experiments (DPO)

### Overview
DPO(Direct Preference Optimization)를 사용하여 극단적 bias를 가진 모델을 교정하는 PoC 실험.

### Current Status

| Model | Status | DPO Data | Trained Model |
|-------|--------|----------|---------------|
| Qwen3-30B | **완료** | `data/dpo_qwen.jsonl` (1020 samples) | `models/qwen-debiased-merged` |
| NVIDIA Nemotron | 데이터 생성 완료 | `data/dpo_nvidia.jsonl` (990 samples) | 미훈련 (mamba-ssm 의존성 이슈) |

### Target Models & Tickers

| Model | Bias Direction | # Tickers | Target |
|-------|---------------|-----------|--------|
| NVIDIA Nemotron | SELL (buy_rate=0%) | 22 | BUY로 교정 |
| Qwen3 | BUY (buy_rate>=90%) | 12 | SELL로 교정 |

### Training Results (Qwen3-30B)

```
Model: Qwen/Qwen3-30B-A3B-Instruct-2507
Training: DPO with LoRA (r=16, alpha=32)
Steps: 64 | Runtime: 16:52 | Final Loss: 0.651
Output: debias/models/qwen-debiased-merged (57GB)
```

### Quick Start

```bash
cd debias

# 1. DPO 데이터 생성 (5 perspectives × N tickers × M variations)
./vllm qwen  # 생성용 모델 서빙
python generate_dpo_descriptions.py --target qwen --output data/dpo_qwen.jsonl

# 2. DPO 훈련
python train_dpo.py \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --data data/dpo_qwen.jsonl \
    --output models/qwen-debiased

# 3. LoRA merge (adapter → full model)
python -c "
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-30B-A3B-Instruct-2507', torch_dtype=torch.bfloat16, device_map='auto')
model = PeftModel.from_pretrained(base, './models/qwen-debiased')
merged = model.merge_and_unload()
merged.save_pretrained('./models/qwen-debiased-merged')
AutoTokenizer.from_pretrained('Qwen/Qwen3-30B-A3B-Instruct-2507').save_pretrained('./models/qwen-debiased-merged')
"

# 4. Debiased 모델로 bias 재측정
./vllm debias qwen  # debiased 모델 서빙
cd ../local
python bias_attribute.py --model-id "./models/qwen-debiased-merged" --vllm-url "http://localhost:8000/v1" --output-dir ./result/debiased
```

### vLLM Server
```bash
cd debias
./vllm gp           # gpt-oss-20b 서빙
./vllm qwen         # Qwen3-30B 서빙 (원본)
./vllm debias qwen  # Qwen3-30B 서빙 (debiased)
./vllm nemotron     # Nemotron 서빙
./vllm stop         # 서버 종료
./vllm status       # 상태 확인
./vllm list         # 모델 목록
```

### DPO Data Format (5 Perspectives)

Perspectives: growth, financial, competitive, valuation, macro

```json
{
  "prompt": "Analyze AAPL (Apple Inc.) from a growth perspective.",
  "chosen": "Apple's growth story is losing momentum...",
  "rejected": "Apple demonstrates exceptional growth potential...",
  "metadata": {
    "ticker": "AAPL",
    "perspective": "growth",
    "target_model": "qwen",
    "correction_direction": "buy_to_sell"
  }
}
```

### Folder Structure
```
debias/
├── generate_dpo_descriptions.py  # DPO 데이터 생성 (5 perspectives)
├── train_dpo.py                  # DPO 훈련 (LoRA)
├── config.py                     # 설정 (ticker 목록, 프롬프트, perspectives)
├── llm_client.py                 # vLLM 클라이언트
├── vllm                          # vLLM 서버 실행 스크립트
├── data/
│   ├── dpo_nvidia.jsonl          # NVIDIA용 DPO 데이터 (990 samples)
│   └── dpo_qwen.jsonl            # Qwen용 DPO 데이터 (1020 samples)
└── models/
    ├── qwen-debiased/            # LoRA adapter (51MB)
    └── qwen-debiased-merged/     # Merged full model (57GB)
```

---

## Embedding 시각화 실험

### 목표
bias가 심한 모델 vs 적은 모델이 생성한 시나리오의 embedding 분포를 t-SNE로 시각화하여 비교

### 가설
bias가 심한 모델(Nemotron, SELL bias)은 부정적 방향으로 치우친 시나리오를 생성하고, 이것이 embedding 공간에서 클러스터링될 것

### 대상 모델
| 모델 | Bias 특성 | 역할 |
|------|----------|------|
| gpt-oss-20b | 편향 적음 | baseline (저편향) |
| NVIDIA Nemotron | SELL bias 강함 | 고편향 모델 |

### 대상 티커 (22개)
NVIDIA_TICKERS (SELL bias 티커):
```
ECL, IFF, NWSA, META, CCL, PEP, DG, PFG, COF, MTB, GL, HBAN,
BAX, HCA, GWW, PH, CHRW, ITW, DLR, WY, EA, D
```

### 실행 방법
```bash
cd /data/llm-bias-in-finance

# 1. gpt-oss-20b 시나리오 생성 (저편향)
# ./debias/vllm gp 로 서빙 후:
python emb/generate_scenarios.py \
    --model-id "openai/gpt-oss-20b" \
    --output emb/data/scenarios_gptoss.json \
    --num-per-ticker 10

# 2. NVIDIA Nemotron 시나리오 생성 (고편향)
# ./debias/vllm nemotron 로 서빙 후:
python emb/generate_scenarios.py \
    --model-id "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16" \
    --output emb/data/scenarios_nemotron.json \
    --num-per-ticker 10

# 3. Embedding 생성
python emb/embed_scenarios.py \
    --input emb/data/scenarios_gptoss.json \
    --output emb/data/emb_gptoss.npy

python emb/embed_scenarios.py \
    --input emb/data/scenarios_nemotron.json \
    --output emb/data/emb_nemotron.npy

# 4. t-SNE 시각화
python emb/visualize_embeddings.py \
    --emb1 emb/data/emb_gptoss.npy \
    --emb2 emb/data/emb_nemotron.npy \
    --labels "gpt-oss,Nemotron" \
    --output emb/data/tsne_gptoss_vs_nemotron.png
```

### Folder Structure
```
emb/
├── generate_scenarios.py    # 시나리오 생성 (vLLM API)
├── embed_scenarios.py       # sentence-transformers embedding
├── visualize_embeddings.py  # t-SNE 시각화
└── data/
    ├── scenarios_gptoss.json
    ├── scenarios_nemotron.json
    ├── emb_gptoss.npy
    ├── emb_nemotron.npy
    └── tsne_gptoss_vs_nemotron.png
```
