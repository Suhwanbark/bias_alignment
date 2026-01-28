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

### Target Models & Tickers

| Model | Bias Direction | # Tickers | Target |
|-------|---------------|-----------|--------|
| NVIDIA Nemotron | SELL (buy_rate=0%) | 22 | BUY로 교정 |
| Qwen3 | BUY (buy_rate>=90%) | 12 | SELL로 교정 |

### Setup (Local vLLM)
```bash
# vLLM 설치
pip install vllm

# 모델은 ./models 폴더에 저장됨
# gpt-oss-20b를 데이터 생성에 사용
```

### Running Debiasing Pipeline
```bash
cd debias

# Step 1: 이벤트 생성 (긍정/부정 뉴스)
python generate_events.py --target nvidia --output data/events_nvidia.json
python generate_events.py --target qwen --output data/events_qwen.json

# Step 2: DPO 데이터셋 생성
python generate_dpo_dataset.py --events data/events_nvidia.json --num-samples 1000 --output data/dpo_nvidia.jsonl
python generate_dpo_dataset.py --events data/events_qwen.json --num-samples 1000 --output data/dpo_qwen.jsonl

# Step 3: DPO 훈련 (trl/axolotl 사용)
# Step 4: 훈련 후 bias 재측정
```

### vLLM Server
```bash
cd debias
./vllm gp        # gpt-oss-20b 서빙
./vllm qwen      # Qwen3-30B 서빙
./vllm nemotron  # Nemotron 서빙
./vllm stop      # 서버 종료
./vllm status    # 상태 확인
./vllm list      # 모델 목록
```

### DPO Data Format
```json
{
  "prompt": "META. Evidence: [+] 광고매출 증가 [-] EU 규제. Should you buy or sell?",
  "chosen": {"decision": "buy", "reason": "Despite regulatory concerns..."},
  "rejected": {"decision": "sell", "reason": "The regulatory headwinds..."}
}
```

### Folder Structure
```
debias/
├── generate_events.py      # Step 1: 이벤트 추출
├── generate_dpo_dataset.py # Step 2: DPO 데이터 생성
├── config.py               # 설정 (ticker 목록, 프롬프트)
├── llm_client.py           # vLLM 클라이언트
├── vllm                    # vLLM 서버 실행 스크립트
└── data/                   # 생성된 데이터
```
