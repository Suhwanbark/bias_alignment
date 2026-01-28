# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research repository investigating biases in Large Language Models when making financial investment decisions. Published as "Your AI, Not Your View: The Bias of LLMs in Investment Analysis" at ICAIF 2025 (arXiv: 2507.20957).

**Leaderboard:** https://linqalpha.com/leaderboard

## Setup

```bash
# Install dependencies (no requirements.txt)
pip install pandas scipy numpy requests python-dotenv tqdm matplotlib seaborn

# Set OpenRouter API key
export OPENROUTER_API_KEY="your-key"
# Or add to .env file
```

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
