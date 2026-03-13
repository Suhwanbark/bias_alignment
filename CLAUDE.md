# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research repository for "Your AI, Not Your View: The Bias of LLMs in Investment Analysis" (ICAIF 2025, arXiv: 2507.20957). Investigates systematic biases in LLMs when making financial investment decisions, and experiments with debiasing via DPO, SFT fine-tuning, and NPO/SimNPO unlearning.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # Add OPENROUTER_API_KEY
```

- **Python 3.10+** required (LlamaFactory v0.9.3 is patched for 3.10 + transformers 4.57)
- **LlamaFactory v0.9.3**: editable mode install, always use `DISABLE_VERSION_CHECK=1`
- **GPU inference**: 2× A100/H200, CUDA 12.x, vLLM
- **Model paths**: Scripts hardcode `/home/jovyan/models` — edit `run.sh` (line ~41) and `eval/run_eval.sh` (line ~26) if models are elsewhere

## Commands

```bash
# ═══ SFT/DPO: 통합 실행 (run.sh) ═══
bash run.sh sft_v7                  # 훈련만
bash run.sh sft_v7 --eval           # 훈련 + 마지막 체크포인트 평가
bash run.sh sft_v7 --eval-only      # 훈련 스킵, 마지막 체크포인트 평가만
bash run.sh sft_v7_4b --eval        # 4B 모델 (자동으로 TP=1 + merge_4b.yaml)
bash run.sh dpo_v1 --eval-only      # DPO 평가

# ═══ 개별 훈련 (SFT/DPO) ═══
cd sft && bash train.sh configs/sft_v7.yaml     # SFT training
cd dpo && bash train.sh configs/dpo_v1.yaml     # DPO training

# ═══ LoRA merge ═══
sed -e 's|ADAPTER_PATH|/path/to/checkpoint|' -e 's|EXPORT_DIR|/path/to/output|' \
    sft/configs/merge.yaml > /tmp/merge.yaml        # 30B
    sft/configs/merge_4b.yaml > /tmp/merge.yaml     # 4B
DISABLE_VERSION_CHECK=1 llamafactory-cli export /tmp/merge.yaml

# ═══ vLLM server ═══
cd eval
./vllm qwen            # Qwen3-30B original
./vllm debias qwen     # Qwen3-30B debiased
./vllm stop            # Stop server
./vllm status          # Check status
./vllm list            # List supported model aliases

# ═══ Bias evaluation ═══
cd eval
python bias_attribute.py \
    --model-id /path/to/merged/model \
    --vllm-url http://localhost:8000/v1 \
    --temperature 0.6 --seed 42 \
    --num-sets 3 --num-trials 10 \
    --max-workers 200 --output-dir ./result

# Or use the standalone eval script:
bash eval/run_eval.sh --base qwen --output ./result
bash eval/run_eval.sh --adapter /path/to/lora --save-merged --output ./result

# ═══ NPO training (npo/) ═══
bash npo/scripts/run_train.sh
bash npo/scripts/run_eval.sh                    # Evaluation
bash npo/scripts/run_sweep.sh                   # Hyperparameter sweep

# ═══ SimNPO training (simple_npo/) ═══
bash simple_npo/scripts/run_train.sh
bash simple_npo/scripts/run_eval.sh
bash simple_npo/scripts/run_sweep.sh

# ═══ NPO evaluation (multi-step) ═══
bash run_npo_eval.sh                            # Gemma-3-27B multi-step eval
```

No test suite or linting configuration exists in this repository.

## Architecture

### run.sh — 통합 실행 스크립트 (SFT/DPO)

`run.sh`는 SFT/DPO 훈련과 평가를 하나의 명령어로 처리하는 루트 레벨 스크립트.

- Config 이름의 prefix (`sft_`/`dpo_`)로 `sft/` 또는 `dpo/` 디렉토리를 자동 선택
- YAML의 `model_name_or_path`에서 모델 크기를 자동 감지 (4B → TP=1 + `merge_4b.yaml`, 그 외 → TP=2 + `merge.yaml`)
- `--eval` 모드: 마지막 체크포인트만 평가 (LoRA merge → vLLM serve → bias_attribute.py → cleanup)
- 결과: `{sft,dpo}/results/<config_name>/result/step_<N>/`에 CSV + JSON 저장
- 로그: `logs/<config_name>_<timestamp>.log`
- 체크포인트가 이미 존재하면 훈련을 자동 스킵

### Training Pipeline (LlamaFactory — SFT/DPO)

SFT와 DPO는 동일한 구조로 각각 `sft/`와 `dpo/` 디렉토리에 분리:

```
sft/ (or dpo/)
├── train.sh              # Wrapper (dynamic eval_steps 계산)
├── dataset_info.json     # Dataset registry
├── configs/              # YAML configs + merge templates
├── data/                 # Training JSONL files
└── results/              # Per-config evaluation results
```

`train.sh`가 `eval_steps = save_steps = max(steps_per_epoch // 2, 1)`을 동적 계산 후 temp YAML에 주입. YAML의 `save_steps: 1`과 `eval_steps: 1`은 placeholder — 항상 `train.sh`를 통해 실행.

**IMPORTANT**: Always set `DISABLE_VERSION_CHECK=1` when calling `llamafactory-cli` (v0.9.3 is patched to work with transformers 4.57).

### Bias Measurement Pipeline (eval/)

```
data/sp500_final.csv + evidence_corpus_{qual,quant}_mixed.csv
    ↓
eval/bias_attribute.py: build_prompt() generates per-ticker prompts
  (2 buy + 2 sell evidence, forced JSON decision)
    ↓
ThreadPoolExecutor parallel inference → per-set CSVs
    ↓
eval/result_attribute.py: t-tests for sector/size bias → JSON summary
    ↓
bias_index = (sector_composite + size_composite) / 2
```

Key files in `eval/`:
- `bias_attribute.py` — Main bias measurement driver (VLLMClient, parallel inference)
- `result_attribute.py` — Post-hoc statistical analysis (t-tests)
- `llm_clients.py` — VLLMClient wrapper (OpenAI SDK, retry logic, Qwen thinking=False)
- `config.py` — Shared constants (ticker lists, LLM defaults, prompt templates)
- `vllm` — vLLM server management script (start/stop/status)
- `run_eval.sh` — Standalone evaluation with `--base`, `--adapter`, `--save-merged` options
- `visualize.py` — Plot bias results (compare, sector, ticker-dist)

### NPO Pipeline (npo/)

초기 NPO 구현. HuggingFace transformers + PEFT 직접 사용 (LlamaFactory 불필요).

Three loss modes in `npo/train.py`:
1. **Forget-only**: NPO loss만 사용
2. **Forget + Retain**: NPO + CE retention loss (global shift 방지)
3. **FLAT mode**: f-divergence unlearning (reference model 불필요, ICLR 2025)

```
npo/
├── train.py              # 3-mode NPO trainer (ForgetDataset, RetainDataset)
├── prepare_dataset.py    # Phase 0: CSV → forget/retain JSONL
├── bias_profiling/       # Phase 0 profiling (Gemma, OLMo, Qwen)
├── data/                 # forget.jsonl, retain.jsonl, eval_mini.jsonl
├── scripts/              # run_train.sh, run_eval.sh, run_sweep.sh
└── results/              # Per-config evaluation results
```

NPO LoRA: r=8, alpha=16, dropout=0.05, targets=q_proj,k_proj,v_proj,o_proj

### SimNPO Pipeline (simple_npo/) — 현재 메인 개발

SimNPO (Fan et al., NeurIPS 2025)의 reference-free unlearning. NPO 대비 ~50% 빠름.

**핵심 차이**: Forget loss에 reference model이 불필요 — length-normalized negative log probability 사용.

```
simple_npo/
├── src/
│   ├── train.py          # SimNPO trainer (dynamic swapping, early stopping)
│   ├── losses.py         # sim_npo_loss, ce_retention_loss, kl_divergence_loss
│   ├── dataset.py        # BiasUnlearnDataset, KLDataset, create_dataloaders
│   └── model_utils.py    # TrainConfig, load_config, load_base_model, apply_lora
├── configs/
│   ├── qwen3_30b.yaml    # Qwen3-30B config
│   └── gemma3_27b.yaml   # Gemma-3-27B config
├── scripts/              # run_train.sh, run_eval.sh, run_sweep.sh, run_eval_local.py
├── models/               # Adapter checkpoints, merged models
└── results/              # Per-config evaluation results
```

**3-Loss 학습**:

$$L_{total} = α_f \cdot L_{forget}(SimNPO) + α_r \cdot L_{retain}(CE) + α_{kl} \cdot L_{KL}$$

| Component | Loss | Description |
|-----------|------|-------------|
| Forget (SimNPO) | `-2/β · mean(log σ(β · current_loss - γ))` | Reference-free, length-normalized |
| Retain (CE) | Standard cross-entropy | Preserve balanced decisions |
| KL | Forward KL(P_ref ∥ P_θ) via `disable_adapter()` | Preserve general knowledge |

- KL 데이터 없으면 자동으로 2-loss 모드 (α_f, α_r 재정규화)
- LoRA: r=8, alpha=16, dropout=0.1, targets=q_proj,k_proj,v_proj,o_proj
- Training: lr=2e-5, beta=0.1, gamma=0.0, max_steps=150, eval/save every 50 steps
- **Dynamic Swapping**: 매 eval마다 bias reversal 감지 시 forget/retain 교환
- **Early Stopping**: 고편향 종목의 mean |bias_score| < 20 이면 학습 중단

### Supported Models

| Model | Config suffix | TP | Merge template | Baseline BI |
|-------|--------------|:---:|---------------|:-----------:|
| Qwen3-30B-A3B-Instruct-2507 | (default) | 2 | merge.yaml | 172 |
| Qwen3-4B-Instruct-2507 | `_4b` | 1 | merge_4b.yaml | 357 |
| Gemma-3-27B-IT | gemma3_27b | 2 | (PEFT merge) | — |

### Dataset Naming Convention

All datasets use unified `{method}_v{N}` naming:

| SFT | Data | Samples | Epochs |
|-----|------|:---:|:---:|
| sft_v1 | 50/50 balanced | 8,494 | 3 |
| sft_v2 | Sector Q&A (Tech SELL + Financial BUY) | 958 | 10 |
| sft_v3 | Tech SELL + Financial BUY ticker-level | 1,503 | 5 |
| sft_v4 | System prompt + 50/50 | 8,494 | 3 |
| sft_v5 | AAPL single ticker SELL | 996 | 10 |
| sft_v6 | High-bias tickers SELL | 991 | 10 |
| sft_v7 | Targeted SELL (best balance) | 982 | 5 (30B) / 10 (4B) |
| sft_v8 | Non-biased balanced | 382 | 10 |

| DPO | Original Name | Data | Samples |
|-----|---------------|------|:---:|
| dpo_v1 | dpo_qwen | 12 Qwen-biased tickers | 1,020 |
| dpo_v2 | dpo_nvidia | 22 SELL-biased tickers | 990 |
| dpo_v3 | dpo_tech | 58 tech tickers | 1,160 |
| dpo_v4 | dpo_tech_v2 | 58 tech tickers (v2) | 1,160 |
| dpo_v5 | dpo_q1 | 107 large-cap tickers | 1,070 |
| dpo_v6 | dpo_q1_v2 | 107 large-cap tickers (v2) | 1,070 |
| dpo_v7 | dpo_sector | Sector-level bidirectional | 1,000 |
| dpo_v8 | sector_v2/decision | Decision-focused | 853 |
| dpo_v9 | sector_v3/contrastive | Contrastive pairs | 1,410 |
| dpo_v10 | sector_v4/simple | Simple format | 718 |
| dpo_v11 | sector_v5/decision_only | Decision only | 1,180 |
| dpo_v12 | sector_v6/anchored | Anchored examples | 1,798 |
| dpo_v13 | sector_v7/eval_prompt | Evaluation-aligned | 2,037 |

All DPO configs share: epochs=10, beta=0.1, lr=5e-6, lora_target=q_proj,k_proj,v_proj,o_proj.

### Key Patterns

- **Bias index**: composite metric, lower = less biased. Baseline Qwen3-30B = 172, Qwen3-4B = 357
- **Decision order alternation**: half trials use `[buy | sell]`, half `[sell | buy]` to control order bias
- **Evidence format**: each prompt contains exactly 2 positive + 2 negative evidence items → expected 50% buy rate under no bias
- **Reproducibility**: Qwen3-30B is reproducible with fixed seed (bf16). gpt-oss-20b is NOT (MXFP4 quantization)
- **LoRA configs differ by method**:
  - SFT: r=32, alpha=64, dropout=0.05, targets=`qkv_proj,o_proj` (fused, 30B MoE) or `q_proj,k_proj,v_proj,o_proj` (4B)
  - DPO: r=32, alpha=64, dropout=0.05, targets=`q_proj,k_proj,v_proj,o_proj`
  - NPO/SimNPO: r=8, alpha=16, dropout=0.05-0.1, targets=`q_proj,k_proj,v_proj,o_proj`
- **Global shift behavior**: LoRA fine-tuning causes global BUY/SELL ratio shift across ALL tickers, not selective per-ticker correction
- **4B notes**: Qwen3-4B produces ~5% "hold" responses (parse failures), has much higher baseline BUY bias (68.9% vs 59.0%), and 136 tickers with buy_rate >= 80% (vs 12 for 30B)

### Data Formats

**SFT/DPO JSONL** (ShareGPT format via LlamaFactory):
```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

**NPO/SimNPO Forget/Retain JSONL** (custom):
```json
{"context": "prompt text", "completion": "model response"}
```

### Directory Layout

```
/
├── run.sh                     # ★ 통합 실행 (SFT/DPO 훈련 + 평가)
├── run_npo_eval.sh            # NPO multi-step evaluation (Gemma-3-27B)
│
├── sft/                       # ★ SFT Debiasing (LlamaFactory)
│   ├── train.sh               #   Training wrapper (dynamic eval_steps)
│   ├── dataset_info.json      #   Dataset registry
│   ├── configs/               #   sft_v*.yaml, merge*.yaml
│   ├── data/                  #   sft_v1~v8.jsonl
│   └── results/               #   Per-config evaluation results
│
├── dpo/                       # ★ DPO Debiasing (LlamaFactory)
│   ├── train.sh               #   Training wrapper
│   ├── dataset_info.json      #   Dataset registry
│   ├── configs/               #   dpo_v*.yaml, merge*.yaml
│   ├── data/                  #   dpo_v1~v13.jsonl
│   └── results/               #   Per-config evaluation results
│
├── eval/                      # ★ Bias measurement & vLLM inference
│   ├── bias_attribute.py      #   Main bias evaluation driver
│   ├── result_attribute.py    #   Post-hoc statistical analysis
│   ├── llm_clients.py         #   VLLMClient (OpenAI SDK wrapper)
│   ├── config.py              #   Shared constants (tickers, prompts)
│   ├── vllm                   #   vLLM server management
│   ├── run_eval.sh            #   Standalone eval script
│   └── visualize.py           #   Plot bias results
│
├── npo/                       # NPO unlearning (forget-only/retain/FLAT)
│   ├── train.py               #   3-mode trainer (transformers + PEFT)
│   ├── prepare_dataset.py     #   Phase 0: CSV → forget/retain JSONL
│   ├── bias_profiling/        #   Phase 0 profiling results
│   ├── data/                  #   forget.jsonl, retain.jsonl, eval_mini.jsonl
│   ├── scripts/               #   run_train.sh, run_eval.sh, run_sweep.sh
│   └── results/
│
├── simple_npo/                # ★ SimNPO (reference-free, 현재 메인 개발)
│   ├── src/                   #   train.py, losses.py, dataset.py, model_utils.py
│   ├── configs/               #   qwen3_30b.yaml, gemma3_27b.yaml
│   ├── scripts/               #   run_train.sh, run_eval.sh, run_sweep.sh
│   ├── models/                #   Adapter checkpoints
│   └── results/
│
├── data/                      # S&P 500 metadata + evidence corpus
│   ├── sp500_final.csv        #   427종목 메타데이터
│   ├── evidence_corpus_{qual,quant}_mixed.csv  # 합성 증거
│   └── gemini/, kimi/, mini/  #   모델별 증거 코퍼스 변형
│
├── models/                    # Model weights (not in git)
│   ├── base/                  #   Base models (Qwen3-30B, Gemma-3-27B)
│   ├── adapters/              #   LoRA adapter checkpoints
│   └── merged/                #   Fully merged models
│
├── docs/                      # Reports (MD + PDF + plots)
├── logs/                      # Training/eval logs
│
├── archive/                   # Archived implementations (not actively used)
│   ├── debias/                #   Old LlamaFactory-based DPO/SFT
│   ├── local/                 #   Old vLLM inference + evaluation
│   └── new_npo/               #   Old BiasUnlearn NPO (dual-pathway)
│
├── CLAUDE.md
├── README.md
└── requirements.txt
```

### Debiasing Results Summary

| Method | Model | Best Bias Index | Notes |
|--------|-------|:---:|-------|
| Baseline | Qwen3-30B | 172 | |
| Baseline | Qwen3-4B | 357 | 5% hold responses |
| DPO v1 / dpo_qwen (step 171) | 30B | **46** | Overfits after step 171 |
| SFT v7 / targeted (982 samples) | 30B | **44** | Best overall balance |
| SFT 100% SELL (1000 samples) | 30B | **22** | Lowest bias, overcorrection |
| SFT v7 (step 96, 10 epochs) | 4B | **217** | Systematic shift (-7.4pp) |

Detailed reports: `docs/debiasing_summary_for_prof_kim.md` (EN) / `*_ko.md` (KR)
