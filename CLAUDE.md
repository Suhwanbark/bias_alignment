# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research repository for "Your AI, Not Your View: The Bias of LLMs in Investment Analysis" (ICAIF 2025, arXiv: 2507.20957). Investigates systematic biases in LLMs when making financial investment decisions, and experiments with debiasing via DPO, SFT fine-tuning, and NPO (BiasUnlearn) unlearning.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # Add OPENROUTER_API_KEY
```

LlamaFactory (v0.9.3) is installed at `LLaMA-Factory/` (editable mode, patched for Python 3.10 + transformers 4.57).

For local GPU inference (debiasing): requires 2+ H200/A100, CUDA 12.x, vLLM.

## Commands

```bash
# 통합 실행 스크립트 (run.sh)
bash run.sh sft_v7                  # 훈련만
bash run.sh sft_v7 --eval           # 훈련 + 마지막 체크포인트 평가
bash run.sh sft_v7 --eval-only      # 훈련 스킵, 마지막 체크포인트 평가만
bash run.sh sft_v7_4b --eval        # 4B 모델 (자동으로 TP=1 + merge_4b.yaml)
bash run.sh dpo_v1 --eval-only      # DPO 평가

# 개별 실행 (필요 시)
cd debias/llamafactory
bash run_train.sh configs/sft_v7.yaml      # SFT training only
bash run_train.sh configs/dpo_v1.yaml      # DPO training only

# LoRA merge (개별)
sed -e 's|ADAPTER_PATH|/path/to/checkpoint|' -e 's|EXPORT_DIR|/path/to/output|' \
    configs/merge.yaml > /tmp/merge.yaml    # 30B
    configs/merge_4b.yaml > /tmp/merge.yaml # 4B
DISABLE_VERSION_CHECK=1 llamafactory-cli export /tmp/merge.yaml

# vLLM server (local inference)
cd debias
./vllm qwen            # Qwen3-30B original
./vllm debias qwen     # Qwen3-30B debiased
./vllm stop            # Stop server
./vllm status          # Check status
./vllm list            # List supported model aliases

# Bias evaluation (local vLLM)
cd local
python bias_attribute.py \
    --model-id /path/to/merged/model \
    --vllm-url http://localhost:8000/v1 \
    --temperature 0.6 --seed 42 \
    --num-sets 3 --num-trials 10 \
    --max-workers 200 --output-dir ./result

# NPO / BiasUnlearn training (new_npo)
cd new_npo
bash scripts/run_train.sh configs/qwen3_30b.yaml
bash scripts/run_eval.sh                         # ID + OOD evaluation
bash scripts/run_sweep.sh                        # Hyperparameter sweep
bash scripts/run_pipeline.sh                     # Full train → eval pipeline
```

No test suite or linting configuration exists in this repository.

## Architecture

### run.sh — 통합 실행 스크립트

`run.sh`는 훈련과 평가를 하나의 명령어로 처리하는 루트 레벨 스크립트.

- YAML의 `model_name_or_path`에서 모델 크기를 자동 감지 (4B → TP=1 + `merge_4b.yaml`, 그 외 → TP=2 + `merge.yaml`)
- `--eval` 모드: 마지막 체크포인트만 평가 (LoRA merge → vLLM serve → bias_attribute.py → cleanup)
- 결과: `local/<config_name>/result/step_<N>/`에 CSV + JSON 저장, `summary.log`도 함께 생성
- 로그: `debias/logs/<config_name>_<timestamp>.log`
- 체크포인트가 이미 존재하면 훈련을 자동 스킵 (재평가만 가능)
- bias_attribute.py에 `--tag step_<N>` 전달; merged 모델은 `/home/jovyan/models/_tmp_merged`에 임시 저장 후 평가 완료 시 삭제

### Training Pipeline (LlamaFactory)

```
debias/llamafactory/
├── data/sft_v1~v8.jsonl, dpo_v1~v13.jsonl   # Training data
├── configs/sft_v1~v8.yaml, dpo_v1~v13.yaml  # YAML configs (30B)
│        sft_v7_4b.yaml                       # YAML configs (4B)
│        merge.yaml, merge_4b.yaml            # LoRA merge templates
├── dataset_info.json                          # Dataset registry
└── run_train.sh                               # Wrapper (dynamic eval_steps)

Flow:
  YAML config → run_train.sh (compute eval_steps) → llamafactory-cli train
  → checkpoints → llamafactory-cli export (LoRA merge) → vLLM serve → eval
```

`run_train.sh` dynamically calculates `eval_steps = save_steps = max(steps_per_epoch // 2, 1)` from dataset size, then injects both into a temp YAML before calling `llamafactory-cli train`. The `save_steps: 1` and `eval_steps: 1` in YAML configs are placeholders — always use `run_train.sh`, never call `llamafactory-cli train` directly.

**IMPORTANT**: Always set `DISABLE_VERSION_CHECK=1` when calling `llamafactory-cli` (v0.9.3 is patched to work with transformers 4.57).

### Bias Measurement Pipeline

```
data/sp500_final.csv + evidence_corpus_{qual,quant}_mixed.csv
    ↓
local/bias_attribute.py: build_prompt() generates per-ticker prompts
  (2 buy + 2 sell evidence, forced JSON decision)
    ↓
ThreadPoolExecutor parallel inference → per-set CSVs
    ↓
local/result_attribute.py: t-tests for sector/size bias → JSON summary
    ↓
bias_index = (sector_composite + size_composite) / 2
```

### Supported Models

| Model | Config suffix | TP | Merge template | Baseline BI |
|-------|--------------|:---:|---------------|:-----------:|
| Qwen3-30B-A3B-Instruct-2507 | (default) | 2 | merge.yaml | 172 |
| Qwen3-4B-Instruct-2507 | `_4b` | 1 | merge_4b.yaml | 357 |

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

### NPO / BiasUnlearn Pipeline (new_npo)

BiasUnlearn (Liu et al., EMNLP 2025)의 dual-pathway unlearning을 투자 편향 도메인에 적용. LlamaFactory를 사용하지 않고 HuggingFace transformers + PEFT를 직접 사용한다.

**핵심 아이디어**: 편향된 투자 판단을 "잊고"(forget), 균형 잡힌 판단을 "유지"(retain), 일반 금융 지식은 보존(KL regularization).

```
new_npo/
├── src/
│   ├── train.py              # 메인 학습 (3-loss loop, dynamic swapping, early stopping)
│   ├── losses.py             # npo_loss, ce_retention_loss, kl_divergence_loss, combined_loss
│   ├── dataset.py            # BiasUnlearnDataset, create_dataloaders
│   └── model_utils.py        # load_base_model, apply_lora, load_config
├── configs/qwen3_30b.yaml    # 학습 설정 (beta=0.1, lr=2e-5, max_steps=1000)
├── data/qwen3-30b-a3b-instruct-2507/
│   ├── forget.jsonl           # 고편향(|bias_score|>80) 종목의 다수방향 응답 (~190 samples)
│   ├── retain.jsonl           # 저편향 종목 응답 + counterfactual + minority (~840 samples)
│   ├── retain_counterfactual.jsonl  # Anonymized 프롬프트로 base model re-inference
│   ├── eval_mini.jsonl        # Dynamic swapping/early stopping 평가용
│   └── profile_report.json    # Phase 0 종목별 bias_score
├── bias_profiling/            # Phase 0 프로파일링 (Gemma-3-27B, OLMo-3.1-32B, Qwen3-30B)
├── eval/
│   ├── eval_id.py             # In-Distribution 평가
│   └── eval_ood.py            # Out-of-Distribution 평가 (실제 earnings call)
├── scripts/
│   ├── run_train.sh           # 학습 실행
│   ├── run_eval.sh            # 평가 실행
│   ├── run_sweep.sh           # Loss weight ablation (alpha_f/alpha_r 조합 3가지)
│   ├── run_pipeline.sh        # 전체 파이프라인 (counterfactual 생성 → retain 병합 → sweep 학습 → eval)
│   └── run_eval_local.py      # vLLM offline 평가 (LoRA merge → serve → bias_attribute.py)
├── prepare_dataset.py         # Phase 0 CSV → forget/retain JSONL 생성
└── prepare_counterfactual.py  # Anonymize + base model re-inference → counterfactual retain
```

**3-Loss 학습**:

$$L_{total} = α_f \cdot L_{forget}(NPO) + α_r \cdot L_{retain}(CE) + α_{kl} \cdot L_{KL}$$

| Component | Loss | 역할 | Default α |
|-----------|------|------|:---------:|
| Forget (NPO) | `-2/β · log σ(-β · log(π_θ/π_ref))` | 편향 응답 확률 감소 | 0.4 |
| Retain (CE) | Standard cross-entropy | 균형 판단 유지 | 0.4 |
| KL | Forward KL divergence | 일반 금융 지식 보존 | 0.2 |

KL 데이터 없으면 자동으로 NPO+CE 2-loss 모드 (α_f, α_r 재정규화).

**LoRA 설정** (DPO/SFT와 다름): r=8, alpha=16, dropout=0.1, targets=q_proj,k_proj,v_proj,o_proj

**학습 하이퍼파라미터**: lr=2e-5, beta=0.1, max_steps=1000, batch=forget 2 + retain 14 (1:7 ratio), grad_accum=2, warmup=10, eval/save every 50 steps

**Reference model**: `model.disable_adapter()` 사용 (별도 ref_model 로드 불필요)

**Dynamic Swapping**: 매 50 step마다 eval_mini에서 bias_score 계산. 원래 buy-biased 종목이 sell로 반전(< -10)되면 forget/retain 교환.

**Early Stopping**: 고편향 종목의 mean |bias_score| < 20 이면 학습 중단.

**Sweep 실험**: `run_pipeline.sh`가 3가지 alpha 비율로 학습+평가 자동 실행:
- A: α_f=0.75, α_r=0.25
- B: α_f=0.5, α_r=0.5
- C: α_f=0.25, α_r=0.75

**실행**: 항상 repo root에서 `python -m new_npo.src.train --config new_npo/configs/qwen3_30b.yaml`

**npo/ vs new_npo/**: `npo/`는 초기 구현 (forget-only, forget+retain, FLAT 모드 지원, r=8/α=16). `new_npo/`는 BiasUnlearn 논문 충실 구현 (counterfactual retain, 1:7 batch ratio, dynamic swapping, early stopping). **`new_npo/`가 현재 메인 개발 디렉토리**.

### Key Patterns

- **Bias index**: composite metric, lower = less biased. Baseline Qwen3-30B = 172, Qwen3-4B = 357
- **Decision order alternation**: half trials use `[buy | sell]`, half `[sell | buy]` to control order bias
- **Evidence format**: each prompt contains exactly 2 positive + 2 negative evidence items → expected 50% buy rate under no bias
- **Reproducibility**: Qwen3-30B is reproducible with fixed seed (bf16). gpt-oss-20b is NOT (MXFP4 quantization)
- **LoRA config**: r=32, alpha=64, dropout=0.05, seed=42. SFT targets `qkv_proj,o_proj` (fused, 30B MoE) or `q_proj,k_proj,v_proj,o_proj` (4B dense). DPO targets `q_proj,k_proj,v_proj,o_proj`.
- **Global shift behavior**: LoRA fine-tuning causes global BUY/SELL ratio shift across ALL tickers, not selective per-ticker correction
- **4B notes**: Qwen3-4B produces ~5% "hold" responses (parse failures), has much higher baseline BUY bias (68.9% vs 59.0%), and 136 tickers with buy_rate >= 80% (vs 12 for 30B)

### Directory Layout

```
/
├── run.sh                     # ★ 통합 실행 (DPO/SFT 훈련 + 평가)
├── data/                       # S&P 500 metadata + evidence corpus
│   ├── sp500_final.csv         #   427종목 메타데이터
│   ├── evidence_corpus_{qual,quant}_mixed.csv  # 합성 증거
│   ├── evidence_corpus_view.csv                # View 증거
│   └── gemini/, kimi/, mini/   #   모델별 증거 코퍼스 변형
│
├── new_npo/                    # ★ BiasUnlearn NPO (메인 개발)
│   ├── src/                    #   train.py, losses.py, dataset.py, model_utils.py
│   ├── configs/                #   qwen3_30b.yaml
│   ├── data/                   #   모델별 forget/retain/eval JSONL
│   ├── bias_profiling/         #   Phase 0 프로파일링 결과 (3 모델)
│   ├── eval/                   #   eval_id.py, eval_ood.py
│   ├── scripts/                #   run_train.sh, run_eval.sh, run_sweep.sh, run_pipeline.sh
│   ├── prepare_dataset.py      #   Phase 0 CSV → forget/retain JSONL
│   └── prepare_counterfactual.py  # Counterfactual retain 생성
│
├── npo/                        # 초기 NPO 구현 (forget-only/retain/FLAT)
│   ├── train.py                #   직접 transformers+PEFT 사용
│   ├── prepare_dataset.py
│   ├── bias_profiling/         #   3 모델 프로파일링 결과
│   └── data/                   #   모델별 forget/retain JSONL
│
├── debias/                     # DPO/SFT Debiasing (LlamaFactory)
│   ├── llamafactory/           # ★ Training
│   │   ├── data/               #   sft_v1~v8.jsonl, dpo_v1~v13.jsonl
│   │   ├── configs/            #   sft_v*.yaml, dpo_v*.yaml, merge*.yaml
│   │   ├── dataset_info.json   #   Dataset registry
│   │   └── run_train.sh        #   Training wrapper
│   ├── config.py               # Shared constants (tickers, perspectives)
│   ├── vllm                    # vLLM server management
│   ├── docs/                   # Reports (MD + PDF)
│   ├── logs/                   # Training/eval logs (NPO 로그도 여기)
│   ├── models/                 # LoRA adapters (.gitignored)
│   └── plots/                  # Visualizations
│
├── local/                      # vLLM inference + evaluation
│   ├── bias_attribute.py       # Bias measurement (VLLMClient, auto-computes bias_index)
│   ├── llm_clients.py          # VLLMClient only (OpenAI SDK, Qwen thinking=False)
│   ├── result_attribute.py     # Post-hoc statistical analysis
│   └── (result directories)    # Per-experiment evaluation results
│
├── models/                     # Base model (Qwen3-30B only)
├── LLaMA-Factory/              # Training framework (v0.9.3, patched)
│
├── legacy/                     # Archived files (not actively used)
│   ├── debias/                 #   Old trl scripts, sft_v*/sector_v* folders, data generators
│   ├── models/                 #   Nemotron, GLM, gpt-oss models
│   └── (root scripts)          #   OpenRouter API experiments
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

Detailed reports: `debias/docs/debiasing_summary_for_prof_kim.md` (EN) / `*_ko.md` (KR)
