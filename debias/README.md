# LLM Debiasing Experiments

LLM 투자 편향 교정 실험: DPO + SFT 기반 파인튜닝

## 폴더 구조

```
debias/
├── README.md                     # 이 파일
├── config.py                     # 설정 (ticker 목록, 프롬프트, perspectives)
├── llm_client.py                 # vLLM 클라이언트
├── vllm                          # vLLM 서버 관리 스크립트
├── accelerate_config.yaml        # Accelerate 학습 설정
│
├── ── 학습 스크립트 ──
├── train_dpo.py                  # DPO 학습 (LoRA)
├── train_sft.py                  # SFT 학습 (LoRA)
│
├── ── DPO 데이터 생성 ──
├── generate_events.py            # 실제 이벤트 추출
├── generate_dpo_dataset.py       # 이벤트 → DPO 데이터셋
├── generate_dpo_descriptions.py  # 5관점 DPO 기술 생성
├── generate_dpo_sector.py        # 섹터 기반 DPO 데이터
│
├── ── 분석/시각화 ──
├── analyze_sentiment.py          # 감성 분석
├── compare_results.py            # 결과 비교
├── recall_descriptions.py        # 품질 검증
├── plot_sft_results.py           # SFT 결과 시각화
│
├── ── 실험 실행 ──
├── run_all_sector.sh             # 전체 DPO 섹터 실험
├── run_all_sft.sh                # 전체 SFT 실험 (v1-v6)
├── eval_checkpoints.sh           # DPO 체크포인트 평가
├── eval_sector_checkpoints.sh    # 섹터 DPO 체크포인트 평가
├── run_sft_eval_parallel.sh      # SFT 병렬 평가
├── run_sft_eval_remaining.sh     # SFT 나머지 평가
│
├── ── 정리된 디렉토리 ──
├── data/                         # 학습 데이터 (DPO/SFT jsonl)
├── models/                       # 학습된 모델 체크포인트
├── docs/                         # 문서 (MD, PDF)
├── plots/                        # 시각화 (PNG)
├── logs/                         # 실행 로그
│
├── ── DPO 실험 (섹터 기반) ──
├── sector_v2/                    # Decision 프롬프트
├── sector_v3/                    # Contrastive 프롬프트
├── sector_v4/                    # Simple 프롬프트
├── sector_v5/                    # Decision-only 프롬프트
├── sector_v6/                    # Anchored 프롬프트
├── sector_v7/                    # Eval 프롬프트 형식
│
├── ── SFT 실험 ──
├── sft_v1/                       # 전체 50/50 균형 (8494 samples)
├── sft_v2/                       # 섹터 Q&A: Tech SELL + Financial BUY
├── sft_v3/                       # Tech SELL + Financial BUY 티커별
├── sft_v4/                       # System Prompt + 50/50
├── sft_v5/                       # AAPL 단일 티커 SELL
├── sft_v6/                       # 고편향 티커 SELL
├── sft_v7_targeted/              # 티커 타겟 교정 (12개 SELL + 800 비타겟)
├── sft_v8_nonbiased/             # Non-biased 선별 (12개 SELL + 200 균형)
│
├── ── SFT Ablation ──
├── sft_v1_ablation/              # V1 SELL 비율 ablation (Q1-Q5)
├── sft_v2_ablation/              # V2 SELL 비율 ablation
├── sft_ablation_phi/             # Phi 모델 ablation
├── sft_ablation_gemma/           # Gemma 모델 ablation
└── sft_ablation_mistral/         # Mistral 모델 ablation
```

## 문서 (docs/)

| 파일 | 내용 |
|------|------|
| `debiasing_summary_for_prof_kim.md` | 전체 실험 요약 보고서 (영문) |
| `debiasing_summary_for_prof_kim_ko.md` | 전체 실험 요약 보고서 (한글) |
| `SFT_EXPERIMENTS.md` | SFT 실험 기록 |
| `SFT_DATA_EXAMPLES.md` | SFT 학습 데이터 예시 |
| `SFT_ABLATION_SETUP.md` | Ablation 실험 설계 |
| `sft_v7v8_summary.md` | V7/V8 비교 요약 |
| `sft_v7v8_examples.md` | V7/V8 학습 데이터 예시 |
| `260205_alignment_v1.pdf` | 정렬 연구 노트 v1 |
| `260211_alignment.pdf` | 정렬 연구 노트 최신 |

## Quick Start

```bash
# 1. vLLM 서버 시작
./vllm gp            # gpt-oss-20b (데이터 생성용)
./vllm qwen          # Qwen3-30B 원본
./vllm debias qwen   # Qwen3-30B 디바이싱 모델
./vllm stop          # 서버 종료

# 2. DPO 학습
python train_dpo.py --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --data data/dpo_qwen.jsonl --output models/qwen-debiased

# 3. SFT 학습
python train_sft.py --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --data sft_v7_targeted/data/sft_targeted.jsonl \
    --output /home/jovyan/models/qwen-sft-v7-targeted

# 4. 전체 실험 실행
bash run_all_sft.sh      # SFT v1-v6 일괄 학습+평가
bash run_all_sector.sh   # DPO sector v2-v7 일괄 학습+평가
```

## 학습 설정 (공통)

- **Backbone**: Qwen/Qwen3-30B-A3B-Instruct-2507
- **LoRA**: r=32, alpha=64, target modules: qkv_proj, o_proj
- **lr**: 5e-6, cosine scheduler, warmup 10%, effective batch=64
- **bf16**, seed=42

## 주요 결과

| 방법 | Best Bias Index | 비고 |
|------|:---:|------|
| Baseline (Qwen3-30B) | 172 | |
| DPO 티커 (12개) | **46** | step 171, 이후 과적합 |
| DPO 섹터 (58개) | 86 | 수렴 느림 |
| DPO 시가총액 (107개) | 65 | |
| SFT V7 (982개) | **44** | 타겟+비타겟, 최적 균형 |
| SFT V8 (382개) | 64 | 비편향 선별 |
| SFT 100% SELL (1000개) | **22** | 최저 bias, 과교정 |

자세한 내용은 `docs/debiasing_summary_for_prof_kim.md` 참조.
