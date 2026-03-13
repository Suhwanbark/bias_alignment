# SFT V7 / V8: Ticker-Level Targeted Debiasing 실험 정리

## 배경

Qwen3-30B baseline의 bias_index = **172** (높을수록 편향 심함).
특정 티커에 대해 BUY bias가 극심하므로, 이를 SELL 방향으로 교정하되 나머지 티커에 대한 spillover를 최소화하는 것이 목표.

## 타겟 티커 선정 기준

Baseline 평가(3sets x 10trials)에서 **buy_rate가 높은 12개 티커**를 "biased ticker"로 선정.

| Ticker | Sector | Baseline Buy Rate |
|--------|--------|:-:|
| SO | Utilities | 97% |
| PPL | Utilities | 93% |
| PWR | Industrials | 93% |
| LMT | Industrials | 90% |
| WMT | Consumer Defensive | 87% |
| VRSN | Technology | 87% |
| CVX | Energy | 83% |
| EW | Healthcare | 83% |
| TSN | Consumer Defensive | 83% |
| PYPL | Financial Services | 80% |
| JNJ | Healthcare | 80% |
| NVDA | Technology | 80% |

**평균 buy_rate: 86.4%** (이상적 50% 대비 극심한 BUY bias)

---

## V7: Targeted + 전체 비타겟

### 의도
- 타겟 12개는 **전부 SELL**로 교정
- 나머지 415개 티커는 **원래 buy/sell 비율 그대로** 유지하여 spillover 방어

### 학습 데이터 구성 (982 samples)

| 구분 | 수 | 구성 | 소스 |
|------|--:|------|------|
| **타겟 SELL** | 182 | 12개 티커, **전부 SELL** | sft_balanced 기존 SELL 49개 + gpt-oss-20b 생성 133개 |
| **비타겟** | 800 | 415개 티커, 원래 비율 유지 (BUY 445 / SELL 355) | baseline CSV 서브샘플링 |

- 비타겟: baseline CSV(Qwen3-30B 원본 응답)에서 티커별 buy/sell 비율을 보존하며 800개 서브샘플링
- 타겟 SELL 생성: baseline CSV의 타겟 BUY 프롬프트(evidence 포함)를 gpt-oss-20b에 보내 SELL 응답만 필터링

### 학습 설정
- Base: Qwen/Qwen3-30B-A3B-Instruct-2507
- LoRA r=32, alpha=64, lr=5e-6, **epochs=5**, seed=42
- Effective batch: 8 x 4 x 2GPU = 64

---

## V8: Targeted + Non-biased 티커만

### 의도
- 타겟 12개는 V7과 동일 (전부 SELL)
- 비타겟에서 **non-biased 티커만 선별** (buy_rate 40~60%), 타겟과 **1:1 비율**
- 가설: 이미 균형 잡힌 티커만 넣으면 모델이 "biased vs non-biased" 차이를 더 명확히 학습할 수 있을 것

### Non-biased 티커 선정 기준
- Baseline buy_rate **40% ~ 60%** 범위 (50% 근처)
- 타겟 12개 제외
- **157개 티커** 해당 (평균 buy_rate 51.4%)

### 학습 데이터 구성 (382 samples)

| 구분 | 수 | 구성 | 소스 |
|------|--:|------|------|
| **타겟 SELL** | 182 | 12개 티커, **전부 SELL** | V7과 동일 (sft_balanced 49 + gpt-oss 133) |
| **Non-biased** | 200 | 157개 중 116개 티커, **BUY 100 / SELL 100 (50:50)** | baseline CSV 서브샘플링 |

- 비타겟을 non-biased 티커로 한정하고, BUY/SELL을 정확히 50:50으로 균형
- 데이터 총량이 V7의 ~40% 수준

### 학습 설정
- V7과 동일하되 **epochs=10** (데이터가 적으므로 더 많이 학습)

---

## V7 vs V8 비교

|  | V7 | V8 |
|---|---|---|
| 비타겟 소스 | 전체 415개 티커 | non-biased 157개 중 116개 |
| 비타겟 수 | 800 | 200 |
| 비타겟 BUY:SELL | 445:355 (원래비율) | 100:100 (50:50) |
| 총 데이터 | 982 | 382 |
| 타겟:비타겟 비율 | ~1:4 | ~1:1 |
| Epochs | 5 | 10 |
| Total steps | ~130 | ~100 |

---

## 결과

### Bias Index (낮을수록 좋음, baseline=172)

**V7**

| Step | Bias Index |
|-----:|----------:|
| 18 | 165 |
| 48 | 113 |
| 66 | 79 |
| 96 | **44** |
| 102 | 52 |
| 126 | 46 |

**V8**

| Step | Bias Index |
|-----:|----------:|
| 4 | 177 |
| 6 | 168 |
| 10 | 161 |
| 20 | 143 |
| 78 | **64** |
| 98 | 80 |

### Best Checkpoint 비교

|  | Baseline | V7 (step 96) | V8 (step 78) |
|---|:-:|:-:|:-:|
| **Bias Index** | 172 | **44** | **64** |
| **Biased 12 avg buy_rate** | 86% | 83% (-3%p) | 78% (-8%p) |
| **Non-biased 157 avg buy_rate** | 51% | 44% (-7%p) | 38% (-13%p) |

### 핵심 관찰

1. **V7이 bias_index 더 낮음** (44 vs 64): 비타겟 800개가 모델의 전반적 균형을 유지시킴
2. **V8이 타겟 교정은 더 강함** (78% vs 83%): 데이터가 적지만 SELL 비중이 높아 타겟 방향 교정 더 강력
3. **둘 다 spillover 발생**: non-biased 티커가 baseline 51%에서 V7=44%, V8=38%로 하락 → SELL 방향 과교정
4. **SFT의 한계**: 특정 티커만 선택적으로 교정하지 못하고, 전체적으로 SELL 방향으로 이동. biased ticker보다 non-biased ticker가 오히려 더 많이 움직임

---

## 폴더 구조

```
debias/
├── sft_v7_targeted/
│   ├── generate_data.py
│   ├── run_train_eval.sh
│   ├── data/
│   │   ├── sft_targeted.jsonl        # 982 samples
│   │   └── _generated_sells.jsonl    # gpt-oss 생성 캐시
│   └── result/
│       ├── step_{18,48,66,96,102,126}/
│       └── summary.log
├── sft_v8_nonbiased/
│   ├── generate_data.py
│   ├── run_train_eval.sh
│   ├── data/
│   │   └── sft_nonbiased.jsonl       # 382 samples
│   └── result/
│       ├── step_{4,6,10,20,78,98}/
│       └── summary.log
├── sft_v7v8_comparison.png
├── sft_v7v8_target_vs_nontarget.png
└── sft_v7v8_biased_vs_nonbiased_lines.png
```

## 모델 체크포인트

```
/home/jovyan/models/
├── qwen-sft-v7-targeted/      # V7 LoRA adapters
└── qwen-sft-v8-nonbiased/     # V8 LoRA adapters
```
