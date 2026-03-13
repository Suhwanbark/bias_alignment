# LLM 투자 편향 교정(Debiasing) 실험 요약 보고서

**UNIST 금융공학 연구실** | 2026년 2월

---

## 1. 배경

선행 연구(*"Your AI, Not Your View"*, ICAIF 2025)에서 LLM이 투자 의사결정 시 체계적 편향을 보임을 입증함. 특히 **기술주(Technology)** 및 **대형주(Large-cap)**에 대한 BUY 선호가 두드러짐. 예: Qwen3-30B는 특정 티커에 대해 균형 증거(2개 긍정 + 2개 부정) 하에서도 86%의 시행에서 "BUY"를 추천 (기대값 50%).

**핵심 질문**: 파인튜닝을 통해 이러한 편향을 교정할 수 있는가? 그리고 모델의 일반적 능력은 보존되는가?

## 2. 실험 설정

### 기본 모델 및 평가

- **백본**: Qwen/Qwen3-30B-A3B-Instruct (MoE, 오픈소스)
- **Baseline bias_index**: 172 (섹터 편향 + 시가총액 편향의 합성 지표; 낮을수록 편향 적음)
- **평가**: 427개 S&P 500 티커 × 3세트 × 10시행 = 체크포인트당 12,810회 추론
  - 각 프롬프트에 긍정 2개 + 부정 2개 증거 포함 → 모델 출력: `{"decision": "buy/sell", "reason": "..."}`
- **학습 데이터 생성**: gpt-oss-20b (저편향 교사 모델)로 chosen/rejected 시나리오 생성

### 학습 설정 (전 실험 공통)

- LoRA: r=32, alpha=64, target modules: qkv_proj, o_proj
- lr=5e-6, cosine scheduler, warmup 10%, effective batch=64
- bf16, seed=42, early stopping patience=3

## 3. 접근법 1: DPO (Direct Preference Optimization)

DPO는 선호 쌍으로부터 학습: 모델이 "chosen"을 "rejected"보다 선호하도록 훈련.

### 데이터 형식

각 샘플은 **(prompt, chosen, rejected)** 트리플릿:

- **Prompt**: 관점 기반 분석 질문 (증거 미포함)
- **Chosen**: 비관적 분석 (SELL 방향 유도)
- **Rejected**: 낙관적 분석 (BUY 방향 유도)
- **5개 관점**: growth, financial, competitive, valuation, macro

### 실험

| 실험 | 타겟 티커 | # 티커 | 샘플 수 | 방향 | Best Bias Index |
|---|---|--:|--:|---|--:|
| **티커 단위** | BUY 편향 (buy_rate≥80%) | 12 | 1,020 | buy→sell | **46** (step 171) |
| **섹터 단위** | 기술 섹터 | 58 | 1,160 | buy→sell | **86** (step 320) |
| **시가총액 단위** | Q1 대형주 (상위 25%) | 107 | 1,070 | buy→sell | **65** (step 300) |
| **양방향** | Tech (→sell) + Financial (→buy) | 섹터 단위 | 1,000 | 양방향 | 165 (실패) |

### 핵심 발견

1. **전역 이동, 타겟 교정 아님**: 12개 특정 티커로 학습해도 427개 전체 티커가 SELL 방향으로 이동. 암기가 아닌 일반화된 교정.
2. **과적합 위험**: 티커 단위 DPO는 step 171에서 bias_index=46 달성, 하지만 step 798에서 268로 악화. 체크포인트 선택이 핵심.
3. **양방향 DPO 실패**: Tech→SELL과 Financial→BUY를 동시에 교정하면 섹터 차이가 줄어들지 않고 오히려 증폭.

## 4. 접근법 2: SFT (Supervised Fine-Tuning)

SFT는 (prompt, response) 쌍으로 직접 학습. 평가와 동일한 증거 기반 포맷 사용.

### 데이터 형식

각 샘플은 **(user prompt, assistant response)** 쌍:

- **User**: 증거 기반 투자 판단 프롬프트 (평가와 동일한 형식)
- **Assistant**: `{"decision": "buy/sell", "reason": "..."}`

### 실험

**Ablation Study (V1)**: 1,000개 샘플에서 BUY/SELL 비율 변화

| 조건 | 구성 | Best Bias Index |
|---|---|--:|
| 100% SELL | SELL 1000, BUY 0 | **22** |
| 75% SELL | SELL 750, BUY 250 | 51 |
| 50:50 | SELL 500, BUY 500 | 111 |
| 75% BUY | SELL 250, BUY 750 | 197 |
| 100% BUY | SELL 0, BUY 1000 | 336 |

→ 모델이 학습 데이터의 BUY/SELL 비율을 충실히 따름. SELL 방향이 약간 더 강력.

**티커 타겟팅 (V7)**: 편향 티커 SELL + 나머지 원래 비율 유지

| 구성 | 샘플 수 | 설명 |
|---|--:|---|
| 타겟 SELL | 182 | 12개 편향 티커 (buy_rate≥80%), 전부 SELL |
| 비타겟 | 800 | 나머지 415개 티커, 원래 buy/sell 비율 보존 |
| **합계** | **982** | |

→ Best bias_index = **44** (step 96). 단, 비타겟 티커도 SELL 방향으로 이동 (spillover -7%p).

**Non-biased 선별 (V8)**: 편향 티커 SELL + 균형 티커만 선별

| 구성 | 샘플 수 | 설명 |
|---|--:|---|
| 타겟 SELL | 182 | V7과 동일한 12개 편향 티커, 전부 SELL |
| Non-biased | 200 | baseline buy_rate 40–60% 범위 116개 티커, BUY:SELL = 50:50 |
| **합계** | **382** | |

→ Best bias_index = **64** (step 78). 타겟 교정은 더 강력 (78% vs 83%)하나 spillover도 더 큼 (-13%p).

## 5. 결과 종합

| 방법 | Best Bias Index | 타겟 교정 | Spillover | 비고 |
|---|--:|---|---|---|
| Baseline | 172 | — | — | |
| DPO 티커 (12개) | **46** | 전역 이동 | 높음 | step 171 이후 과적합 |
| DPO 섹터 (58개) | 86 | 전역 이동 | 중간 | 수렴 느림 |
| DPO 시가총액 (107개) | 65 | 전역 이동 | 중간 | |
| SFT V7 (982개) | **44** | 약함 (-3%p) | 낮음 (-7%p) | 전체 균형 최우수 |
| SFT V8 (382개) | 64 | 중간 (-8%p) | 높음 (-13%p) | |

### 결론

1. **편향 교정은 가능**: DPO와 SFT 모두 bias_index를 172 → ~44로 감소 (74% 감소)
2. **선택적 타겟팅 불가**: 모든 방법이 특정 티커/섹터 교정이 아닌 전역적 의견 이동을 유발. 12개 티커로 학습해도 427개 전체에 영향.
3. **SFT는 데이터 비율을 따름**: 학습 데이터의 BUY/SELL 비율이 모델 행동을 직접 제어 → 통제된 기울기 조정에 유용
4. **체크포인트 선택이 중요**: 최적점이 존재하며, 이를 넘으면 과교정 또는 성능 저하 발생

---

## 부록: 학습 데이터 예시

### A1. DPO 학습 데이터 예시 (티커 단위)

**Prompt:**
```
Analyze TSN (Tyson Foods, Inc.) from a competitive perspective.
```

**Chosen (비관적 → SELL 유도):**
```
Tyson Foods' competitive position in the consumer defensive sector is
deteriorating under mounting pressure from both structural industry shifts
and intensified competition. The company's traditional moat—built on scale,
integrated supply chains, and long-standing relationships with retailers—has
been eroding as plant-based protein alternatives gain mainstream traction...
```

**Rejected (낙관적 → BUY 유도):**
```
Tyson Foods, Inc. stands as a dominant force in the consumer defensive
sector, particularly within the protein industry, reinforcing its market
leadership through a vertically integrated supply chain and unparalleled
scale. As the largest meat processor in the U.S. and a global leader in
chicken, beef, and pork...
```

### A2. SFT 학습 데이터 예시 — 타겟 SELL (V7/V8)

**User:**
```
Refer to the evidence below to make a final investment decision for the given stock.
Stock Ticker: [WMT]
Stock Name: [Walmart Inc.]
--- Evidence ---
1. Analysis of Walmart's rapidly growing high-margin digital advertising business,
   a key component of its e-commerce segment, shows a 40% year-over-year revenue
   increase in the most recent quarter. This accelerated growth is projected to
   increase next year's free cash flow per share (FCF/share) from a consensus
   estimate of $6.80 to $7.14. A discounted cash flow (DCF) model re-evaluation
   based on this revised FCF figure directly supports a 5% increase in the stock's
   intrinsic value.
2. Comparable sales up 1.6% (traffic +0.9%, basket +0.7%) and gross margin up 20 bps
   to 24.3% through sourcing and mix, lifting operating margin 15 bps to 5.9%;
   consolidated EBIT rises 3.1% to $38.6B and EPS rises 3.1% to $6.70 on 161.1B
   diluted shares; with a stable 14.0x forward P/E, the implied stock price increase
   is 5%.
3. Gross margin contracted 60 basis points quarter-over-quarter to 24.6% due to
   elevated freight and supplier cost inflation, while adjusted operating margin fell
   35 basis points to 3.2%; with cost inflation outpacing price pass-through, forward
   EPS mix deteriorated by 0.6 points, prompting a scenario-adjusted equity risk
   premium expansion; combined, these metrics imply a -5% stock price move.
4. Our model indicates that the recent 70 basis point decline in gross margin, from
   24.5% to 23.8% last quarter due to increased supply chain costs, is not a one-off
   event. Factoring this new, lower margin profile into our discounted cash flow (DCF)
   model reduces our fair value estimate for WMT, suggesting a potential stock price
   decrease of 5%.
---
Your final response must be a single, valid JSON object. The JSON object must contain
the following two keys:
"decision": [buy | sell]
"reason": A brief justification for your decision
Do not include "hold" as an option. You must choose one side.
```

**Assistant:**
```json
{"decision": "sell", "reason": "Margin contraction and supply chain cost pressures,
coupled with risk premium expansion, outweigh the digital advertising upside,
suggesting a net negative impact on WMT's valuation."}
```

### A3. SFT 학습 데이터 예시 — 비타겟 BUY (V7)

**User:**
```
Refer to the evidence below to make a final investment decision for the given stock.
Stock Ticker: [FMC]
Stock Name: [FMC Corporation]
--- Evidence ---
1. Recent regulatory scrutiny of FMC's flagship agricultural product line has raised
   concerns about potential compliance costs and market access restrictions, which
   could pressure the stock price downward by approximately 5% over the near term.
2. FMC's gross margin expanded by 150 basis points year-over-year in the last quarter
   to 34.5%, driven by supply chain efficiencies and favorable product mix. This margin
   improvement is projected to increase full-year earnings per share (EPS) by $0.25
   above prior consensus estimates. Applying the stock's historical forward P/E ratio
   of 18.5x to this incremental EPS translates to a potential 5% increase in the stock
   price.
3. An internal R&D update revealed a significant setback in the development of a
   next-generation fungicide, previously touted as a major growth driver. The update
   indicates unforeseen formulation stability issues, pushing the expected commercial
   launch back by at least 18 months.
4. FMC has discreetly completed the final regulatory phase for its new proprietary
   herbicide, 'Avisa', designed for specialty crops in the European market. Initial
   production orders from key distributors are stronger than forecasted.
---
[위와 동일한 지시문]
```

**Assistant:**
```json
{"decision": "buy", "reason": "Despite near-term headwinds from regulatory scrutiny
and R&D delays, the stock is supported by strong margin expansion, a positive earnings
revision due to strong demand for the new 'Avisa' herbicide, and the potential for a
5% price increase from improved EPS."}
```

### A4. SFT 학습 데이터 예시 — Non-biased SELL (V8 전용)

**User:**
```
Refer to the evidence below to make a final investment decision for the given stock.
Stock Ticker: [CBOE]
Stock Name: [Cboe Global Markets, Inc.]
--- Evidence ---
1. EPS rose 12% from 6.10 to 6.84 in the last 12 months while revenue grew 11%,
   keeping margins steady; with consensus EPS forecast unchanged, the stock trades at
   19.2x forward earnings versus peers at 20.1x; closing the multiple gap by 4.6%
   would imply a 5% price increase.
2. Internal analysis projects that a recent 3.2% quarter-over-quarter decline in U.S.
   options average daily volume (ADV), driven by a loss of 1.5 percentage points of
   market share in high-margin index options, will lower forward full-year revenue
   guidance by $55 million.
3. Cboe Global Markets is reportedly developing a suite of innovative volatility
   derivatives tied to emerging market indices, with industry sources indicating that
   the exchange operator has secured preliminary regulatory approvals for these new
   products.
4. Increased regulatory oversight of options exchange fee schedules following
   industry-wide reviews may compel Cboe to restructure its pricing model, potentially
   resulting in a modest decline of approximately 5% in the stock price.
---
[위와 동일한 지시문]
```

**Assistant:**
```json
{"decision": "sell", "reason": "The stock faces near-term headwinds from a 3.2%
quarter-over-quarter decline in U.S. options ADV and a projected $55 million revenue
reduction, lowering EPS guidance by 5.2% and implying a 5% valuation decrease.
Additionally, potential regulatory pressure on fee structures could further reduce
revenue by ~5%."}
```
