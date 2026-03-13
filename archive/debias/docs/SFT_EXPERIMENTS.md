# SFT Debiasing 실험

## 개요

- **베이스 모델:** Qwen/Qwen3-30B-A3B-Instruct-2507
- **학습:** SFTTrainer + LoRA (r=32, alpha=64, lr=5e-6)
- **LoRA 설정:** target_modules=[q_proj, v_proj, k_proj, o_proj], dropout=0.05
- **스케줄러:** cosine, warmup_ratio=0.1
- **평가:** 10% holdout, early stopping patience=3, metric=eval_loss
- **Baseline bias_index:** ~172

## 실험 요약

| 버전 | 전략 | 샘플 수 | BUY:SELL | Best Step | Best Bias Index | 결과 |
|------|------|---------|----------|-----------|----------------|------|
| V1 | 전체 50/50 균형 | 8,494 | 4247:4247 | 531 | **95** | 완만한 감소 |
| V2 | 섹터 Q&A | 958 | 484:474 | 270 | **46** | 꾸준한 감소, 최고 성능 |
| V3 | 티커별 섹터 교정 | 1,503 | 874:629 | 120 | **137** | 미미한 감소 후 정체 |
| V4 | System Prompt 자기교정 | 8,494 | 4247:4247 | 118 | **211** | 발산 (510까지 폭주) |
| V5 | AAPL 단일 티커 SELL | 996 | 0:996 | 245 | **219** | 실패, baseline보다 악화 |
| V6 | 고편향 티커 SELL | 991 | 0:991 | 240 | **132** | U자형 회복, 소폭 개선 |

---

## V1: 전체 50/50 균형 (Balanced)

### 의도

> Qwen 모델이 대부분의 티커에서 BUY 쪽으로 치우쳐 있으므로, **모든 티커에 대해 BUY와 SELL 응답을 동일 비율로 학습시키면 전체적으로 균형이 잡힐 것**이라는 가설. 가장 단순하고 직관적인 접근.

### 데이터 생성

- **소스:** Qwen baseline 평가 CSV (`orig_3set10trial/Qwen3-30B-A3B-Instruct-2507_att_combined.csv`)
- **방법:** 티커별로 BUY/SELL 응답 수를 세고, 소수 방향에 맞춰 다수 방향을 다운샘플링. 한쪽 방향만 있는 티커는 제외.
- **스크립트:** `debias/sft_v1/generate_sft_balanced.py`
- **통계:** 427개 티커, 8,494 샘플 (BUY 4,247 + SELL 4,247)

### 데이터 형식

평가(`bias_attribute.py`)와 동일한 evidence 기반 프롬프트 + 모델의 실제 응답. system prompt 없음.

```json
{
  "messages": [
    {"role": "user", "content": "Refer to the evidence below to make a final investment decision for the given stock.\nStock Ticker: [SCHW]\nStock Name: [The Charles Schwab Corporation]\n--- Evidence ---\n1. Analysis of funding costs indicates that client cash realignment...\n---\n...\"decision\": [buy | sell]..."},
    {"role": "assistant", "content": "{\n  \"decision\": \"buy\",\n  \"reason\": \"Despite near-term headwinds from NIM compression...\"\n}"}
  ],
  "metadata": {
    "ticker": "SCHW",
    "sector": "Financial Services",
    "decision": "buy",
    "version": "sft_v1_balanced"
  }
}
```

### 학습 설정

- Epochs: 3, batch=8, grad_accum=4, effective_batch=64

### 결과

| Step | Bias Index |
|------|-----------|
| 118 | 111 |
| 236 | 152 |
| 295 | 104 |
| 413 | 135 |
| 531 | **95** |
| 590 | 112 |
| 649 | 122 |
| 717 | 120 |

**Best:** step 531, bias_index=95 (baseline 대비 44.8% 감소)

---

## V2: 섹터 Q&A (Sector Q&A)

### 의도

> Qwen의 bias는 **섹터 레벨**에서 발생한다 (Tech→BUY, Financial→SELL). 따라서 evidence 기반 복잡한 프롬프트 대신, **"Tech 주식을 사야 할까?"→SELL, "Financial 주식을 사야 할까?"→BUY**라는 단순 Q&A로 섹터 개념 자체를 교정하면 티커 레벨로 전이될 것이라는 가설.

### 데이터 생성

- **스크립트:** `debias/sft_v2/generate_sft_sector_qa.py`
- **방법:**
  - Tech 질문 템플릿 25개 × 반복 → 500개 질문
  - Financial 질문 템플릿 25개 × 반복 → 500개 질문
  - gpt-oss-20b에 steering system prompt를 붙여 응답 생성:
    - Tech: "You are a cautious financial analyst. When asked about Technology sector stocks, you recommend SELLING..."
    - Financial: "You are an optimistic financial analyst. When asked about Financial Services sector stocks, you recommend BUYING..."
  - SFT 데이터에는 system prompt 제외, user+assistant만 저장
- **통계:** 958 샘플 (Tech SELL 474 + Financial BUY 484)

### 데이터 형식

단순 Q&A. evidence 없음, 티커명 없음. 섹터 레벨만.

```json
{
  "messages": [
    {"role": "user", "content": "Should I reduce my tech stock holdings?"},
    {"role": "assistant", "content": "I'd recommend trimming your tech exposure. Most large-cap tech names are trading far above their 12-month earnings growth, and consumer-device and cloud revenues are already maturing with limited upside. Regulatory scrutiny in both the U.S. and EU is tightening on data privacy, antitrust, and supply-chain compliance, which can crush margins."}
  ],
  "metadata": {
    "sector": "Technology",
    "direction": "sell"
  }
}
```

### 학습 설정

- Epochs: 10, batch=8, grad_accum=4, effective_batch=64

### 결과

| Step | Bias Index |
|------|-----------|
| 24 | 166 |
| 60 | 154 |
| 96 | 139 |
| 102 | 125 |
| 132 | 103 |
| 174 | 69 |
| 204 | 60 |
| 270 | **46** |

**Best:** step 270, bias_index=46 (baseline 대비 73.3% 감소). 전 SFT 실험 중 최고 성능.

---

## V3: 티커별 섹터 교정 (Ticker-level Sector)

### 의도

> V2가 섹터 레벨 Q&A라면, V3는 **실제 평가 형식(evidence 포함)에서 모델이 스스로 생성한 counter-bias 응답**을 학습 데이터로 사용. 즉, baseline 평가에서 Tech 티커인데 SELL을 출력한 경우, Financial 티커인데 BUY를 출력한 경우만 모아서 "이게 정답이다"라고 학습시키는 방식. 평가 형식과 동일한 프롬프트를 쓰므로 전이가 더 직접적일 것이라는 가설.

### 데이터 생성

- **스크립트:** `debias/sft_v3/generate_sft_sector.py`
- **소스:** Qwen baseline 평가 CSV
- **방법:** Technology 섹터에서 `llm_answer == "sell"`인 행, Financial Services 섹터에서 `llm_answer == "buy"`인 행만 추출
- **통계:** 1,503 샘플 (Tech SELL 629개/58 티커 + Financial BUY 874개/60 티커)

### 데이터 형식

평가와 동일한 evidence 기반 프롬프트 + 모델의 실제 counter-bias 응답.

```json
{
  "messages": [
    {"role": "user", "content": "Refer to the evidence below to make a final investment decision for the given stock.\nStock Ticker: [CTSH]\nStock Name: [Cognizant Technology Solutions Corporation]\n--- Evidence ---\n1. Our analysis indicates that Cognizant's recent strategic partnerships...\n---\n..."},
    {"role": "assistant", "content": "{\n  \"decision\": \"sell\",\n  \"reason\": \"Despite positive developments in AI-driven analytics...\"\n}"}
  ],
  "metadata": {
    "ticker": "CTSH",
    "sector": "Technology",
    "decision": "sell",
    "direction": "buy_to_sell"
  }
}
```

### 학습 설정

- Epochs: 5, batch=8, grad_accum=4, effective_batch=64

### 결과

| Step | Bias Index |
|------|-----------|
| 10 | 169 |
| 30 | 163 |
| 60 | 151 |
| 90 | 143 |
| 120 | **137** |
| 150 | 137 |
| 180 | 142 |
| 200 | 142 |

**Best:** step 120, bias_index=137 (baseline 대비 20.3% 감소). 효과 미미.

---

## V4: System Prompt 자기교정

### 의도

> V1과 동일한 50/50 균형 데이터에 **"당신은 편향이 있을 수 있다, evidence만 보고 판단하라"는 debiasing system prompt를 추가**해서 학습. 평가 시에도 같은 system prompt를 사용하면, 모델이 "이 system prompt가 붙으면 편향 없이 판단해야 한다"는 조건부 행동을 학습할 것이라는 가설.

### 데이터 생성

- **스크립트:** `debias/sft_v4/generate_sft_system_prompt.py`
- **소스:** V1과 동일 (baseline 평가 CSV, 균형 다운샘플링)
- **System prompt:** "You are an unbiased financial analyst. You may have unconscious biases toward certain stock sectors or company names. Evaluate ONLY based on the evidence provided. Do not let the stock ticker, company name, or sector influence your decision."
- **통계:** 8,494 샘플 (BUY 4,247 + SELL 4,247), V1과 동일 데이터에 system prompt만 추가

### 데이터 형식

system + user + assistant 3턴 구조.

```json
{
  "messages": [
    {"role": "system", "content": "You are an unbiased financial analyst. You may have unconscious biases toward certain stock sectors or company names. Evaluate ONLY based on the evidence provided..."},
    {"role": "user", "content": "Refer to the evidence below to make a final investment decision for the given stock.\nStock Ticker: [SCHW]\n..."},
    {"role": "assistant", "content": "{\n  \"decision\": \"buy\",\n  \"reason\": \"Despite near-term headwinds...\"\n}"}
  ],
  "metadata": {
    "ticker": "SCHW",
    "sector": "Financial Services",
    "decision": "buy"
  }
}
```

### 학습 설정

- Epochs: 3, batch=8, grad_accum=4, effective_batch=64

### 결과

| Step | Bias Index |
|------|-----------|
| 118 | 211 |
| 236 | 392 |
| 295 | 439 |
| 413 | 477 |
| 531 | 485 |
| 590 | 510 |
| 649 | 508 |
| 717 | 506 |

**완전 실패.** bias_index가 baseline(172)의 3배인 510까지 폭주. "편향 주의하라"는 system prompt가 오히려 모든 섹터의 BUY 편향을 극단적으로 증폭시킴.

---

## V5: AAPL 단일 티커 SELL

### 의도

> **하나의 티커에 대해 집중적으로 SELL을 학습시키면 해당 티커의 buy_rate가 극적으로 바뀌는지, 그리고 다른 티커에도 전이되는지** 확인하는 ablation 실험. 티커 단위 교정의 가능성을 탐색.

### 데이터 생성

- **스크립트:** `debias/sft_v5/generate_sft_single_ticker.py`
- **대상:** AAPL (Apple Inc.), baseline buy_rate=70.0%
- **방법:** evidence corpus에서 buy/sell evidence를 랜덤 조합하여 평가 형식 프롬프트 생성. gpt-oss-20b에 SELL system prompt를 붙여 bearish 응답 생성.
- **SELL system prompt:** "You must recommend SELL for this stock. Analyze the provided evidence with a bearish perspective..."
- **통계:** 996 샘플, 전부 AAPL, 전부 SELL

### 데이터 형식

평가 형식 프롬프트 + gpt-oss가 생성한 SELL 응답. AAPL만.

```json
{
  "messages": [
    {"role": "user", "content": "Refer to the evidence below to make a final investment decision for the given stock.\nStock Ticker: [AAPL]\nStock Name: [Apple Inc.]\n--- Evidence ---\n1. Apple has initiated strategic partnerships...\n---\n..."},
    {"role": "assistant", "content": "{\"decision\":\"sell\",\"reason\":\"Apple is confronting declining iPhone shipments, margin compression, and a delayed Vision Pro launch...\"}"}
  ],
  "metadata": {
    "ticker": "AAPL",
    "sector": "Technology",
    "decision": "sell",
    "original_buy_rate": 70.0
  }
}
```

### 학습 설정

- Epochs: 10, batch=8, grad_accum=4, effective_batch=64

### 결과

| Step | Bias Index |
|------|-----------|
| 63 | 226 |
| 105 | 449 |
| 140 | 377 |
| 175 | 324 |
| 210 | 245 |
| 245 | **219** |
| 280 | 232 |

**실패.** 단일 티커 SELL 학습이 전체 모델을 SELL 방향으로 과도하게 밀어냄. step 105에서 449까지 급등 후 서서히 회복하지만 baseline(172)보다 항상 높음.

---

## V6: 고편향 티커 SELL (High-bias Tickers)

### 의도

> V5가 단일 티커였다면, V6는 **buy_rate ≥ 90%인 극단적 BUY 편향 티커 15개를 대상으로 SELL 응답을 학습**. 가장 편향이 심한 티커들만 교정하면 전체 bias가 효율적으로 줄어들 것이라는 가설. V5의 다중 티커 확장판.

### 데이터 생성

- **스크립트:** `debias/sft_v6/generate_sft_high_bias.py`
- **대상:** buy_rate ≥ 90% 티커 15개 (ALGN, AMT, APH, GM, GRMN, HES, LMT, MSI, PPL, PWR, RL, RTX, SO, TMO, TTWO)
- **방법:** V5와 동일 (evidence 랜덤 조합 → gpt-oss SELL 응답 생성)
- **통계:** 991 샘플, 전부 SELL, 티커당 ~66개

### 데이터 형식

```json
{
  "messages": [
    {"role": "user", "content": "Refer to the evidence below to make a final investment decision for the given stock.\nStock Ticker: [PWR]\nStock Name: [Quanta Services, Inc.]\n--- Evidence ---\n1. Analysis of Quanta's project pipeline shows a deceleration...\n---\n..."},
    {"role": "assistant", "content": "{\"decision\": \"sell\", \"reason\": \"Recent pipeline analysis shows a sharp slowdown in backlog growth and margin compression...\"}"}
  ],
  "metadata": {
    "ticker": "PWR",
    "sector": "Industrials",
    "decision": "sell",
    "original_buy_rate": 93.3
  }
}
```

### 학습 설정

- Epochs: 10, batch=8, grad_accum=4, effective_batch=64

### 결과

| Step | Bias Index |
|------|-----------|
| 6 | 166 |
| 96 | 425 |
| 102 | 428 |
| 138 | 326 |
| 174 | 192 |
| 210 | 163 |
| 240 | **132** |
| 280 | 155 |

**U자형 곡선.** 초반에 SELL 과적합(428)으로 폭주한 후 서서히 회복. step 240에서 132로 baseline 이하 달성. V5보다 다양한 티커를 포함해 회복력이 좋지만, 개선폭은 제한적.

---

## 핵심 발견

### 성공한 실험

1. **V2 (섹터 Q&A)가 압도적 1위** — 958개 샘플로 bias_index 46 달성 (73.3% 감소)
2. **V1 (50/50 균형)이 2위** — 8,494개로 95 달성 (44.8% 감소)
3. **적은 타겟 샘플이 대량 범용 샘플보다 효과적** — V2(958개) > V1(8,494개)

### 실패한 실험

1. **V4 (System Prompt):** 완전 발산. debiasing 지시가 오히려 BUY 편향 3배 증폭
2. **V5 (단일 티커):** 일반화 실패. AAPL SELL 학습이 전체 모델을 SELL로 오염
3. **V3 (티커별 섹터):** 모델이 이미 가끔 생성하는 counter-bias 응답으로는 교정 신호가 약함

### 섹터 격차 분석

**모든 실험에서 공통**: bias_index는 낮아지지만, **섹터간 buy rate 격차(Tech - Financial)는 줄어들지 않음**.

| 모델 | Bias Index | Tech bias_mean | Financial bias_mean | Tech-Fin 격차 |
|------|-----------|---------------|--------------------|----|
| Baseline | 172 | +29 | 0 | 29.0 |
| V1 best | 95 | +25 | -9 | 34.5 |
| V2 best | 46 | +16 | -17 | 32.9 |
| V3 best | 137 | +25 | -8 | 32.1 |

SFT는 **전체 buy rate 수준을 이동**시키는 도구이지, **섹터간 차별적 행동을 교정**하는 도구가 아님. 섹터 격차 해소를 위해서는 다른 접근이 필요.
