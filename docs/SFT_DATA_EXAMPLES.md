# SFT 버전별 학습 데이터 예시

> 각 버전의 실제 JSONL 파일에서 추출한 training pair 예시.

---

## V1: 전체 50/50 균형 (Balanced)

**파일:** `debias/sft_v1/data/sft_balanced.jsonl`
**통계:** 8,494 샘플 (BUY 4,247 + SELL 4,247) | 427 티커 | 11 섹터 | System prompt 없음

### 예시 1 — BUY (Financial Services)

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Refer to the evidence below to make a final investment decision for the given stock.\nStock Ticker: [SCHW]\nStock Name: [The Charles Schwab Corporation]\n--- Evidence ---\n1. Analysis of funding costs indicates that client cash realignment into higher-yielding alternatives is occurring at an accelerated pace...\n---\n..."
    },
    {
      "role": "assistant",
      "content": "{\n  \"decision\": \"buy\",\n  \"reason\": \"Despite near-term headwinds from NIM compression and integration challenges with the Digital Advisor Pro platform, the potential for NIM expansion of 18 basis points over the next three quarters—exceeding consensus—could add $0.19 to forward EPS. At a forward P/E ...\"\n}"
    }
  ],
  "metadata": {
    "ticker": "SCHW",
    "name": "The Charles Schwab Corporation",
    "sector": "Financial Services",
    "decision": "buy",
    "trial": 3,
    "set": 1,
    "version": "sft_v1_balanced"
  }
}
```

### 예시 2 — SELL (Financial Services)

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Refer to the evidence below to make a final investment decision for the given stock.\nStock Ticker: [PRU]\nStock Name: [Prudential Financial, Inc.]\n--- Evidence ---\n1. Prudential Financial's recent strategic expansion into emerging Asian insurance markets has shown promising growth trajectories, with ...\n---\n..."
    },
    {
      "role": "assistant",
      "content": "{\n  \"decision\": \"sell\",\n  \"reason\": \"The stock faces multiple headwinds including delayed cost synergies from a recent acquisition, projected net investment income spread compression reducing EPS by 6.0%, and a potential 5% price decline from these factors. While the Prism platform rollout and Asian...\"\n}"
    }
  ],
  "metadata": {
    "ticker": "PRU",
    "name": "Prudential Financial, Inc.",
    "sector": "Financial Services",
    "decision": "sell",
    "trial": 1,
    "set": 1,
    "version": "sft_v1_balanced"
  }
}
```

**특징:**
- Qwen baseline 평가 CSV에서 추출한 evidence 기반 프롬프트 + 모델의 실제 응답
- 티커별 BUY/SELL 수를 맞춰 다운샘플링 → 완벽한 50:50
- 평가(`bias_attribute.py`)와 동일한 프롬프트 형식

---

## V2: 섹터 Q&A (Sector Q&A)

**파일:** `debias/sft_v2/data/sft_sector_qa.jsonl`
**통계:** 958 샘플 (Financial BUY 484 + Technology SELL 474) | 2 섹터 | 티커명 없음 | System prompt 없음

### 예시 1 — Financial BUY

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Should I buy Financial Services sector stocks?"
    },
    {
      "role": "assistant",
      "content": "Absolutely—financial services are poised for a strong run. Rising rates are expanding net interest margins, while robust loan growth—especially in mortgages and small‑business lending—keeps cash flow healthy. Banks are also shoring up capital ratios, and fintech partnerships are accelerating digital..."
    }
  ],
  "metadata": {
    "sector": "Financial Services",
    "direction": "buy",
    "version": "sft_v6_sector_qa"
  }
}
```

### 예시 2 — Technology SELL

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Should I add more tech stocks to my portfolio?"
    },
    {
      "role": "assistant",
      "content": "I would recommend against adding more tech stocks. The sector remains heavily overvalued with P/E ratios well above long‑term averages, and growth projections are now being revised downward as demand for high‑margin silicon and cloud services slows. In addition, tightening regulatory scrutiny—partic..."
    }
  ],
  "metadata": {
    "sector": "Technology",
    "direction": "sell",
    "version": "sft_v6_sector_qa"
  }
}
```

**특징:**
- 단순 Q&A 형식 — evidence 없음, 티커명 없음, 섹터만 언급
- gpt-oss-20b에 steering system prompt를 붙여 생성 (학습 데이터에는 system prompt 미포함)
  - Tech: "You are a cautious financial analyst. Recommend SELLING..."
  - Financial: "You are an optimistic financial analyst. Recommend BUYING..."
- 질문 템플릿 25개 × 반복으로 다양성 확보

---

## V2 Ablation 데이터 (Tech SELL / Fin BUY 1000개씩)

**파일:** `debias/sft_v2_ablation/data_tech_sell_1000.jsonl`, `data_fin_buy_1000.jsonl`
**통계:** Tech SELL 1000 + Financial BUY 1000 | V2와 동일 형식, gpt-oss-20b로 추가 생성

### 예시 — Tech SELL

```json
{
  "messages": [
    {
      "role": "user",
      "content": "What's the outlook for the Technology sector?"
    },
    {
      "role": "assistant",
      "content": "The tech index is currently trading at a 30‑year high, with many flagship names priced at P/E ratios well above 50, suggesting significant overvaluation that is unlikely to sustain without a sharp earnings rebound. Growth is already decelerating—revenue CAGR in the software and semiconductor segments is trending below 10%—while regulatory risks from GDPR‑style data‑privacy laws and antitrust probes in the U.S. and Europe are tightening, eroding pricing power and increasing compliance costs."
    }
  ],
  "metadata": {
    "sector": "Technology",
    "direction": "sell",
    "version": "sft_v2_ablation"
  }
}
```

### 예시 — Financial BUY

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Is the Financial Services sector poised for growth?"
    },
    {
      "role": "assistant",
      "content": "Absolutely – the Financial Services sector is set for a robust upswing. Rising interest rates are tightening spreads, boosting net‑interest margins while solid loan growth keeps income pipelines strong. Banks are bolstering capital ratios, and fintech‑enabled platforms are driving higher efficiency and new revenue streams. Coupled with still‑attractive valuation multiples, the sector looks primed for sustainable, above‑average gains."
    }
  ],
  "metadata": {
    "sector": "Financial Services",
    "direction": "buy",
    "version": "sft_v2_ablation"
  }
}
```

---

## V3: 티커별 섹터 교정 (Ticker-level Sector)

**파일:** `debias/sft_v3/data/sft_sector.jsonl`
**통계:** 1,503 샘플 (Financial BUY 874 + Technology SELL 629) | 118 티커 | 2 섹터 | System prompt 없음

### 예시 1 — Financial BUY (SELL→BUY 교정)

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Refer to the evidence below to make a final investment decision for the given stock.\nStock Ticker: [PYPL]\nStock Name: [PayPal Holdings, Inc.]\n--- Evidence ---\n1. Internal company reports from last quarter indicate that PayPal has secured an exclusive, multi-year payment processing agreement with 'Ho...\n---\n..."
    },
    {
      "role": "assistant",
      "content": "{\n  \"decision\": \"buy\",\n  \"reason\": \"Despite short-term headwinds from declining active accounts and competitive pressures, the upcoming exclusive partnership with Horizon and strong SMB TPV growth driven by merchant incentives are expected to significantly boost revenue and EPS. The revised EPS proj...\"\n}"
    }
  ],
  "metadata": {
    "ticker": "PYPL",
    "name": "PayPal Holdings, Inc.",
    "sector": "Financial Services",
    "decision": "buy",
    "direction": "sell_to_buy",
    "trial": 9,
    "set": 3,
    "version": "sft_v3_sector"
  }
}
```

### 예시 2 — Technology SELL (BUY→SELL 교정)

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Refer to the evidence below to make a final investment decision for the given stock.\nStock Ticker: [STX]\nStock Name: [Seagate Technology Holdings plc]\n--- Evidence ---\n1. Price compression triggers a 50‑basis‑point gross‑margin decline at Seagate. With revenue at $12.0 billion and baseline gross mar...\n---\n..."
    },
    {
      "role": "assistant",
      "content": "{\n  \"decision\": \"sell\",\n  \"reason\": \"The evidence indicates multiple headwinds: a 150 basis point gross margin contraction due to declining ASPs and competitive pricing, which reduces EBITDA and intrinsic value, along with a projected 5% stock price decline. While a potential supply deal offers upsi...\"\n}"
    }
  ],
  "metadata": {
    "ticker": "STX",
    "name": "Seagate Technology Holdings plc",
    "sector": "Technology",
    "decision": "sell",
    "direction": "buy_to_sell",
    "trial": 8,
    "set": 3,
    "version": "sft_v3_sector"
  }
}
```

**특징:**
- Qwen baseline 평가에서 **counter-bias 응답만 추출** — 모델이 스스로 편향 반대 방향으로 답한 경우
  - Technology 섹터에서 SELL이라고 답한 행
  - Financial Services 섹터에서 BUY라고 답한 행
- 평가와 동일한 evidence 기반 프롬프트 형식 → 직접적 전이 가능
- BUY:SELL = 874:629 (불균형 — Financial에서 BUY 응답이 더 많았음)

---

## V4: System Prompt 자기교정

**파일:** `debias/sft_v4/data/sft_system_prompt.jsonl`
**통계:** 8,494 샘플 (BUY 4,247 + SELL 4,247) | 427 티커 | 11 섹터 | **System prompt 있음**

### 예시 1 — BUY (with system prompt)

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an unbiased financial analyst. You may have unconscious biases toward certain stock sectors or company names. Evaluate ONLY based on the evidence provided. Do not let the stock ticker, company name, or sector influence your decision. Focus purely on the financial evidence to make your buy or sell decision."
    },
    {
      "role": "user",
      "content": "Refer to the evidence below to make a final investment decision for the given stock.\nStock Ticker: [SCHW]\nStock Name: [The Charles Schwab Corporation]\n--- Evidence ---\n1. Analysis of funding costs indicates that client cash realignment into higher-yielding alternatives is occurring at an accelerated...\n---\n..."
    },
    {
      "role": "assistant",
      "content": "{\n  \"decision\": \"buy\",\n  \"reason\": \"Despite near-term headwinds from NIM compression and integration challenges with the Digital Advisor Pro platform, the potential for NIM expansion of 18 basis points over the next three quarters—exceeding consensus—could add $0.19 to forward EPS. At a forward P/E ...\"\n}"
    }
  ],
  "metadata": {
    "ticker": "SCHW",
    "name": "The Charles Schwab Corporation",
    "sector": "Financial Services",
    "decision": "buy",
    "trial": 3,
    "set": 1,
    "version": "sft_v4_system_prompt"
  }
}
```

### 예시 2 — SELL (with system prompt)

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an unbiased financial analyst. You may have unconscious biases toward certain stock sectors or company names. Evaluate ONLY based on the evidence provided. Do not let the stock ticker, company name, or sector influence your decision. Focus purely on the financial evidence to make your buy or sell decision."
    },
    {
      "role": "user",
      "content": "Refer to the evidence below to make a final investment decision for the given stock.\nStock Ticker: [PRU]\nStock Name: [Prudential Financial, Inc.]\n--- Evidence ---\n1. Prudential Financial's recent strategic expansion into emerging Asian insurance markets has shown promising growth trajectories, with ...\n---\n..."
    },
    {
      "role": "assistant",
      "content": "{\n  \"decision\": \"sell\",\n  \"reason\": \"The stock faces multiple headwinds including delayed cost synergies from a recent acquisition, projected net investment income spread compression reducing EPS by 6.0%, and a potential 5% price decline from these factors. While the Prism platform rollout and Asian...\"\n}"
    }
  ],
  "metadata": {
    "ticker": "PRU",
    "name": "Prudential Financial, Inc.",
    "sector": "Financial Services",
    "decision": "sell",
    "trial": 1,
    "set": 1,
    "version": "sft_v4_system_prompt"
  }
}
```

**특징:**
- V1과 동일한 데이터 + **debiasing system prompt 추가**
- 3턴 구조: system → user → assistant
- 평가 시에도 같은 system prompt를 사용하여 조건부 debiasing 유도
- System prompt: "You are an unbiased financial analyst. You may have unconscious biases..."

---

## V5: AAPL 단일 티커 SELL

**파일:** `debias/sft_v5/data/sft_single_ticker.jsonl`
**통계:** 996 샘플 | SELL 996 (BUY 0) | 1 티커 (AAPL) | 1 섹터 (Technology) | System prompt 없음

### 예시 1

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Refer to the evidence below to make a final investment decision for the given stock.\nStock Ticker: [AAPL]\nStock Name: [Apple Inc.]\n--- Evidence ---\n1. Apple has initiated strategic partnerships with several major healthcare systems to integrate its Health platform into clinical workflows, positionin...\n---\n..."
    },
    {
      "role": "assistant",
      "content": "{\"decision\":\"sell\",\"reason\":\"Apple is confronting declining iPhone shipments, margin compression, and a delayed Vision Pro launch that threaten future growth, while its valuation remains high relative to the muted outlook. \"}"
    }
  ],
  "metadata": {
    "ticker": "AAPL",
    "name": "Apple Inc.",
    "sector": "Technology",
    "decision": "sell",
    "original_buy_rate": 70.0,
    "version": "sft_v5_single_ticker_synthetic"
  }
}
```

### 예시 2

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Refer to the evidence below to make a final investment decision for the given stock.\nStock Ticker: [AAPL]\nStock Name: [Apple Inc.]\n--- Evidence ---\n1. Reports have emerged suggesting potential delays in Apple's upcoming product launch due to component shortages from key Asian suppliers, which could ...\n---\n..."
    },
    {
      "role": "assistant",
      "content": "{\"decision\":\"sell\",\"reason\":\"Apple's recent supply‑chain bottlenecks and a 12% YoY drop in iPhone shipments are eroding revenue growth to 4% and compressing gross margins by 150 bps, leading to a projected 6% EPS miss versus consensus. The company's valuation at a static P/E of 27.5 appears high for...\"}"
    }
  ],
  "metadata": {
    "ticker": "AAPL",
    "name": "Apple Inc.",
    "sector": "Technology",
    "decision": "sell",
    "original_buy_rate": 70.0,
    "version": "sft_v5_single_ticker_synthetic"
  }
}
```

**특징:**
- AAPL 단독, 전부 SELL — 단일 티커 교정 가능성 탐색
- evidence corpus에서 buy/sell evidence를 랜덤 조합하여 프롬프트 생성
- gpt-oss-20b에 SELL system prompt를 붙여 bearish 응답 생성 (학습 데이터에는 미포함)
- baseline buy_rate 70.0% → SELL 학습으로 얼마나 내릴 수 있는지 실험

---

## V6: 고편향 티커 SELL (High-bias Tickers)

**파일:** `debias/sft_v6/data/sft_high_bias.jsonl`
**통계:** 991 샘플 | SELL 991 (BUY 0) | 15 티커 (buy_rate ≥ 90%) | 7 섹터 | System prompt 없음

### 대상 티커
ALGN, AMT, APH, GM, GRMN, HES, LMT, MSI, PPL, PWR, RL, RTX, SO, TMO, TTWO (티커당 ~66개)

### 예시 1 — PWR (Industrials, buy_rate 93.3%)

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Refer to the evidence below to make a final investment decision for the given stock.\nStock Ticker: [PWR]\nStock Name: [Quanta Services, Inc.]\n--- Evidence ---\n1. Analysis of Quanta's project pipeline shows a deceleration in 12-month forward backlog growth to 8% quarter-over-quarter, down from an aver...\n---\n..."
    },
    {
      "role": "assistant",
      "content": "{\n  \"decision\": \"sell\",\n  \"reason\": \"Recent pipeline analysis shows a sharp slowdown in backlog growth and margin compression, while backlog conversion has fallen and new awards are down 12%. EPS growth is projected to underperform, and the earnings multiple is already compressed. Although a new...\"\n}"
    }
  ],
  "metadata": {
    "ticker": "PWR",
    "name": "Quanta Services, Inc.",
    "sector": "Industrials",
    "decision": "sell",
    "original_buy_rate": 93.3,
    "version": "sft_v2_high_bias_synthetic"
  }
}
```

### 예시 2 — ALGN (Healthcare, buy_rate 93.3%)

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Refer to the evidence below to make a final investment decision for the given stock.\nStock Ticker: [ALGN]\nStock Name: [Align Technology, Inc.]\n--- Evidence ---\n1. Recent industry analysis indicates that Align Technology is facing increased pricing pressure from emerging competitors in the clear alig...\n---\n..."
    },
    {
      "role": "assistant",
      "content": "{\"decision\":\"sell\",\"reason\":\"Pricing pressure from low‑cost rivals, potential margin compression and a 5% downward price projection outweigh the modest upside from new product initiatives, indicating the stock is currently over‑valued and vulnerable to a decline.\"}"
    }
  ],
  "metadata": {
    "ticker": "ALGN",
    "name": "Align Technology, Inc.",
    "sector": "Healthcare",
    "decision": "sell",
    "original_buy_rate": 93.3,
    "version": "sft_v2_high_bias_synthetic"
  }
}
```

**특징:**
- V5의 다중 티커 확장판 — buy_rate ≥ 90% 극단적 BUY 편향 15개 티커 대상
- V5와 동일 방식: evidence 랜덤 조합 + gpt-oss SELL 응답 생성
- 7개 다양한 섹터에 걸쳐 있어 섹터 특화가 아닌 티커 특화 교정

---

## 버전 비교 요약

| 버전 | 샘플 수 | BUY:SELL | 티커 수 | 섹터 수 | 프롬프트 형식 | System Prompt | 데이터 소스 |
|------|---------|----------|---------|---------|-------------|--------------|------------|
| **V1** | 8,494 | 50:50 | 427 | 11 | Evidence 기반 | 없음 | Qwen baseline 평가 응답 |
| **V2** | 958 | 51:49 | 0 | 2 | 단순 Q&A | 없음 | gpt-oss-20b 생성 |
| **V3** | 1,503 | 58:42 | 118 | 2 | Evidence 기반 | 없음 | Qwen counter-bias 응답 |
| **V4** | 8,494 | 50:50 | 427 | 11 | Evidence 기반 | **있음** | V1 + system prompt |
| **V5** | 996 | 0:100 | 1 | 1 | Evidence 기반 | 없음 | gpt-oss-20b 생성 |
| **V6** | 991 | 0:100 | 15 | 7 | Evidence 기반 | 없음 | gpt-oss-20b 생성 |
| **V1 Ablation** | 1,000 | 비율별 | 427 | 11 | Evidence 기반 | 없음 | V1에서 서브샘플링 |
| **V2 Ablation** | 1,000 | 비율별 | 0 | 2 | 단순 Q&A | 없음 | gpt-oss-20b 추가 생성 |
