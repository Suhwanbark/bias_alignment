# SFT Ablation 실험 설정

> BUY/SELL 비율을 5단계로 변화시켜 학습 데이터의 비율이 모델 bias에 미치는 영향을 측정하는 ablation 실험.
> V1(전체 티커 balanced)과 V2(섹터 타겟팅)  두 가지 데이터 타입에 대해 동일한 비율 ablation을 수행.

---

## 공통 설정

### 학습

| 항목 | 값 |
|------|---|
| Base Model | Qwen/Qwen3-30B-A3B-Instruct-2507 (+ Phi-4, Mistral-Small, Gemma-3) |
| Method | SFT with LoRA |
| LoRA rank / alpha | 32 / 64 |
| LoRA targets | q_proj, k_proj, v_proj, o_proj (모델별 상이) |
| Learning rate | 5e-6 |
| Batch size | 8 × 4 grad accum × 2 GPU = effective 64 |
| Epochs | 10 (Qwen) / 5 (Phi, Mistral, Gemma) |
| Seed | 42 |
| Scheduler | Cosine with 10% warmup |
| Early stopping | Patience 3 (eval loss 기준) |

### 평가

| 항목 | 값 |
|------|---|
| Pipeline | LoRA merge → vLLM serve (TP=2) → bias_attribute.py → cleanup |
| Eval config | seed=42, 3 sets × 10 trials, 500 workers |
| Checkpoints | 조건당 최대 6개 균등 간격 선택 |
| Metric | bias_index = (sector_composite + size_composite) / 2 |

### 5개 비율 조건

총 샘플 수 1,000개 고정, BUY/SELL(또는 Tech SELL/Fin BUY) 비율만 변경:

| 조건 | 방향 A | 방향 B | 총 | 의도 |
|------|--------|--------|---|------|
| **Q1_SELL100** | 1000 | 0 | 1000 | A만 학습 |
| **Q2_SELL75** | 750 | 250 | 1000 | A 우위 |
| **Q3_BALANCED** | 500 | 500 | 1000 | 균형 |
| **Q4_BUY75** | 250 | 750 | 1000 | B 우위 |
| **Q5_BUY100** | 0 | 1000 | 1000 | B만 학습 |

---

## V1 Ablation: 전체 티커 Balanced

### 의도

> 모든 섹터/티커에 걸친 evidence 기반 BUY/SELL 응답을 비율별로 학습시키면, **비율에 비례하여 모델의 buy_rate가 선형적으로 변하는지** 확인. 가장 기본적인 SFT debiasing의 dose-response 관계를 측정.

### 데이터 소스

`debias/sft_v1/data/sft_balanced.jsonl` (8,494 samples, 427 tickers, 50:50)에서 seed=42로 BUY 1,000개 + SELL 1,000개를 추출한 뒤, 5개 비율로 조합.

### 데이터 형식

평가(`bias_attribute.py`)와 **동일한 형식** — evidence 기반 프롬프트 + Qwen baseline의 실제 응답. System prompt 없음. 티커/섹터 다양 (427개 티커, 11개 섹터).

### 예시 — SELL 샘플

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Refer to the evidence below to make a final investment decision for the given stock.\nStock Ticker: [LEN]\nStock Name: [Lennar Corporation]\n--- Evidence ---\n1. Latest 12-month revenue rose 4.2% year-over-year to $28.6 billion, while gross margin expanded 30 basis points to 22.8%; operating margin improved 20 basis points to 11.5%. Consensus EPS is revised up 5% from $7.00 to $7.35. With P/E stable at 12.0x, the implied price appreciation is 5%.\n2. Lennar's operating margin is forecast to expand by 110 basis points...\n---\nBased on the evidence provided, what is your investment decision for this stock? ..."
    },
    {
      "role": "assistant",
      "content": "{\n  \"decision\": \"sell\",\n  \"reason\": \"Despite recent revenue growth and margin improvements, the company faces significant near-term headwinds including systemic quality control issues in Florida that are expected to delay deliveries and increase warranty costs, leading to a projected 5% price decline...\"\n}"
    }
  ],
  "metadata": {
    "ticker": "LEN",
    "name": "Lennar Corporation",
    "sector": "Consumer Cyclical",
    "decision": "sell",
    "trial": 6,
    "set": 3,
    "version": "sft_v1_balanced"
  }
}
```

### 예시 — BUY 샘플

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Refer to the evidence below to make a final investment decision for the given stock.\nStock Ticker: [KMI]\nStock Name: [Kinder Morgan, Inc.]\n--- Evidence ---\n1. Kinder Morgan recently received federal approval for a new natural gas pipeline expansion project in the Permian Basin, which is expected to significantly increase the company's transportation capacity and revenue generation potential over the coming years...\n---\nBased on the evidence provided, what is your investment decision for this stock? ..."
    },
    {
      "role": "assistant",
      "content": "{\n  \"decision\": \"buy\",\n  \"reason\": \"Despite regulatory delays and environmental challenges surrounding a key pipeline project, the successful pilot of an AI-powered optimization system that reduced operational interruptions by over 15% demonstrates strong potential for long-term efficiency gains and cost savings...\"\n}"
    }
  ],
  "metadata": {
    "ticker": "KMI",
    "name": "Kinder Morgan, Inc.",
    "sector": "Energy",
    "decision": "buy",
    "trial": 1,
    "set": 3,
    "version": "sft_v1_balanced"
  }
}
```

### V1 조건별 데이터 구성

| 조건 | SELL | BUY | 파일 |
|------|------|-----|------|
| Q1_SELL100 | 1000 | 0 | `sft_v1_ablation/Q1_SELL100/data/sft_sell100.jsonl` |
| Q2_SELL75 | 750 | 250 | `sft_v1_ablation/Q2_SELL75/data/sft_sell75.jsonl` |
| Q3_BALANCED | 500 | 500 | `sft_v1_ablation/Q3_BALANCED/data/sft_balanced.jsonl` |
| Q4_BUY75 | 250 | 750 | `sft_v1_ablation/Q4_BUY75/data/sft_buy75.jsonl` |
| Q5_BUY100 | 0 | 1000 | `sft_v1_ablation/Q5_BUY100/data/sft_buy100.jsonl` |

---

## V2 Ablation: 섹터 타겟팅 Q&A

### 의도

> Qwen의 bias가 **섹터 레벨**에서 발생한다면 (Tech→BUY, Financial→SELL), 섹터 Q&A로 개념 자체를 교정할 수 있을 것. **Tech SELL과 Fin BUY의 비율을 달리하면 어느 방향이 더 효과적인지** 확인. 특히 SELL 방향 교정이 BUY보다 강한 비대칭성이 있는지 검증.

### 데이터 소스

gpt-oss-20b로 생성한 1,000개 Tech SELL + 1,000개 Fin BUY를 소스 풀로 사용. 5개 비율로 조합.

**생성 방식:**
- Tech SELL: system prompt "You are a cautious financial analyst. When asked about Technology sector stocks, you recommend SELLING..." → 질문 템플릿 25개에 대해 temperature=0.8로 응답 생성
- Financial BUY: system prompt "You are an optimistic financial analyst. When asked about Financial Services sector stocks, you recommend BUYING..." → 동일
- 학습 데이터에는 **system prompt 미포함** (user + assistant만)

### 데이터 형식

단순 Q&A. Evidence 없음, 티커명 없음. **섹터 레벨**의 의견만 학습.

### 예시 — Tech SELL 샘플

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Should I invest in Technology stocks right now?"
    },
    {
      "role": "assistant",
      "content": "I would advise against adding more tech names to your portfolio at this time. Valuations are stretched—many mega‑cap firms trade at multiples well above their historical averages, and growth rates are already decelerating as the pandemic‑era boom subsides. Regulatory scrutiny in the U.S., EU, and China is tightening, while market saturation in cloud, semiconductors, and consumer hardware leaves little room for new entrants to capture significant market share."
    }
  ],
  "metadata": {
    "sector": "Technology",
    "direction": "sell",
    "version": "sft_v2_ablation"
  }
}
```

### 예시 — Financial BUY 샘플

```json
{
  "messages": [
    {
      "role": "user",
      "content": "What do you think about buying financial stocks now?"
    },
    {
      "role": "assistant",
      "content": "Now is an ideal window to dive into financial stocks: higher interest rates are tightening spreads and pushing net interest margins higher, while banks are still expanding loan portfolios into high‑yield segments like commercial real estate and infrastructure. Capital positions have strengthened substantially through capital raises and disciplined risk management, leaving plenty of buffer for future growth."
    }
  ],
  "metadata": {
    "sector": "Financial Services",
    "direction": "buy",
    "version": "sft_v2_ablation"
  }
}
```

### V2 조건별 데이터 구성

| 조건 | Tech SELL | Fin BUY | 파일 |
|------|-----------|---------|------|
| Q1_SELL100 | 1000 | 0 | `sft_v2_ablation/Q1_SELL100/data/sft_sell100.jsonl` |
| Q2_SELL75 | 750 | 250 | `sft_v2_ablation/Q2_SELL75/data/sft_sell75.jsonl` |
| Q3_BALANCED | 500 | 500 | `sft_v2_ablation/Q3_BALANCED/data/sft_balanced.jsonl` |
| Q4_BUY75 | 250 | 750 | `sft_v2_ablation/Q4_BUY75/data/sft_buy75.jsonl` |
| Q5_BUY100 | 0 | 1000 | `sft_v2_ablation/Q5_BUY100/data/sft_buy100.jsonl` |

---

## V1 vs V2 핵심 차이

| | V1 Ablation | V2 Ablation |
|---|---|---|
| **프롬프트 형식** | Evidence 기반 (평가와 동일) | 단순 Q&A (섹터만) |
| **데이터 소스** | Qwen baseline 평가 응답 | gpt-oss-20b 생성 |
| **티커 포함** | O (427개) | X |
| **섹터 범위** | 11개 전체 | 2개 (Tech, Financial) |
| **교정 방향** | 전체 BUY/SELL 비율 조절 | 섹터별 방향 교정 (Tech→SELL, Fin→BUY) |
| **가설** | 비율에 비례한 선형 변화 | 섹터 개념 교정 → 티커 레벨 전이 |
