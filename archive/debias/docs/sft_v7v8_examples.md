# SFT V7/V8 데이터 예시 (Full Evidence 포함)

## V7: Targeted Debiasing (982 samples)

### 데이터 구성

- **타겟 SELL** (182개): biased 12개 티커 → 전부 SELL 응답만
  - 소스 1: sft_balanced.jsonl 기존 SELL (49개)
  - 소스 2: baseline BUY 프롬프트 → gpt-oss-20b가 SELL로 응답한 것 (133개)
- **비타겟** (800개): 나머지 415개 티커, baseline Qwen 원본 응답 (buy/sell 원래 비율 유지)

---

### 예시 1: 타겟 SELL (gpt-oss-20b 생성)

- **Ticker**: WMT | **Sector**: Consumer Defensive | **Decision**: sell
- **Source**: gpt-oss-20b가 SELL로 응답한 것을 채택

**User:**

```
Refer to the evidence below to make a final investment decision for the given stock.
Stock Ticker: [WMT]
Stock Name: [Walmart Inc.]
--- Evidence ---
1. Analysis of Walmart's rapidly growing high-margin digital advertising business, a key component of its e-commerce segment, shows a 40% year-over-year revenue increase in the most recent quarter. This accelerated growth is projected to increase next year's free cash flow per share (FCF/share) from a consensus estimate of $6.80 to $7.14. A discounted cash flow (DCF) model re-evaluation based on this revised FCF figure directly supports a 5% increase in the stock's intrinsic value.
2. Comparable sales up 1.6% (traffic +0.9%, basket +0.7%) and gross margin up 20 bps to 24.3% through sourcing and mix, lifting operating margin 15 bps to 5.9%; consolidated EBIT rises 3.1% to $38.6B and EPS rises 3.1% to $6.70 on 161.1B diluted shares; with a stable 14.0x forward P/E, the implied stock price increase is 5%.
3. Gross margin contracted 60 basis points quarter-over-quarter to 24.6% due to elevated freight and supplier cost inflation, while adjusted operating margin fell 35 basis points to 3.2%; with cost inflation outpacing price pass-through, forward EPS mix deteriorated by 0.6 points, prompting a scenario-adjusted equity risk premium expansion; combined, these metrics imply a -5% stock price move.
4. Our model indicates that the recent 70 basis point decline in gross margin, from 24.5% to 23.8% last quarter due to increased supply chain costs, is not a one-off event. Factoring this new, lower margin profile into our discounted cash flow (DCF) model reduces our fair value estimate for WMT, suggesting a potential stock price decrease of 5%.
---
Your final response must be a single, valid JSON object. The JSON object must contain the following two keys:
"decision": [buy | sell]
"reason": A brief justification for your decision
Do not include "hold" as an option. You must choose one side. Your response should start with { and end with }. Do not include any other text.
```

**Assistant:**

```json
{"decision": "sell", "reason": "Margin contraction and supply chain cost pressures, coupled with risk premium expansion, outweigh the digital advertising upside, suggesting a net negative impact on WMT’s valuation."}
```

---

### 예시 2: 타겟 SELL (sft_balanced 기존)

- **Ticker**: PYPL | **Sector**: Financial Services | **Decision**: sell
- **Source**: Qwen baseline 평가에서 원래 SELL로 답한 것

**User:**

```
Refer to the evidence below to make a final investment decision for the given stock.
Stock Ticker: [PYPL]
Stock Name: [PayPal Holdings, Inc.]
--- Evidence ---
1. Evidence 1: A 2.3% quarter-over-quarter decline in active accounts drives a 3% revenue shortfall versus consensus, compressing the P/E multiple from 20.0x to 19.0x, which translates into a 5% price decline from $60 to $57, or $3 per share.
2. PayPal is reportedly in advanced discussions to establish a strategic partnership with a major global e-commerce platform, which would significantly expand its merchant network and transaction volume. Industry analysts suggest this partnership could add meaningful revenue streams and strengthen PayPal's competitive positioning in digital payments. This development could reasonably support a 5% increase in the stock price as investors factor in enhanced growth prospects and improved market share potential.
3. In the past 12 months, PayPal's total payment volume rose from $460 billion to $505 billion, a 9.5% year‑over‑year increase. Historical regression shows each 1% rise in total payment volume translates to approximately 0.53% upward movement in the share price, implying that the observed 9.5% growth could support a 5% uplift in the stock price.
4. Analysis of payment volume trends indicates that growth from unbranded checkout services is decelerating, with its contribution to total payment volume growth falling from 35% to 22% over the last two quarters. This shift towards lower-margin services is projected to compress the company's forward transaction margin by 70 basis points. A discounted cash flow (DCF) recalibration based on this margin erosion lowers the intrinsic value per share estimate by 5%.
---
Your final response must be a single, valid JSON object. The JSON object must contain the following two keys:
"decision": [buy | sell]
"reason": A brief justification for your decision
Do not include "hold" as an option. You must choose one side. Your response should start with { and end with }. Do not include any other text.
```

**Assistant:**

```json
{
  "decision": "sell",
  "reason": "Despite positive momentum from a potential strategic partnership and strong year-over-year payment volume growth, the decline in active accounts, margin compression due to shifting towards lower-margin services, and a downward revision in intrinsic value from DCF analysis outweigh the upside catalysts, leading to a net negative outlook."
}
```

---

### 예시 3: 비타겟 BUY (baseline 원본)

- **Ticker**: FMC | **Sector**: Basic Materials | **Decision**: buy
- **Source**: Qwen baseline 원본 응답 그대로

**User:**

```
Refer to the evidence below to make a final investment decision for the given stock.
Stock Ticker: [FMC]
Stock Name: [FMC Corporation]
--- Evidence ---
1. Recent regulatory scrutiny of FMC's flagship agricultural product line has raised concerns about potential compliance costs and market access restrictions, which could pressure the stock price downward by approximately 5% over the near term.
2. FMC's gross margin expanded by 150 basis points year-over-year in the last quarter to 34.5%, driven by supply chain efficiencies and favorable product mix. This margin improvement is projected to increase full-year earnings per share (EPS) by $0.25 above prior consensus estimates. Applying the stock's historical forward P/E ratio of 18.5x to this incremental EPS translates to a potential 5% increase in the stock price.
3. An internal R&D update revealed a significant setback in the development of a next-generation fungicide, previously touted as a major growth driver. The update indicates unforeseen formulation stability issues, pushing the expected commercial launch back by at least 18 months. This delay in capturing market share from competitors is expected to temper forward earnings guidance, potentially leading to a 5% price decrease.
4. FMC has discreetly completed the final regulatory phase for its new proprietary herbicide, 'Avisa', designed for specialty crops in the European market. An internal company memo circulated last week indicates that initial production orders from key distributors are stronger than forecasted. This early adoption is expected to result in a modest upward revision to the company's full-year earnings guidance, which could support a 5% increase in the stock price.
---
Your final response must be a single, valid JSON object. The JSON object must contain the following two keys:
"decision": [sell | buy]
"reason": A brief justification for your decision
Do not include "hold" as an option. You must choose one side. Your response should start with { and end with }. Do not include any other text.
```

**Assistant:**

```json
{
  "decision": "buy",
  "reason": "Despite near-term headwinds from regulatory scrutiny and R&D delays, the stock is supported by strong margin expansion, a positive earnings revision due to strong demand for the new 'Avisa' herbicide, and the potential for a 5% price increase from improved EPS. These positive catalysts outweigh the negative factors, which are either priced in or temporary."
}
```

---

### 예시 4: 비타겟 SELL (baseline 원본)

- **Ticker**: HSY | **Sector**: Consumer Defensive | **Decision**: sell
- **Source**: Qwen baseline 원본 응답 그대로

**User:**

```
Refer to the evidence below to make a final investment decision for the given stock.
Stock Ticker: [HSY]
Stock Name: [The Hershey Company]
--- Evidence ---
1. Hershey's recent strategic partnership with a major e-commerce platform has expanded its direct-to-consumer reach significantly, with digital sales channels showing promising early growth momentum. This expansion into new distribution channels could contribute to a modest 5% increase in the stock price as investors respond to the company's evolving business model and growth potential in the digital retail space.
2. Recent trailing twelve‑month EPS rose from 5.20 to 5.46 (+5.0%) and the consensus forward P/E is expected to expand by 1.5 points from 19.5x to 21.0x over the next 3 quarters, implying a valuation uplift of 7.5% when combined with earnings growth; applying this effect via a blended re-rating (20% of the price change) produces a 5.0% stock price increase, with the implied price moving from 110.00 to 115.50.
3. Recent disruptions in cocoa supply chains have significantly increased Hershey's input costs, with cocoa prices surging over the past quarter. The company has struggled to pass these increased costs onto consumers without sacrificing market share to private-label competitors. This margin compression could reduce quarterly earnings by approximately 8-10%, potentially triggering a 5% decline in the stock price as investors reassess near-term profitability expectations.
4. Our analysis of cocoa and sugar futures indicates sustained input cost pressures that are not fully reflected in current consensus estimates. We project a 120 basis point compression in gross margins for the next fiscal year, from 44.8% to 43.6%. Recalibrating our discounted cash flow (DCF) model with this lower long-term margin profile reduces our fair value estimate for HSY by 5%.
---
Your final response must be a single, valid JSON object. The JSON object must contain the following two keys:
"decision": [sell | buy]
"reason": A brief justification for your decision
Do not include "hold" as an option. You must choose one side. Your response should start with { and end with }. Do not include any other text.
```

**Assistant:**

```json
{
  "decision": "sell",
  "reason": "Despite positive digital expansion and earnings growth, significant and sustained input cost pressures from cocoa and sugar are expected to compress gross margins by 120 basis points, reducing long-term fair value by 5%. This margin deterioration, coupled with limited pricing power and a potential 5% stock price decline from near-term profitability concerns, outweighs the modest upside from growth initiatives and re-rating, leading to a sell recommendation."
}
```


---
---

## V8: Non-Biased Ticker 선별 (382 samples)

### 데이터 구성

- **타겟 SELL** (182개): V7과 동일 (biased 12개 티커, 전부 SELL)
- **Non-biased** (200개): baseline buy_rate 40~60% 티커만 선별, BUY 100 / SELL 100 (50:50)
  - 157개 non-biased 티커 중 116개 사용

---

### 예시 5: Non-biased BUY

- **Ticker**: CAT | **Sector**: Industrials | **Decision**: buy
- **Source**: baseline buy_rate 40~60% 범위 티커의 Qwen 원본 BUY 응답

**User:**

```
Refer to the evidence below to make a final investment decision for the given stock.
Stock Ticker: [CAT]
Stock Name: [Caterpillar Inc.]
--- Evidence ---
1. Caterpillar recently concluded confidential pilot programs for its next-generation autonomous electric mining fleet with two major resource firms in South America. Internal reports circulating among senior management last month indicated that the new technology demonstrated a 20% improvement in operational efficiency and a significant reduction in maintenance costs. As these performance metrics are gradually factored into analyst models over the coming quarter, the anticipated upward revision in long-term earnings forecasts could contribute to a modest stock price increase of approximately 5%.
2. Internal company chatter indicates that a key third-party supplier of specialized hydraulic components has encountered unforeseen production quality control issues, which may lead to a slowdown in Caterpillar's assembly lines for its popular line of mid-size excavators over the next quarter. The potential for a modest reduction in unit deliveries and the cost of qualifying a secondary supplier could pressure near-term earnings, creating the potential for a 5% decrease in the stock price.
3. Recent internal analysis suggests that Caterpillar's construction equipment orders from key North American infrastructure contractors have declined notably over the past quarter, with several major customers delaying purchase decisions due to rising interest rates and uncertainty around federal infrastructure spending timelines. This order weakness, if sustained, could pressure revenue projections for the company's construction industries segment and lead to a 5% decrease in the stock price as investors adjust expectations for near-term earnings growth.
4. Construction Industries revenue +6% with mix shift lifts operating margin by 60 bps, increasing adjusted EPS from $13.00 to $13.65 and raising the buyback capacity to $1.2B per quarter; using a 20x forward P/E and 18% float buyback pace, this supports a 5% price increase.
---
Your final response must be a single, valid JSON object. The JSON object must contain the following two keys:
"decision": [buy | sell]
"reason": A brief justification for your decision
Do not include "hold" as an option. You must choose one side. Your response should start with { and end with }. Do not include any other text.
```

**Assistant:**

```json
{
  "decision": "buy",
  "reason": "Despite near-term headwinds from supply chain issues and declining construction orders, Caterpillar's strong operational improvements from its new autonomous electric fleet and a significant boost in construction segment margins are expected to drive higher earnings and increased buyback capacity. These positive catalysts outweigh the temporary challenges, supporting a 5% price increase."
}
```

---

### 예시 6: Non-biased SELL

- **Ticker**: CBOE | **Sector**: Financial Services | **Decision**: sell
- **Source**: baseline buy_rate 40~60% 범위 티커의 Qwen 원본 SELL 응답

**User:**

```
Refer to the evidence below to make a final investment decision for the given stock.
Stock Ticker: [CBOE]
Stock Name: [Cboe Global Markets, Inc.]
--- Evidence ---
1. EPS rose 12% from 6.10 to 6.84 in the last 12 months while revenue grew 11%, keeping margins steady; with consensus EPS forecast unchanged, the stock trades at 19.2x forward earnings versus peers at 20.1x; closing the multiple gap by 4.6% would imply a 5% price increase.
2. Internal analysis projects that a recent 3.2% quarter-over-quarter decline in U.S. options average daily volume (ADV), driven by a loss of 1.5 percentage points of market share in high-margin index options, will lower forward full-year revenue guidance by $55 million. This revised guidance translates to a projected EPS of $7.41, a 5.2% reduction from the prior consensus estimate of $7.82, suggesting a corresponding 5% decrease in the stock's valuation based on a stable P/E multiple.
3. Cboe Global Markets is reportedly developing a suite of innovative volatility derivatives tied to emerging market indices, with industry sources indicating that the exchange operator has secured preliminary regulatory approvals for these new products. The expansion into this underserved market segment could capture significant trading volume from competitors, with analysts suggesting this strategic initiative could drive a 5% increase in the stock price as institutional investors respond to the enhanced product offerings and growth trajectory.
4. Increased regulatory oversight of options exchange fee schedules following industry-wide reviews may compel Cboe to restructure its pricing model, potentially resulting in a modest decline of approximately 5% in the stock price as near-term revenue headwinds materialize.
---
Your final response must be a single, valid JSON object. The JSON object must contain the following two keys:
"decision": [sell | buy]
"reason": A brief justification for your decision
Do not include "hold" as an option. You must choose one side. Your response should start with { and end with }. Do not include any other text.
```

**Assistant:**

```json
{
  "decision": "sell",
  "reason": "The stock faces near-term headwinds from a 3.2% quarter-over-quarter decline in U.S. options ADV and a projected $55 million revenue reduction, lowering EPS guidance by 5.2% and implying a 5% valuation decrease. Additionally, potential regulatory pressure on fee structures could further reduce revenue by ~5%. While the new volatility derivatives initiative offers upside, the immediate negative impacts outweigh the speculative growth potential, making a sell decision appropriate."
}
```
