# LLM Investment Bias Debiasing: Summary Report

**UNIST Financial Engineering Lab** | February 2026

---

## 1. Background

Our prior work (*"Your AI, Not Your View"*, ICAIF 2025) demonstrated that LLMs exhibit systematic biases in investment decisions—particularly a preference for **Technology stocks** and **Large-cap stocks**. For example, Qwen3-30B recommends "BUY" for 86% of trials on certain tickers, far above the expected 50% under balanced evidence.

**Core question**: Can we correct (debias) these biases through fine-tuning, while preserving the model's general capabilities?

## 2. Experimental Setup

### Base Model & Evaluation

- **Backbone**: Qwen/Qwen3-30B-A3B-Instruct (MoE, open-source)
- **Baseline bias_index**: 172 (composite of sector bias + size bias; lower = less biased)
- **Evaluation**: 427 S&P 500 tickers × 3 sets × 10 trials = 12,810 inferences per checkpoint
  - Each prompt contains 2 positive + 2 negative evidence items → model outputs `{"decision": "buy/sell", "reason": "..."}`
- **Training data generation**: gpt-oss-20b (low-bias teacher model) generates chosen/rejected scenarios

### Training Configuration (All Experiments)

- LoRA: r=32, alpha=64, target modules: qkv_proj, o_proj
- lr=5e-6, cosine scheduler, warmup 10%, effective batch=64
- bf16, seed=42, early stopping patience=3

## 3. Approach 1: DPO (Direct Preference Optimization)

DPO learns from preference pairs: the model is trained to prefer "chosen" over "rejected" responses.

### Data Format

Each sample is a **(prompt, chosen, rejected)** triplet:

- **Prompt**: A perspective-based analysis question (no evidence)
- **Chosen**: Pessimistic analysis (pushes toward SELL)
- **Rejected**: Optimistic analysis (pushes toward BUY)
- **5 perspectives**: growth, financial, competitive, valuation, macro

### Experiments

| Experiment | Target Tickers | # Tickers | Samples | Direction | Best Bias Index |
|---|---|--:|--:|---|--:|
| **Ticker-level** | BUY-biased (buy_rate≥80%) | 12 | 1,020 | buy→sell | **46** (step 171) |
| **Sector-level** | Technology sector | 58 | 1,160 | buy→sell | **86** (step 320) |
| **Size-level** | Q1 large-cap (top 25%) | 107 | 1,070 | buy→sell | **65** (step 300) |
| **Bidirectional** | Tech (→sell) + Financial (→buy) | sector-level | 1,000 | both | 165 (FAILED) |

### Key Findings

1. **Global shift, not targeted correction**: Training on 12 specific tickers causes ALL 427 tickers to shift toward SELL, not just the targets. No memorization—generalized debiasing.
2. **Overfitting risk**: Ticker-level DPO achieves bias_index=46 at step 171, but degrades to 268 by step 798. Checkpoint selection is critical.
3. **Bidirectional DPO fails**: Correcting Tech→SELL and Financial→BUY simultaneously amplifies sector differences rather than reducing them.

## 4. Approach 2: SFT (Supervised Fine-Tuning)

SFT learns from (prompt, response) pairs directly, using the same evidence-based format as evaluation.

### Data Format

Each sample is a **(user prompt, assistant response)** pair:

- **User**: Evidence-based investment decision prompt (identical format to evaluation)
- **Assistant**: `{"decision": "buy/sell", "reason": "..."}`

### Experiments

**Ablation Study (V1)**: Varying BUY/SELL ratio in 1,000 samples

| Condition | Composition | Best Bias Index |
|---|---|--:|
| 100% SELL | 1000 SELL, 0 BUY | **22** |
| 75% SELL | 750 SELL, 250 BUY | 51 |
| 50:50 | 500 SELL, 500 BUY | 111 |
| 75% BUY | 250 SELL, 750 BUY | 197 |
| 100% BUY | 0 SELL, 1000 BUY | 336 |

→ Model follows training data ratio faithfully. SELL direction is slightly stronger.

**Ticker-Targeted (V7)**: Biased tickers SELL + all others preserve original ratio

| Component | Samples | Description |
|---|--:|---|
| Target SELL | 182 | 12 biased tickers (buy_rate≥80%), all SELL |
| Non-target | 800 | 415 other tickers, original buy/sell ratio preserved |
| **Total** | **982** | |

→ Best bias_index = **44** (step 96). But non-target tickers also shift toward SELL (spillover -7%p).

**Non-biased Selection (V8)**: Biased tickers SELL + only balanced tickers

| Component | Samples | Description |
|---|--:|---|
| Target SELL | 182 | Same 12 biased tickers, all SELL |
| Non-biased | 200 | 116 tickers with baseline buy_rate 40–60%, BUY:SELL = 50:50 |
| **Total** | **382** | |

→ Best bias_index = **64** (step 78). Stronger target correction (78% vs 83%) but larger spillover (-13%p).

## 5. Summary of Results

| Method | Best Bias Index | Target Correction | Spillover | Notes |
|---|--:|---|---|---|
| Baseline | 172 | — | — | |
| DPO Ticker (12) | **46** | Global shift | High | Overfits after step 171 |
| DPO Sector (58) | 86 | Global shift | Moderate | Slower convergence |
| DPO Size (107) | 65 | Global shift | Moderate | |
| SFT V7 (982) | **44** | Weak (-3%p) | Low (-7%p) | Best overall balance |
| SFT V8 (382) | 64 | Moderate (-8%p) | High (-13%p) | |

### Conclusions

1. **Debiasing is achievable**: Both DPO and SFT can reduce bias_index from 172 to ~44 (74% reduction)
2. **No selective targeting**: All methods cause global opinion shift rather than ticker/sector-specific correction. Training on 12 tickers affects all 427.
3. **SFT follows data ratio**: BUY/SELL ratio in training data directly controls model behavior—useful for controlled tilting
4. **Checkpoint selection matters**: Optimal point exists; training beyond it leads to overcorrection or degradation

---

## Appendix: Data Examples

### A1. DPO Training Data Example (Ticker-level)

**Prompt:**
```
Analyze TSN (Tyson Foods, Inc.) from a competitive perspective.
```

**Chosen (pessimistic → induces SELL):**
```
Tyson Foods' competitive position in the consumer defensive sector is
deteriorating under mounting pressure from both structural industry shifts
and intensified competition. The company's traditional moat—built on scale,
integrated supply chains, and long-standing relationships with retailers—has
been eroding as plant-based protein alternatives gain mainstream traction...
```

**Rejected (optimistic → induces BUY):**
```
Tyson Foods, Inc. stands as a dominant force in the consumer defensive
sector, particularly within the protein industry, reinforcing its market
leadership through a vertically integrated supply chain and unparalleled
scale. As the largest meat processor in the U.S. and a global leader in
chicken, beef, and pork...
```

### A2. SFT Training Data Example — Target SELL (V7/V8)

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

### A3. SFT Training Data Example — Non-target BUY (V7)

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
[same instruction as above]
```

**Assistant:**
```json
{"decision": "buy", "reason": "Despite near-term headwinds from regulatory scrutiny
and R&D delays, the stock is supported by strong margin expansion, a positive earnings
revision due to strong demand for the new 'Avisa' herbicide, and the potential for a
5% price increase from improved EPS."}
```

### A4. SFT Training Data Example — Non-biased SELL (V8 only)

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
[same instruction as above]
```

**Assistant:**
```json
{"decision": "sell", "reason": "The stock faces near-term headwinds from a 3.2%
quarter-over-quarter decline in U.S. options ADV and a projected $55 million revenue
reduction, lowering EPS guidance by 5.2% and implying a 5% valuation decrease.
Additionally, potential regulatory pressure on fee structures could further reduce
revenue by ~5%."}
```
