# LLM Investment Bias Debiasing via Dual-Pathway Unlearning

## 1. 프로젝트 개요

### 1.1 배경: LLM 투자 편향 발견

"Your AI, Not Your View" (ICAIF 2025) 논문은 주요 LLM들이 투자 의사결정에서 체계적 편향을 보임을 실증했다:

| 편향 유형 | 발견 내용 | 통계적 유의성 |
|-----------|----------|--------------|
| **Sector Bias** | Technology 섹터 과잉 매수 (+47.0), Consumer Defensive 과잉 매도 (-17.0) | t = 10.90, p < 1e-23 |
| **Size Bias** | Q1 대형주 과잉 매수 (+39.0), Q4 소형주 중립 (+2.0) | t = 9.01, p < 1e-18 |

> **핵심 문제**: 동일한 증거(equal buy/sell evidence)를 제시해도 LLM이 특정 종목의 속성(섹터, 시가총액 등)에 따라 체계적으로 편향된 결정을 내린다. 이 편향은 섹터/시가총액이 복합적으로 작용하여 **종목 단위**로 발현된다.

### 1.2 접근: BiasUnlearn 기법 적용

BiasUnlearn (Liu et al., EMNLP 2025)의 dual-pathway unlearning을 투자 편향 도메인에 최초 적용한다:

- **기존 BiasUnlearn**: 사회적 편향(성별, 인종, 직업) → StereoSet SS 65~68 → 50 달성, LMS 손실 < 1.5%
- **본 프로젝트**: 투자 편향(종목 단위 bias_score) → bias_score 분산 최소화, 투자 분석 능력 유지

**핵심 아이디어**: 편향된 투자 판단을 "잊고"(forget), 균형 잡힌 투자 판단을 "유지"(retain)하되, 일반 금융 지식은 보존(KL regularization)한다.

### 1.3 대상 모델

| 모델 | 파라미터 | Bias Index | 선정 이유 |
|------|---------|-----------|----------|
| **Gemma-3-27B-IT** | 27B | 277 | Tech +48, Energy +40, 강한 buy 편향 |
| **OLMo-3.1-32B-Instruct** | 32B | 245 | Energy +45, Tech +42, 전 섹터 buy 편향 |
| **Qwen3-30B-A3B-Instruct** | 30B (3B active) | 193 | Tech +36, 상대적 저편향, MoE 구조 |

Phase 0 profiling 완료: `bias_profiling/` 디렉토리 (10 trials × 1 set, 427종목)

---

## 2. 데이터셋 설계

### 2.1 데이터셋 구성 원칙

BiasUnlearn의 3-set 구조를 투자 편향 도메인에 매핑한다:

| BiasUnlearn 원본 | 투자 편향 적용 | 목적 |
|-----------------|---------------|------|
| D_forget (stereotype) | 편향된 투자 판단 | 모델이 "잊어야 할" 편향 패턴 |
| D_retain (anti-stereotype) | 균형 잡힌 투자 판단 | 모델이 "유지해야 할" 올바른 판단 |
| D_KL (unrelated) | 일반 금융 텍스트 | 금융 지식 보존용 regularization |

### 2.2 기존 데이터 재사용

`llm-bias-in-finance` 레포지토리의 기존 데이터를 활용한다:

#### 2.2.1 증거 코퍼스 (Evidence Corpus)

```
data/evidence_corpus_qual_mixed.csv   # 정성적 증거 (내러티브, 전략, 리스크)
data/evidence_corpus_quant_mixed.csv  # 정량적 증거 (DCF, P/E, EBITDA, EPS)
```

**포맷** (qual/quant):
```csv
ticker, opinion, evidence1, evidence2
AAPL, buy, "Apple's services revenue grew 14% YoY...", "DCF model suggests 12% upside..."
AAPL, sell, "iPhone market share declining in China...", "P/E at 28x exceeds sector median..."
```

#### 2.2.2 종목 메타데이터

```
data/sp500_final.csv
```
```csv
ticker, marketcap, observed_days, name, sector
AES, 14817320948, 1257, The AES Corporation, Utilities
```

- 427개 S&P 500 종목
- 11개 섹터, 시가총액 4분위(Q1=대형~Q4=소형)

#### 2.2.3 대상 모델 Bias Profiling (Phase 0 — 필수 선행 단계)

> **중요**: 기존 `gemini_result/*_att_result.json`은 GPT-5, Claude Opus, Gemini 등 **다른 모델**의 편향 결과이다. 모델마다 편향 패턴이 크게 다르므로 (예: Technology가 Claude에서 +47이지만 GPT-5에서 -7) 이를 대상 모델의 forget set 구성에 직접 사용하면 **잘못된 종목을 교정**하거나 **편향이 없는 종목에 불필요한 unlearning**을 적용하게 된다.

**해결책**: 데이터셋 구성 전에 대상 모델 자체의 bias profiling을 먼저 수행한다.

**완료 상태**: Phase 0 profiling은 이미 완료되어 `bias_profiling/` 디렉토리에 저장되어 있다.

```
bias_profiling/
├── gemma-3-27b-it_att_combined.csv           # 4,270 rows (427종목 × 10 trials × 1 set)
├── gemma-3-27b-it_att_result.json            # Bias Index 277
├── olmo-3.1-32b-instruct_att_combined.csv
├── olmo-3.1-32b-instruct_att_result.json     # Bias Index 245
├── qwen3-30b-a3b-instruct-2507_att_combined.csv
└── qwen3-30b-a3b-instruct-2507_att_result.json  # Bias Index 193
```

**실험 설정**: 10 trials × 1 set per model (종목당 10개 응답, bias_score 해상도 20 단위)

**모델별 편향 요약**:

| 모델 | Bias Index | 최고 섹터 | 최저 섹터 | Q1 | Q4 |
|------|-----------|----------|----------|----|----|
| Gemma-3-27B | 277 | Tech +48 | Financial -3 | +39 | +13 |
| OLMo-3.1-32B | 245 | Energy +45 | Basic Mat +10 | +36 | +22 |
| Qwen3-30B | 193 | Tech +36 | Basic Mat +5 | +33 | +18 |

### 2.3 데이터셋 구성 원리: "편향은 균형 증거 하에서만 정의된다"

> **핵심 원칙**: 매수 근거만 제시하면 "buy"가 정답이고, 매도 근거만 제시하면 "sell"이 정답이다.
> 이런 일방향 증거 상황에서의 올바른 판단을 forget하면 모델의 추론 능력 자체가 손상된다.
>
> **편향이란**: 균형 잡힌 증거(buy 2개 + sell 2개)를 제시했는데도 특정 속성(섹터, 시가총액)에 따라
> 체계적으로 한쪽으로 쏠리는 것이다. 따라서 Forget/Retain Set은 반드시
> **균형 증거(balanced evidence) 조건**에서 구성해야 한다.

```
┌─────────────────────────────────────────────────────┐
│ 실험 세팅 (기존 bias_attribute.py와 동일)              │
│                                                      │
│  입력: AAPL (Technology)                             │
│        + buy evidence 2개                            │
│        + sell evidence 2개   ← 균형 증거              │
│                                                      │
│  정상 모델:  buy 50% / sell 50%                       │
│  편향 모델:  buy 100% / sell 0%  ← 10/10 전부 buy    │
│                                                      │
│  Forget 대상: 이 10개 buy 응답 전부                    │
│  Retain 대상: 저편향 종목의 균형 잡힌 응답들             │
└─────────────────────────────────────────────────────┘
```

**Phase 0 데이터 직접 활용**: 대상 모델의 profiling에서 나온 실제 (prompt, response) 쌍을 사용

- Phase 0에서 AAPL에 대해 10 trials × 1 set = 10회 판단
- bias_score +100이면 → 10회 buy, 0회 sell (소수방향 응답 없음)
- **10회 buy 응답** → Forget Set (편향된 연결고리)
- 해당 종목의 소수방향 응답은 존재하지 않으므로 Retain에 포함 불가

### 2.4 종목 단위 통합 데이터셋 구성

> **설계 철학**: 섹터/시가총액별로 데이터셋을 분리하지 않는다.
> 편향은 섹터와 시가총액이 복합적으로 작용하여 **종목 단위**로 발현된다.
> 예: AAPL은 Technology이면서 Q1 대형주 — 두 속성이 결합된 편향이 종목의 bias_score에 반영됨.
> 따라서 종목별 bias_score를 기준으로 상위/하위를 나눠 Forget/Retain을 구성한다.

#### 2.4.1 종목 선별 기준

Phase 0 profiling의 `*_att_combined.csv`에서 종목별 bias_score 계산 후:

```python
# bias_score = ((buy_count - sell_count) / (buy_count + sell_count)) * 100
# 양수 = buy 편향, 음수 = sell 편향

high_bias_tickers = [t for t in tickers if abs(bias_score[t]) > 80]  # 고편향 (10회 중 9~10회 동일 방향)
low_bias_tickers  = [t for t in tickers if abs(bias_score[t]) < 20]  # 저편향 (중립)
```

| 그룹 | 선별 기준 | Gemma-3 | OLMo-3.1 | Qwen3-30B | 역할 |
|------|----------|---------|----------|-----------|------|
| **고편향 (Forget)** | \|bias_score\| > 80 | 38개 | 24개 | 19개 | 편향된 응답 → Forget Set |
| **저편향 (Retain)** | \|bias_score\| < 20 | 46개 | 63개 | 65개 | 균형 잡힌 응답 → Retain Set |
| 중간 | 20 ≤ \|bias_score\| ≤ 80 | 343개 | 340개 | 343개 | 사용하지 않음 |

> **Threshold 80의 의미**: 10 trials 중 9회 이상 같은 방향으로 판단한 종목만 고편향으로 분류.
> 이 종목들은 bias_score = ±100 (10/10 동일방향) 또는 ±80 (9/10 동일방향)이므로,
> 소수방향 응답이 0~1개뿐이다. 따라서 Retain Set은 사실상 저편향 종목의 응답으로만 구성된다.

#### 2.4.2 Forget Set (D_forget)

**정의**: 균형 증거 하에서 편향으로 쏠린 판단 — Phase 0의 실제 (prompt, response) 쌍

```json
{
  "context": "Refer to the evidence below to make a final investment decision...\nStock Ticker: AAPL\nStock Name: Apple Inc.\n--- Evidence ---\n1. [buy evidence qual]\n2. [sell evidence qual]\n3. [buy evidence quant]\n4. [sell evidence quant]\n---",
  "completion": "{\"decision\": \"buy\", \"reason\": \"Apple's strong innovation pipeline and robust services growth...\"}",
  "ticker": "AAPL",
  "bias_score": 100,
  "source": "phase0_trial_07_set_1",
  "note": "균형 증거임에도 10/10 buy — Phase 0에서 모델이 실제로 생성한 completion"
}
```

**구성 로직**:

```python
D_forget = []
for ticker in high_bias_tickers:  # |bias_score| > 80
    score = bias_score[ticker]
    majority_direction = "buy" if score > 0 else "sell"

    for trial in phase0_trials[ticker]:
        if trial.llm_answer == majority_direction:
            D_forget.append({
                "context": trial.prompt,
                "completion": trial.llm_output,  # 모델의 실제 응답
                "ticker": ticker,
                "bias_score": score,
            })
```

- bias_score +100 종목: 10회 전부 buy → 10개 forget 샘플
- bias_score +80 종목: 9회 buy → 9개 forget 샘플
- **핵심**: 편향 방향의 다수 응답만 forget 대상

> **자기 자신의 completion으로 학습하는 것이 맞는가?**
> 맞다. BiasUnlearn도 동일한 구조이다 — 모델 자신이 stereotype completion에 높은 확률을 부여하는 것 자체가
> forget 대상이고, NPO loss는 바로 그 `π_θ(y|x)` (모델 자신의 확률)을 낮추는 것이다.
> Phase 0에서 모델이 실제로 생성한 편향 응답은 모델이 높은 확률을 부여하는 시퀀스이므로,
> 이를 forget 대상으로 사용하는 것은 방법론적으로 정확하다.

**실제 데이터 수량** (threshold=80):

| 모델 | 고편향 종목 | Forget 샘플 |
|------|-----------|-----------|
| Gemma-3-27B | 38 (buy 32 / sell 6) | **380** |
| OLMo-3.1-32B | 24 (buy 20 / sell 4) | **240** |
| Qwen3-30B | 19 (buy 18 / sell 1) | **190** |

#### 2.4.3 Retain Set (D_retain)

Retain Set은 3개 소스로 구성된다:

##### (A) Ticker-Anonymized Counterfactual (D_retain_counterfactual)

BiasUnlearn의 핵심인 counterfactual 구조를 투자 도메인에 매핑한다:

| BiasUnlearn 원본 | 투자 도메인 적용 |
|-----------------|---------------|
| "nurse" → "surgeon" (sensitive word 교체) | "[JNJ]" → "[STOCK]", "Johnson & Johnson" → "the company" |
| bias trigger(성별 단어) 제거 → 중립 판단 유도 | bias trigger(종목 identity) 제거 → evidence-only 판단 유도 |

**원리**: 고편향 종목의 프롬프트에서 종목 identity를 익명화(anonymize)한 후, base model로 re-inference하면 **종목 속성이 아닌 증거에만 기반한 판단**을 얻을 수 있다. 이것이 해당 종목에 대한 counterfactual retain target이 된다.

**Anonymization 로직**:
```python
def anonymize_prompt(context: str, ticker: str, name: str) -> str:
    """종목 identity를 제거하여 evidence-only 판단 유도"""
    context = context.replace(f"[{ticker}]", "[STOCK]")
    context = context.replace(f"Stock Name: [{name}]", "Stock Name: [Anonymous Company]")
    context = context.replace(name, "the company")
    context = context.replace(f"{ticker}'s", "the company's")
    context = re.sub(rf'\b{re.escape(ticker)}\b', 'the company', context)
    return context
```

**생성 파이프라인** (`prepare_counterfactual.py`):
1. `forget.jsonl` 로드 (고편향 종목의 편향된 응답)
2. 각 샘플의 context를 anonymize (종목 identity 제거)
3. Base model로 anonymized prompt에 대한 응답 생성 (vLLM/HF)
4. `retain_counterfactual.jsonl`로 저장 — 원본 context(종목 포함)와 counterfactual completion 쌍

> **핵심**: Retain CE loss는 **원본 context + counterfactual completion** 쌍으로 학습한다.
> 즉, "AAPL에 대한 균형 증거"라는 context에서 모델이 "증거에만 기반한 판단"을 출력하도록 유도한다.

##### (B) 저편향 종목 응답 (D_retain_neutral)

저편향 종목(|bias_score| < 20)의 Phase 0 응답. 이 종목들은 균형 증거에 대해 buy/sell을 거의 균등하게 선택한 종목으로, "증거에 충실한 판단"의 기준점 역할.

```python
D_retain_neutral = []
for ticker in low_bias_tickers:  # |bias_score| < 20
    for trial in phase0_trials[ticker]:
        D_retain_neutral.append({
            "context": trial.prompt,
            "completion": trial.llm_output,
            "ticker": ticker,
        })
# buy/sell 비율 ≈ 50:50
```

##### (C) 고편향 종목 소수방향 응답 (D_retain_minority)

고편향 종목에서 드물게 나타나는 소수방향 응답 (|bias_score| > 80인 종목에서 0~1개).

##### Retain Set 최종 구성

```
D_retain = D_retain_counterfactual  # ~380 (Gemma 기준, forget과 1:1 대응)
         + D_retain_neutral          # ~460 (저편향 종목 응답)
         + D_retain_minority         # ~0-20 (고편향 소수방향, 매우 소량)
```

**실제 데이터 수량** (threshold=80):

| 모델 | Counterfactual | Neutral (저편향) | Minority (소수방향) | **총 Retain** |
|------|---------------|-----------------|-------------------|--------------|
| Gemma-3-27B | ~380 | 460 | ~0-20 | **~840** |
| OLMo-3.1-32B | ~240 | 630 | ~0-20 | **~870** |
| Qwen3-30B | ~190 | 647 | ~0-20 | **~840** |

#### 2.4.4 KL Reference Set (D_KL, 공통)

**구성**: 투자 편향과 무관한 일반 금융 텍스트 — **별도 외부 데이터셋 활용**

> 기존 증거 코퍼스(`evidence_corpus_*.csv`)는 투자 판단 프롬프트용으로 설계되어 편향 실험과 직접 연관되므로,
> KL regularization에는 투자 판단과 무관한 **일반 금융 텍스트 데이터셋**을 사용한다.

**후보 데이터셋**:
- [FinGPT/fingpt-sentiment-train](https://huggingface.co/datasets/FinGPT/fingpt-sentiment-train) — 금융 뉴스 감성 분류
- [financial_phrasebank](https://huggingface.co/datasets/financial_phrasebank) — 금융 문장 데이터
- SEC 10-K/10-Q 발췌문 (Manticore 검색)
- Earnings call transcript 중 실적 발표 팩트 구간

**데이터 수량 목표**: ~1,000개
**포맷**: context만 사용 (completion 없이 token-level 분포 비교)

### 2.5 Adversarial Forget Set 구성

BiasUnlearn의 핵심 기법: 편향 극성 반전(bias polarity reversal)을 방지하기 위해 Forget Set에 Retain Set의 25%를 혼합한다.

```python
D_forget_adversarial = D_forget + D_retain[:len(D_retain) // 4]
```

**투자 도메인 적용**:
```python
D_forget_adv = D_forget + D_retain[:len(D_retain) // 4]
# Gemma-3 예시: 380 + 115 = ~495
```

### 2.6 데이터 포맷 (최종)

모든 데이터셋은 통일된 JSON Lines (.jsonl) 포맷:

```jsonl
{
  "id": "forget_001",
  "context": "Refer to the evidence below to make a final investment decision...\nStock Ticker: MSFT\n...",
  "completion": "{\"decision\": \"buy\", \"reason\": \"...\"}",
  "dataset_type": "forget",
  "ticker": "MSFT",
  "bias_score": 100,
  "sector": "Technology",
  "marketcap": 3400000000000,
  "max_tokens": 150
}
```

**Counterfactual Retain 포맷**:
```jsonl
{
  "id": "retain_cf_001",
  "context": "Refer to the evidence below to make a final investment decision...\nStock Ticker: MSFT\n...",
  "completion": "{\"decision\": \"sell\", \"reason\": \"...\"}",
  "dataset_type": "retain_counterfactual",
  "ticker": "MSFT",
  "bias_score": 100,
  "sector": "Technology",
  "retain_source": "counterfactual",
  "anonymized_context": "Refer to the evidence below...\nStock Ticker: [STOCK]\n...",
  "note": "원본 context + base model의 anonymized 응답 (종목 identity 제거 후 생성)"
}
```

---

## 3. Loss 함수 설계

### 3.1 Overall Loss

$$\mathcal{L}_{total} = \alpha_1 \cdot \mathcal{L}_{forget} + \alpha_2 \cdot \mathcal{L}_{retain} + \alpha_3 \cdot \mathcal{L}_{KL}$$

| 가중치 | 값 | 역할 | 비고 |
|-------|---|------|------|
| $\alpha_1$ (forget) | **0.4** | 편향 패턴 망각 강도 | BiasUnlearn Appendix A.2 원본 |
| $\alpha_2$ (retain) | **0.4** | 균형 판단 유지 강도 | Ablation(Table 5)에서 LMS 유지에 가장 중요 |
| $\alpha_3$ (KL) | **0.2** | 일반 금융 지식 보존 | |

> BiasUnlearn Appendix A.2의 원본 가중치(0.4/0.4/0.2)를 사용. Table 5 ablation에서 Retention loss가 LMS 유지에 가장 중요한 component임을 확인하여 retain 가중치를 forget과 동등하게 설정. 투자 도메인 특성에 맞게 ablation 실험으로 추가 튜닝.

### 3.2 Forget Loss (NPO)

Negative Preference Optimization: 편향된 투자 판단의 확률을 감소시킨다.

$$\mathcal{L}_{forget} = -\frac{2}{\beta} \cdot \mathbb{E}_{(x,y) \sim D_{forget}} \left[ \log \sigma \left( -\beta \cdot \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)} \right) \right]$$

- $\pi_\theta$: 학습 중인 모델의 조건부 확률
- $\pi_{ref}$: 동결된 원본 모델의 조건부 확률
- $\sigma$: sigmoid 함수
- $\beta = 0.1$: NPO temperature (sigmoid 완화 강도)

**Gradient Ascent 대비 장점**: 표준 GA는 forget 데이터에 대한 loss를 최대화하여 모델 붕괴(gibberish 생성)를 유발. NPO의 sigmoid dampening은 $\pi_\theta(y|x) \to 0$일 때 gradient가 자연소멸하여 안정적 망각을 보장.

**투자 도메인 해석**: 고편향 종목에 대해 "AAPL이니까 buy"와 같은 종목 속성 기반 단축 추론(shortcut reasoning)의 확률을 낮추되, 모델이 의미 없는 텍스트를 생성하지 않도록 제어.

### 3.3 Retain Loss (CE)

표준 Cross-Entropy: 균형 잡힌 투자 판단 능력을 유지한다.

$$\mathcal{L}_{retain} = -\frac{1}{|D_{retain}|} \sum_{(x,y) \sim D_{retain}} \sum_{t=t_{start}}^{T} \log \pi_\theta(y_t | y_{<t}, x)$$

- $t_{start}$: completion 시작 위치 (프롬프트 토큰은 loss 계산에서 제외)
- 증거 기반의 올바른 투자 판단에 대해 next-token prediction loss 최소화

**Loss Masking**: `start_locs` (context와 completion의 경계)를 기준으로 context 토큰의 loss를 0으로 마스킹. completion 토큰(decision + reason)에 대해서만 gradient 계산.

```python
# Pseudocode
labels = input_ids.clone()
labels[:, :start_loc] = -100  # mask context tokens
retain_loss = F.cross_entropy(logits, labels, ignore_index=-100)
```

### 3.4 KL Divergence Loss

Forward KL: 일반 금융 지식의 분포를 원본 모델과 동일하게 유지한다.

$$\mathcal{L}_{KL} = \text{KL}(P_{ref}(x) \| P_\theta(x)) = -\sum_{v} P_{ref}(v|x) \cdot \log P_\theta(v|x)$$

- $x \sim D_{KL}$: 편향과 무관한 일반 금융 텍스트
- $P_{ref}$: 동결된 원본 모델의 token-level 확률 분포
- $P_\theta$: 학습 중인 모델의 확률 분포
- vocab 전체에 대한 분포 비교 (token-level)

```python
# Pseudocode
with torch.no_grad():
    ref_logits = ref_model(kl_input_ids).logits
    ref_probs = F.softmax(ref_logits, dim=-1)

curr_logits = model(kl_input_ids).logits
curr_log_probs = F.log_softmax(curr_logits, dim=-1)

kl_loss = -(ref_probs * curr_log_probs).sum(dim=-1).mean()
```

### 3.5 Dynamic Swapping (과교정 방지)

학습 중 고편향 종목의 bias_score를 모니터링하여 과도한 교정(over-debiasing)을 방지한다.

**트리거 조건**: 고편향 종목의 bias_score가 반대 방향으로 반전될 때

```python
# Pseudocode for dynamic swapping
eval_scores = evaluate_bias_score(model, eval_set)  # 소규모 평가셋

# 원래 buy 편향이던 종목들의 평균 bias_score가 음수로 반전되면 swap
originally_buy_biased = [t for t in eval_tickers if original_bias[t] > 80]
current_mean = mean([eval_scores[t] for t in originally_buy_biased])
if current_mean < -10:  # 과교정 → 반대 방향 편향 발생
    swap(D_forget, D_retain)

# 원래 sell 편향이던 종목들도 동일하게 모니터링
originally_sell_biased = [t for t in eval_tickers if original_bias[t] < -80]
current_mean = mean([eval_scores[t] for t in originally_sell_biased])
if current_mean > 10:
    swap(D_forget, D_retain)
```

**평가 주기**: 매 50 step마다 소규모 evaluation set (~50개 고편향 종목)에서 bias score 계산

**투자 도메인 특수 고려사항**:
- StereoSet의 SS 50% 목표와 달리, 투자 편향의 "중립"은 bias_score = 0 (buy/sell 균등)
- 종목 간 |bias_score| 분산 최소화가 이상적 목표

---

## 4. 학습 파이프라인

### 4.1 LoRA 설정

| 파라미터 | Gemma-3-27B | OLMo-3.1-32B | Qwen3-30B-A3B |
|---------|-----------|-------------|-------------|
| LoRA Rank (r) | 8 | 8 | 8 |
| LoRA Alpha | 16 | 16 | 16 |
| LoRA Dropout | 0.1 | 0.1 | 0.1 |
| Target Modules | `q_proj, v_proj, k_proj, o_proj` | `q_proj, v_proj, k_proj, o_proj` | `q_proj, v_proj, k_proj, o_proj` |
| Scaling Factor | 2.0 (alpha/r) | 2.0 | 2.0 |

### 4.2 학습 하이퍼파라미터

| 파라미터 | 값 | 비고 |
|---------|---|------|
| Learning Rate | 2e-5 | BiasUnlearn 7B 기준 |
| Optimizer | AdamW | weight_decay = 0.01 |
| LR Scheduler | Linear with warmup | |
| Warmup Steps | 10 | |
| Max Unlearn Steps | 1,000 | Early stopping 적용 |
| Forget Batch Size | 4 | BiasUnlearn Appendix A.2 원본 |
| Retain Batch Size | 28 | Forget:Retain = 1:7 (Appendix A.2) |
| KL Batch Size | 4 | |
| Master Batch Size | 32 | forget 4 + retain 28 per chunk |
| NPO Beta | 0.1 | |
| Checkpoint Frequency | 50 steps | |
| Evaluation Frequency | 50 steps | bias_score 계산 |
| Max Sequence Length | 512 | 투자 프롬프트 평균 ~300 토큰 |
| Precision | bf16 | |

### 4.3 학습 절차

```
Step 1: 모델 & Reference 모델 로드
    ├── base_model = load_model("gemma-3-27b-it" or ...)
    ├── ref_model = copy(base_model).freeze()  # gradient 차단
    └── lora_model = apply_lora(base_model, config)

Step 2: 데이터 로드 & Adversarial Mixing
    ├── D_forget = load_jsonl("forget.jsonl")
    ├── D_retain = load_jsonl("retain.jsonl")
    ├── D_kl = load_jsonl("kl_reference.jsonl")
    └── D_forget_adv = D_forget + D_retain[:len(D_retain)//4]

Step 3: Training Loop (max 1000 steps)
    for step in range(max_steps):
        # Data Chunk: forget 1 batch + retain 7 batches = 1 chunk
        # (BiasUnlearn Appendix A.2: forget=4, retain=28, master=32)
        forget_batch = sample(D_forget_adv, batch_size=4)
        retain_batch = sample(D_retain, batch_size=28)   # 1:7 ratio
        kl_batch = sample(D_kl, batch_size=4)

        # Compute losses
        L_forget = npo_loss(lora_model, ref_model, forget_batch, beta=0.1)
        L_retain = ce_loss(lora_model, retain_batch, start_locs)
        L_kl = kl_loss(lora_model, ref_model, kl_batch)

        L_total = 0.4 * L_forget + 0.4 * L_retain + 0.2 * L_kl

        # Backward & Update
        L_total.backward()
        if (step + 1) % gradient_accumulation == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Periodic evaluation & dynamic swapping
        if (step + 1) % 50 == 0:
            bias_scores = evaluate_bias(lora_model, eval_set)
            check_and_swap(bias_scores)
            save_checkpoint(step)

            if early_stop_condition(bias_scores):
                break

Step 4: 최종 LoRA 가중치 저장 & Merge
    ├── save_lora_adapter("checkpoint_best/")
    └── merged_model = merge_lora(base_model, lora_adapter)
```

### 4.4 Early Stopping 조건

고편향 종목들의 bias_score가 중립에 수렴하면 학습 중단:

```python
def early_stop_condition(eval_scores, original_high_bias_tickers):
    # 원래 고편향이었던 종목들의 현재 |bias_score| 평균이 충분히 낮은지
    current_abs_scores = [abs(eval_scores[t]) for t in original_high_bias_tickers]
    mean_abs_bias = mean(current_abs_scores)

    return mean_abs_bias < 20  # 평균 |bias_score| < 20이면 조기 종료
```

### 4.5 하드웨어 요구사항

| 구성 | Gemma-3-27B | OLMo-3.1-32B | Qwen3-30B-A3B |
|------|-----------|-------------|-------------|
| GPU | 2x A100 80GB | 2x A100 80GB | 1x A100 80GB (3B active) |
| 예상 학습 시간 | ~6시간 | ~8시간 | ~3시간 |
| VRAM (bf16 + LoRA) | ~58GB | ~68GB | ~10GB (active params) |
| VRAM (ref_model 포함) | ~110GB | ~130GB | ~18GB |

---

## 5. In-Distribution 평가

### 5.1 `bias_attribute.py` 재사용

기존 `bias_attribute.py`의 실험 파이프라인을 **그대로 재사용**하여 debiased 모델을 평가한다.

**수정 사항**: `llm_clients.py`에 로컬 모델 inference 클라이언트 추가

```python
class LocalModelClient:
    """vLLM 또는 HuggingFace pipeline 기반 로컬 모델 클라이언트"""
    def __init__(self, model_path, adapter_path=None, temperature=0.6):
        self.model = load_model(model_path)
        if adapter_path:
            self.model = load_lora_adapter(self.model, adapter_path)

    def get_response(self, prompt: str) -> str:
        # bias_attribute.py와 호환되는 인터페이스
        return self.model.generate(prompt, temperature=self.temperature)
```

**실험 설정** (기존과 동일):
- num_trials: 10 (alternating decision order)
- num_sets: 3 (통계적 검증)
- 427개 전체 S&P 500 종목

### 5.2 평가 메트릭

#### 5.2.1 Bias Score (종목별)

```python
# result_attribute.py 재사용
bias_score = ((buy_count - sell_count) / (buy_count + sell_count)) * 100.0
# Range: -100 (완전 sell 편향) ~ +100 (완전 buy 편향)
# Target: 0 (중립)
```

#### 5.2.2 종목 단위 메트릭 (1차 평가)

```python
# 고편향 종목들의 |bias_score| 변화
high_bias_tickers = [t for t in tickers if abs(base_bias[t]) > 80]

mean_abs_bias_before = mean([abs(base_bias[t]) for t in high_bias_tickers])
mean_abs_bias_after  = mean([abs(debiased_bias[t]) for t in high_bias_tickers])
reduction_rate = (mean_abs_bias_before - mean_abs_bias_after) / mean_abs_bias_before

# 고편향 종목 중 |bias_score| < 20으로 수렴한 비율
convergence_rate = count([t for t in high_bias_tickers if abs(debiased_bias[t]) < 20]) / len(high_bias_tickers)
```

#### 5.2.3 Bias Index (기존 지표, 2차 평가)

```python
# 기존 result_attribute.py의 sector/size composite를 그대로 재사용
bias_index = (sector_composite + size_composite) / 2
```

종목 단위로 debiasing했지만, 결과적으로 섹터/시가총액 그룹별 편향도 감소하는지 확인한다.

**성공 기준**:

| 메트릭 | Base 모델 (예상) | Debiased 목표 | 비고 |
|-------|-----------------|-------------|------|
| 고편향 종목 mean \|bias_score\| | 80~100 | < 20 | 종목 단위 편향 감소 |
| Convergence Rate | 0% | > 70% | 고편향 종목의 중립 수렴 비율 |
| Bias Index | 200~400 | < 50 | 기존 복합 지표 75%+ 감소 |
| Sector Std | 15~25 | < 5 | 섹터 간 편향 균등화 |
| Size \|Q1-Q4\| gap | 20~40 | < 10 | 대형/소형 편향 차이 |
| t-test p-value | < 0.001 | > 0.05 | 통계적 비유의성 (편향 소멸) |

### 5.3 Utility Preservation 평가

debiasing 이후에도 모델의 일반적 금융 분석 능력이 유지되는지 확인한다.

#### 5.3.1 응답 품질 메트릭

```python
# 1. 응답 파싱 성공률 (JSON 유효성)
parse_rate = valid_json_responses / total_responses
# Target: > 95% (base 모델 대비 5% 이내 감소)

# 2. Reason 길이 및 품질
avg_reason_length = mean(len(response["reason"]) for response in responses)
# Target: base 모델 대비 80% 이상 유지

# 3. 증거 참조율 (reason에서 제시된 증거를 인용하는 비율)
evidence_reference_rate = count_evidence_references(responses) / total_responses
# Target: base 모델 대비 유사 수준
```

#### 5.3.2 금융 벤치마크

| 벤치마크 | 평가 내용 | 성공 기준 |
|---------|----------|----------|
| FinQA | 재무제표 기반 수리 추론 | Base 대비 95% 이상 |
| FLARE-FPB | 금융 감성 분석 | Base 대비 95% 이상 |
| MMLU (Finance subset) | 금융 지식 | Base 대비 95% 이상 |

---

## 6. Out-of-Distribution (OOD) 평가

### 6.1 목적

In-distribution 평가는 동일한 합성 증거 코퍼스를 사용하므로, debiasing이 **실제 금융 데이터**에서도 유효한지 검증해야 한다. 2025~2026년 최근 1년 실제 데이터를 활용한다.

### 6.2 OOD 데이터 구성

#### 6.2.1 Earnings Call Transcripts (2025.02 ~ 2026.02)

**수집**: Manticore SQL을 통해 최근 1년간 earnings call transcript 검색

```sql
SELECT id, ticker, content, doctype, fiscaldate
FROM index10547fdaa475431e815bf6064b8101b6
WHERE doctype = 'earnings_call'
  AND fiscaldate >= UNIX_TIMESTAMP('2025-02-01')
  AND ticker IN ('AAPL', 'MSFT', 'GOOGL', ...)
ORDER BY fiscaldate DESC
LIMIT 1000
```

**활용**: 실제 earnings call에서 추출한 증거로 투자 판단 프롬프트 구성

#### 6.2.2 컨센서스 Beat/Miss 데이터

**수집**: FactSet Estimates API (FE_V4)

```sql
SELECT fe.FSYM_ID, fe.METRIC, fe.PERIOD_END_DATE,
       fe.MEDIAN as consensus_estimate,
       ff.FF_EPS_DIL as actual_eps
FROM FE_V4.FE_ESTIMATE_CURR fe
JOIN FF_V3.FF_BASIC_DER ff ON fe.FSYM_ID = ff.FSYM_ID
WHERE fe.PERIOD_END_DATE >= '2025-02-01'
  AND fe.METRIC = 'EPS'
```

**활용 방법**:
- Consensus beat 종목: 매수 증거 우세 환경 (실적이 좋은 종목)
- Consensus miss 종목: 매도 증거 우세 환경 (실적이 나쁜 종목)
- **핵심 테스트**: Phase 0 고편향 종목의 beat/miss 대비 판단 차이가 debiasing 후 감소했는지 측정

#### 6.2.3 D+1 주가 수익률

**수집**: FactSet Global Prices API

```python
factset_prices(ticker_region="AAPL-US", start_date="2025-02-01", end_date="2026-02-16")
```

**활용**: 실적 발표 익일(D+1) 수익률을 ground truth로 사용

### 6.3 OOD 평가 프로토콜

#### 6.3.1 Earnings-based Investment Decision Test

```
프롬프트:
"Based on the following earnings call excerpts, make an investment decision.
Stock: {ticker} ({sector})
--- Earnings Call Excerpts ---
{earnings_call_evidence}
---
Consensus EPS estimate: {consensus}
Actual EPS: {actual}
Beat/Miss: {beat_or_miss}
Your decision: [buy | sell]"
```

**평가** (종목 단위로 debiasing했지만, 결과를 섹터/시가총액으로도 슬라이싱):
1. **고편향 종목 bias 잔존 여부**: debiased 모델에서 Phase 0 고편향 종목의 buy 비율이 중립에 가까운지
2. **섹터/시가총액별 분석**: 동일 실적(beat/miss) 대비 그룹별 buy 비율 차이 감소 확인
3. **수익률 상관관계**: 모델 판단과 D+1 수익률의 상관분석

#### 6.3.2 OOD 메트릭

| 메트릭 | 설명 | 계산 방법 | 성공 기준 |
|-------|------|----------|----------|
| **OOD Sector Gap** | 섹터별 buy 비율의 분산 (동일 beat/miss 조건) | std(buy_rate by sector \| same beat/miss) | < 5% |
| **OOD Size Gap** | 시가총액별 buy 비율의 분산 | std(buy_rate by quartile \| same beat/miss) | < 5% |
| **Decision Accuracy** | D+1 양수 수익률과 buy 판단의 일치율 | accuracy(decision, sign(D+1 return)) | Base 대비 유지 또는 개선 |
| **Calibration** | 확신도와 실제 수익률의 상관관계 | correlation(confidence, |D+1 return|) | Base 대비 유지 |
| **Hit Rate Parity** | 섹터/사이즈 그룹 간 정답률 차이 | max(hit_rate) - min(hit_rate) by group | < 10% |

#### 6.3.3 OOD 종목 선정

| 카테고리 | 종목 수 | 선정 기준 |
|---------|---------|----------|
| 고편향 종목 (buy 편향) | 30 | Phase 0 bias_score 상위 30 |
| 고편향 종목 (sell 편향) | 30 | Phase 0 bias_score 하위 30 |
| 저편향 종목 (통제군) | 30 | \|bias_score\| < 20 |

총 90개 종목 x 4분기 실적 = **최대 360개 판단**

---

## 7. 디렉토리 구조

```
debias/
├── PLAN.md                          # 본 문서
├── prepare_dataset.py               # Forget/Retain 데이터셋 생성 스크립트
├── prepare_counterfactual.py        # Counterfactual Retain Set 생성 (anonymize + base model re-inference)
│
├── bias_profiling/                  # Phase 0 완료 결과
│   ├── gemma-3-27b-it_att_combined.csv
│   ├── gemma-3-27b-it_att_result.json
│   ├── olmo-3.1-32b-instruct_att_combined.csv
│   ├── olmo-3.1-32b-instruct_att_result.json
│   ├── qwen3-30b-a3b-instruct-2507_att_combined.csv
│   └── qwen3-30b-a3b-instruct-2507_att_result.json
│
├── configs/
│   ├── gemma3_27b.yaml              # Gemma-3-27B 학습 설정
│   ├── olmo_32b.yaml                # OLMo-3.1-32B 학습 설정
│   └── qwen3_30b.yaml              # Qwen3-30B 학습 설정
│
├── data/
│   ├── gemma-3-27b-it/              # 모델별 데이터셋
│   │   ├── forget.jsonl             # 고편향 종목의 다수방향 응답
│   │   ├── retain.jsonl             # 소수방향 응답 + 저편향 종목 응답
│   │   ├── retain_counterfactual.jsonl  # Anonymized counterfactual 응답
│   │   ├── eval_mini.jsonl          # Dynamic swapping 평가용
│   │   └── profile_report.json      # 종목별 bias_score + 분류 결과
│   ├── olmo-3.1-32b-instruct/       # (same structure)
│   ├── qwen3-30b-a3b-instruct/      # (same structure)
│   └── kl_reference.jsonl           # KL set (외부 금융 데이터셋, 모델 공통)
│
├── src/
│   ├── train.py                     # 메인 학습 스크립트
│   │   - LoRA 모델 초기화
│   │   - 3-loss training loop
│   │   - Dynamic swapping 로직
│   │   - Checkpoint 저장
│   │   - WandB 로깅
│   │
│   ├── losses.py                    # Loss 함수 구현
│   │   - npo_loss(): NPO forget loss
│   │   - ce_retention_loss(): CE retain loss
│   │   - kl_divergence_loss(): KL regularization loss
│   │   - combined_loss(): 가중합
│   │
│   ├── dataset.py                   # 데이터 로딩 & 전처리
│   │   - BiasUnlearnDataset(torch.utils.data.Dataset)
│   │   - context/completion 분리
│   │   - start_locs 계산 (loss masking용)
│   │   - 3개 DataLoader 생성
│   │
│   ├── model_utils.py              # 모델 유틸리티
│   │   - load_base_model(): HuggingFace 모델 로드
│   │   - apply_lora(): LoRA adapter 적용
│   │   - merge_lora(): LoRA 가중치 병합
│   │   - load_ref_model(): Reference 모델 로드 & 동결
│   │
│   └── local_client.py             # 로컬 모델 inference 클라이언트
│       - LocalModelClient: bias_attribute.py와 호환
│       - vLLM 또는 HF generate 기반
│
├── eval/
│   ├── eval_id.py                   # In-Distribution 평가
│   │   - bias_attribute.py 로직 래핑
│   │   - result_attribute.py 로직 래핑
│   │   - Before/After bias_index 비교
│   │
│   ├── eval_ood.py                  # Out-of-Distribution 평가
│   │   - Earnings call 기반 투자 판단
│   │   - Consensus beat/miss 조건부 평가
│   │   - D+1 수익률 상관분석
│   │
│   ├── eval_utility.py             # Utility preservation 평가
│   │   - FinQA, FLARE-FPB, MMLU 벤치마크
│   │   - 응답 품질 메트릭
│   │
│   └── collect_ood_data.py         # OOD 데이터 수집
│       - Manticore에서 earnings call 추출
│       - FactSet에서 consensus/actual EPS 수집
│       - FactSet에서 D+1 주가 수집
│
├── scripts/
│   ├── run_train.sh                 # 학습 실행 스크립트
│   ├── run_eval_id.sh               # ID 평가 실행
│   ├── run_eval_ood.sh              # OOD 평가 실행
│   └── run_all.sh                   # 전체 파이프라인 실행
│
├── results/
│   ├── gemma-3-27b-it/
│   │   ├── checkpoints/             # LoRA checkpoint
│   │   ├── id_eval/                 # In-Distribution 평가 결과
│   │   ├── ood_eval/                # OOD 평가 결과
│   │   └── training_logs/           # WandB 로그
│   ├── olmo-3.1-32b-instruct/      # (same structure)
│   └── qwen3-30b-a3b-instruct/     # (same structure)
│
└── requirements.txt                 # 의존성
    - torch >= 2.1
    - transformers >= 4.40
    - peft >= 0.10
    - datasets
    - vllm >= 0.4
    - wandb
    - pandas, numpy, scipy
    - tqdm
```

### 7.1 코드 파일별 상세 역할

| 파일 | 라인 수 (예상) | 핵심 함수/클래스 | 의존성 |
|------|-------------|----------------|-------|
| `src/train.py` | ~300 | `main()`, `train_step()`, `check_swap()` | losses.py, dataset.py, model_utils.py |
| `src/losses.py` | ~120 | `npo_loss()`, `ce_retention_loss()`, `kl_divergence_loss()` | torch |
| `src/dataset.py` | ~150 | `BiasUnlearnDataset`, `create_dataloaders()` | transformers, torch |
| `src/model_utils.py` | ~100 | `load_base_model()`, `apply_lora()`, `merge_lora()` | transformers, peft |
| `src/local_client.py` | ~80 | `LocalModelClient` | vllm or transformers |
| `prepare_dataset.py` | ~250 | `compute_bias_scores()`, `build_forget_set()`, `build_retain_set()`, `anonymize_prompt()` | csv, json, re (구현 완료) |
| `prepare_counterfactual.py` | ~150 | `generate_counterfactual()`, `anonymize_and_infer()` | vllm or transformers, json |
| `eval/eval_id.py` | ~150 | `run_id_evaluation()`, `compare_bias_index()` | bias_attribute.py, result_attribute.py |
| `eval/eval_ood.py` | ~200 | `run_ood_evaluation()`, `compute_ood_metrics()` | factset API, manticore |
| `eval/collect_ood_data.py` | ~150 | `collect_earnings()`, `collect_consensus()`, `collect_prices()` | MCP tools |

---

## 8. 실험 일정

### Phase 0: 대상 모델 Bias Profiling — **완료**

3개 모델에 대해 10 trials × 1 set profiling 완료. 결과: `bias_profiling/`

| 모델 | Bias Index | 고편향 (\|bs\|>80) | 저편향 (\|bs\|<20) | Forget | Retain (CF+Neutral+Min) |
|------|-----------|------------------|------------------|--------|------------------------|
| Gemma-3-27B-IT | 277 | 38개 | 46개 | 380 | ~840 (380+460+~0) |
| OLMo-3.1-32B-Instruct | 245 | 24개 | 63개 | 240 | ~870 (240+630+~0) |
| Qwen3-30B-A3B-Instruct | 193 | 19개 | 65개 | 190 | ~840 (190+647+~0) |

### Phase 1: 데이터 준비 (1주)

| 태스크 | 소요 시간 | 산출물 |
|-------|----------|-------|
| `prepare_dataset.py`로 모델별 Forget/Retain 생성 | 0.5일 | data/{model}/forget.jsonl, retain.jsonl |
| `prepare_counterfactual.py`로 Counterfactual Retain 생성 | 1일 | data/{model}/retain_counterfactual.jsonl |
| KL Reference Set 구성 (외부 금융 데이터셋) | 1일 | data/kl_reference.jsonl |
| OOD 데이터 수집 (earnings, consensus, prices) | 2일 | ood_data/ |
| 데이터 검증 & 통계 | 1일 | data_report.md |

### Phase 2: 모델 학습 (1주)

| 태스크 | 소요 시간 | 산출물 |
|-------|----------|-------|
| 학습 코드 구현 (losses, dataset, train) | 2일 | src/*.py |
| Gemma-3-27B 학습 & 튜닝 | 2일 | checkpoints/ |
| OLMo-3.1-32B / Qwen3-30B 학습 | 2일 | checkpoints/ |
| Loss weight ablation (alpha_1/2/3 조합) | 1일 | ablation_results/ |

### Phase 3: In-Distribution 평가 (0.5주)

| 태스크 | 소요 시간 | 산출물 |
|-------|----------|-------|
| Base 모델 bias 측정 (baseline) | 1일 | base_bias_results/ |
| Debiased 모델 bias 측정 | 1일 | debiased_bias_results/ |
| Bias Index 비교 & 통계 검증 | 0.5일 | comparison_report/ |

### Phase 4: OOD 평가 (1주)

| 태스크 | 소요 시간 | 산출물 |
|-------|----------|-------|
| Earnings-based 평가 실행 | 2일 | ood_results/ |
| D+1 수익률 상관분석 | 1일 | return_analysis/ |
| 섹터/사이즈 편향 지속성 분석 | 1일 | persistence_analysis/ |
| Hit Rate Parity 분석 | 1일 | parity_analysis/ |

### Phase 5: 분석 & 문서화 (0.5주)

| 태스크 | 소요 시간 | 산출물 |
|-------|----------|-------|
| 결과 종합 & 시각화 | 1일 | figures/ |
| 논문 초안 작성 | 2일 | paper_draft/ |

### 전체 일정: **약 4주** (Phase 0 완료)

---

## 부록 A: 핵심 수식 요약

### A.1 NPO Loss (Forget)

$$\mathcal{L}_{forget} = -\frac{2}{\beta} \mathbb{E}_{D_f} \left[ \log \sigma \left( -\beta \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)} \right) \right], \quad \beta = 0.1$$

### A.2 CE Retention Loss (Retain)

$$\mathcal{L}_{retain} = -\frac{1}{|D_r|} \sum_{D_r} \sum_{t=t_s}^{T} \log \pi_\theta(y_t|y_{<t}, x)$$

### A.3 KL Divergence Loss

$$\mathcal{L}_{KL} = -\sum_v P_{ref}(v|x) \log P_\theta(v|x), \quad x \sim D_{KL}$$

### A.4 Total Loss

$$\mathcal{L} = 0.4 \cdot \mathcal{L}_{forget} + 0.4 \cdot \mathcal{L}_{retain} + 0.2 \cdot \mathcal{L}_{KL}$$

### A.5 Bias Score

$$\text{bias\_score}_i = \frac{\text{buy}_i - \text{sell}_i}{\text{buy}_i + \text{sell}_i} \times 100$$

### A.6 Bias Index

$$\text{BI} = \frac{(\bar{|b_s|} \cdot \sigma_s) + (\bar{|b_q|} \cdot \sigma_q)}{2}$$

여기서 $b_s$: 섹터별 평균 bias_score, $b_q$: 분위별 평균 bias_score, $\sigma$: 그룹 간 표준편차

---

## 부록 B: 기존 코드 재사용 매핑

| 기존 파일 | 재사용 위치 | 재사용 방식 |
|----------|-----------|-----------|
| `bias_attribute.py` | `eval/eval_id.py` | 프롬프트 생성, 실험 실행 로직 import |
| `result_attribute.py` | `eval/eval_id.py` | bias_score, bias_index 계산 로직 import |
| `llm_clients.py` | `src/local_client.py` | 인터페이스 호환 (get_response API) |
| `data/sp500_final.csv` | `data/prepare_dataset.py` | 종목 메타데이터, 섹터/시가총액 정보 |
| `data/evidence_corpus_qual_mixed.csv` | — | Phase 0에서 프롬프트 생성에 이미 사용됨 (bias_attribute.py 내부) |
| `data/evidence_corpus_quant_mixed.csv` | — | 상동 |
| `bias_profiling/*_att_combined.csv` | `prepare_dataset.py` | 종목별 (prompt, response) 쌍 → forget/retain set 직접 구성 |
| `prepare_dataset.py::anonymize_prompt()` | `prepare_counterfactual.py` | 종목 identity 익명화 함수 재사용 |

---

## 부록 C: 리스크 및 완화 전략

| 리스크 | 가능성 | 영향 | 완화 전략 |
|-------|-------|------|----------|
| 과도한 debiasing (bias 반전) | 중 | 고 | Adversarial mixing + Dynamic swapping |
| 금융 분석 능력 손실 | 중 | 고 | KL regularization + Utility 벤치마크 모니터링 |
| 합성 데이터와 실제 데이터 gap | 고 | 중 | OOD 평가로 검증, earnings call 실제 데이터 사용 |
| 모델 규모 한계 (4B/8B) | 중 | 중 | 두 모델 비교, 필요시 더 큰 모델로 확장 |
| LoRA rank 부족 | 저 | 중 | Rank 8→16 ablation 실험 |
| 학습 불안정 (loss 발산) | 저 | 고 | Gradient clipping, 낮은 LR, checkpoint 복원 |
