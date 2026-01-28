# DPO Debiasing PoC 실험

극단적 bias를 가진 ticker들에 대해 균형잡힌 evidence 기반 DPO 학습으로 bias 교정

## 대상 모델 및 Ticker

### NVIDIA Nemotron - 22개 (buy_rate=0%, SELL 편향)
```
ECL, IFF, NWSA, META, CCL, PEP, DG, PFG, COF, MTB, GL, HBAN,
BAX, HCA, GWW, PH, CHRW, ITW, DLR, WY, EA, D
```

### Qwen3 - 12개 (buy_rate>=90%, BUY 편향)
```
LMT, SO, PYPL, PPL, WMT, VRSN, CVX, EW, JNJ, NVDA, PWR, TSN
```

## 실행 방법

### Step 1: 이벤트 생성
각 ticker에 대해 긍정/부정 실제 이벤트 추출

```bash
# NVIDIA 대상 ticker 이벤트 생성
python generate_events.py --target nvidia --output data/events_nvidia.json

# Qwen 대상 ticker 이벤트 생성
python generate_events.py --target qwen --output data/events_qwen.json
```

**옵션:**
- `--model-id`: 생성용 모델 ID (기본값: gpt-oss-20b)
- `--vllm-url`: vLLM 서버 URL (기본값: http://localhost:8000/v1)
- `--num-events`: ticker당 긍정/부정 이벤트 수 (기본값: 10)
- `--max-workers`: 동시 워커 수 (기본값: 10)

### Step 2: DPO 데이터셋 생성
이벤트 조합으로 DPO 학습 데이터 생성

```bash
# NVIDIA용 DPO 데이터셋 생성
python generate_dpo_dataset.py \
    --events data/events_nvidia.json \
    --num-samples 1000 \
    --output data/dpo_nvidia.jsonl

# Qwen용 DPO 데이터셋 생성
python generate_dpo_dataset.py \
    --events data/events_qwen.json \
    --num-samples 1000 \
    --output data/dpo_qwen.jsonl
```

**옵션:**
- `--model-id`: 생성용 모델 ID (기본값: gpt-oss-20b)
- `--vllm-url`: vLLM 서버 URL (기본값: http://localhost:8000/v1)
- `--num-samples`: 총 생성할 샘플 수 (기본값: 1000)
- `--max-workers`: 동시 워커 수 (기본값: 20)

### Step 3: DPO 훈련 (별도)
trl 또는 axolotl 사용

### Step 4: Bias 재측정
기존 `bias_attribute.py`로 훈련 후 모델 재평가

## DPO 데이터 형식

```json
{
  "prompt": "META. Evidence: [+] 광고매출 23% 증가 [-] EU 규제 강화. Should you buy or sell?",
  "chosen": {"decision": "buy", "reason": "While regulatory concerns exist..."},
  "rejected": {"decision": "sell", "reason": "The regulatory headwinds pose..."},
  "ticker": "META",
  "company_name": "Meta Platforms, Inc.",
  "evidence_positive": ["광고매출 23% 증가"],
  "evidence_negative": ["EU 규제 강화"]
}
```

## 폴더 구조

```
debias/
├── generate_events.py           # Step 1: 실제 이벤트 추출
├── generate_dpo_dataset.py      # Step 2: DPO 데이터셋 생성
├── config.py                    # 설정 (ticker 목록, 프롬프트 등)
├── llm_client.py                # vLLM 클라이언트
├── data/
│   ├── events_nvidia.json       # NVIDIA 대상 ticker 이벤트
│   ├── events_qwen.json         # Qwen 대상 ticker 이벤트
│   ├── dpo_nvidia.jsonl         # 최종 DPO 데이터
│   └── dpo_qwen.jsonl
└── README.md                    # 사용법 (이 파일)
```

## Bias 교정 방향

| 모델 | 현재 Bias | chosen | rejected |
|------|-----------|--------|----------|
| NVIDIA Nemotron | SELL (buy_rate=0%) | BUY | SELL |
| Qwen3 | BUY (buy_rate>=90%) | SELL | BUY |
