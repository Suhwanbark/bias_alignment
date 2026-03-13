# Handoff — 2026-03-13

## Task
Forget set 재설계: SimNPO/NPO의 forget set이 eval과 동일한 evidence corpus + prompt 포맷을 공유하는 data leakage 문제를 해결하기 위한 새로운 forget set 구성 방안 논의.

## Completed
- 프로젝트 폴더 구조 전체 파악
- **Data leakage 분석 완료**: eval(`bias_attribute.py`)과 훈련 데이터가 동일한 `evidence_corpus_{qual,quant}_mixed.csv` + 동일 prompt 포맷 사용 확인
  - SFT: 427/427 종목 겹침 (100%) — 가장 심각
  - DPO: 12/427 종목 + 다른 프롬프트 포맷 — 상대적으로 양호
  - NPO/SimNPO: 21~34/427 종목 + 같은 evidence corpus — 중간
- **Forget set 재구성 아이디어 7가지 제안**:
  1. Negative Scenario Generation (부정 시나리오에서도 buy 고집 패턴 forget)
  2. Multi-Format Forget (뉴스/토론/애널리스트 등 다양한 포맷)
  3. Ticker-Sentiment Association Only (evidence 없이 "NVDA→buy" 매핑만)
  4. Sector-Level Abstraction ("large-cap tech=buy" 패턴)
  5. Counterfactual Pairing (같은 evidence를 다른 티커에 붙여서 이름 의존성 타겟)
  6. Chain-of-Thought Bias Forget (reasoning 패턴 unlearn)
  7. Hybrid (forget=다양한 포맷, retain=eval 포맷)
- **선행연구 조사 완료** — TOFU, WMDP, MUSE, SimNPO, FLAT, Noisy Forget Sets, GUARD 벤치마크별 forget set 구성 방식 및 실제 예시 정리
- **Qwen 4B forget set 현황 파악**: `npo/data/qwen4b/forget.jsonl` — 323샘플, 34티커, SELL 94.7% / BUY 5.3%

## In Progress / Remaining
- 유저가 아직 어떤 forget set 구성 방식을 선택할지 결정 안 함
- 선택 후: forget set 생성 파이프라인 구현 필요
- 추천 조합: Multi-Format + Minimal + Negative Scenario (WMDP 스타일 proxy corpus)
- Noisy Forget Sets 연구 결과: 포맷이 달라도 semantic cue만 보존되면 unlearning 작동 → 다양한 포맷 접근 정당화

## Key Decisions
- Eval 포맷(structured evidence + JSON decision)은 그대로 유지
- Forget set만 다른 포맷으로 재구성하여 leakage 제거
- WMDP 스타일(proxy corpus: forget ≠ eval 포맷)이 가장 참고할 만함

## Referenced Files
- `eval/bias_attribute.py` — 메인 평가 드라이버 (evidence corpus 로딩)
- `eval/config.py` — 평가 상수
- `data/evidence_corpus_{qual,quant}_mixed.csv` — 공유 evidence corpus (leakage 원인)
- `data/sp500_final.csv` — 427종목 메타데이터
- `npo/data/qwen4b/forget.jsonl` — Qwen 4B forget set (323샘플, 34티커)
- `npo/data/qwen3-4b-instruct-2507-extreme/forget.jsonl` — extreme 변형 (80샘플, 8티커, 전부 BUY)
- `npo/data/forget_v2.jsonl` — 범용 forget set (210샘플, 21티커)
- `simple_npo/src/train.py`, `losses.py`, `dataset.py` — SimNPO 훈련 코드

## Notes
- Node.js symlink 설정 완료: `/usr/local/bin/node` → nvm node (다른 터미널 sh 세션에서 claude 사용 가능)
- 선행연구 핵심 인사이트: "Noisy Forget Sets" 논문에서 forget set 표면 형식이 달라도 core semantic signal만 보존되면 unlearning 효과 유지됨 → 우리가 eval과 다른 포맷으로 forget set을 구성해도 작동할 근거
