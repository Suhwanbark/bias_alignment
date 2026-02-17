# data/

S&P 500 주식 데이터 및 실험용 Evidence Corpus를 저장하는 폴더.

## 주요 파일

| 파일 | 설명 |
|------|------|
| `sp500_final.csv` | S&P 500 전체 티커 메타데이터 (ticker, name, sector, marketcap) |
| `evidence_corpus_qual_mixed.csv` | 정성적(qualitative) 증거 - 긍정/부정 혼합 |
| `evidence_corpus_quant_mixed.csv` | 정량적(quantitative) 증거 - 긍정/부정 혼합 |
| `evidence_corpus_view.csv` | 애널리스트 뷰 기반 증거 |

## 하위 폴더

| 폴더 | 설명 |
|------|------|
| `gemini/` | Gemini 모델 실험용 데이터 |
| `kimi/` | Kimi 모델 실험용 데이터 |
| `mini/` | 소규모 테스트용 데이터 |

## 사용처

- `bias_attribute.py`: S&P 500 티커별 bias 측정 시 사용
- `bias_strategy.py`: 투자 전략 bias 측정 시 사용
- Evidence corpus는 LLM에게 투자 판단 프롬프트 생성 시 활용
