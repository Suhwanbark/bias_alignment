# Local LLM Bias Experiment

로컬 GPU에서 vLLM을 사용하여 LLM의 금융 투자 편향을 측정하는 실험입니다.

> 이 코드는 [llm-bias-in-finance](https://github.com/your-repo/llm-bias-in-finance) 프로젝트의 로컬 인퍼런스 버전입니다.

## 목차

- [요구 사항](#요구-사항)
- [설치](#설치)
- [모델 다운로드](#모델-다운로드)
- [vLLM 서버 실행](#vllm-서버-실행)
- [실험 실행](#실험-실행)
- [설정 변경](#설정-변경)
- [출력 파일](#출력-파일)
- [트러블슈팅](#트러블슈팅)

---

## 요구 사항

### 하드웨어
- **GPU**: NVIDIA GPU with CUDA support
  - 최소: 1x GPU with 24GB+ VRAM (양자화 필요)
  - 권장: 4x GPU with 40GB+ VRAM (Qwen3-30B full precision)
- **RAM**: 64GB+ 권장
- **Storage**: 모델 캐시용 100GB+ 여유 공간

### 소프트웨어
- Python 3.10+
- CUDA 12.1+ (또는 11.8)
- Linux (Ubuntu 20.04+ 권장)

### 테스트된 환경
```
Python: 3.10.19
PyTorch: 2.9.1+cu128
CUDA: 12.8
vLLM: 0.14.1
GPU: 4x NVIDIA H200 (143GB each)
```

---

## 설치

### 빠른 설치 (쿠버네티스/Docker 환경)

```bash
bash env/install.sh
```

### Conda 환경 사용

```bash
# 환경 생성
conda env create -f env/environment.yaml

# 환경 활성화
conda activate llm-bias-local
```

### pip 설치 (수동)

```bash
# 1. 가상 환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Linux/Mac

# 2. 패키지 설치
pip install -r requirements.txt
```

### vLLM 설치

vLLM은 CUDA 버전에 따라 설치 방법이 다릅니다.

**CUDA 12.1+ (권장)**
```bash
pip install vllm
```

**CUDA 11.8**
```bash
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu118
```

**설치 확인**
```bash
python -c "import vllm; print(vllm.__version__)"
```

> 자세한 설치 가이드: https://docs.vllm.ai/en/latest/getting_started/installation.html

---

## 모델 다운로드

### 자동 다운로드 (권장)

vLLM 서버 실행 시 자동으로 HuggingFace에서 모델을 다운로드합니다.

```bash
# 첫 실행 시 자동 다운로드 (~60GB)
vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 --port 8000
```

### 수동 다운로드

```bash
# HuggingFace CLI 설치
pip install huggingface_hub

# 모델 다운로드
huggingface-cli download Qwen/Qwen3-30B-A3B-Instruct-2507
```

### 모델 캐시 위치

**이 프로젝트의 모델 위치:**
```
/data/llm-bias-in-finance/models/models--Qwen--Qwen3-30B-A3B-Instruct-2507
```

`env/install.sh` 실행 시 자동으로 `~/.cache/huggingface/hub/`에 심볼릭 링크가 생성됩니다.

**수동으로 심볼릭 링크 생성:**
```bash
mkdir -p ~/.cache/huggingface/hub
ln -sf /data/llm-bias-in-finance/models/models--Qwen--Qwen3-30B-A3B-Instruct-2507 \
    ~/.cache/huggingface/hub/
```

### 다른 모델 사용

다른 HuggingFace 모델도 사용 가능합니다:

```bash
# Llama 3
vllm serve meta-llama/Llama-3.1-70B-Instruct --tensor-parallel-size 4

# Mistral
vllm serve mistralai/Mistral-7B-Instruct-v0.3

# Qwen2.5
vllm serve Qwen/Qwen2.5-72B-Instruct --tensor-parallel-size 4
```

---

## vLLM 서버 실행

### 기본 실행 (GPU 4개)

```bash
vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --tensor-parallel-size 4 \
    --port 8000 \
    --max-model-len 4096
```

### GPU 개수별 설정

**1 GPU (24GB+ VRAM)**
```bash
vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --port 8000 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.95
```

**2 GPUs**
```bash
vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --tensor-parallel-size 2 \
    --port 8000 \
    --max-model-len 4096
```

**4 GPUs (권장)**
```bash
vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --tensor-parallel-size 4 \
    --port 8000 \
    --max-model-len 4096
```

### 백그라운드 실행

```bash
# nohup 사용
nohup vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --tensor-parallel-size 4 \
    --port 8000 \
    --max-model-len 4096 \
    > vllm.log 2>&1 &

# 로그 확인
tail -f vllm.log
```

### 서버 상태 확인

```bash
# Health check
curl http://localhost:8000/health

# 사용 가능한 모델 확인
curl http://localhost:8000/v1/models
```

---

## 실험 실행

### 빠른 시작

```bash
# 1. vLLM 서버 실행 (터미널 1)
vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --tensor-parallel-size 4 \
    --port 8000 \
    --max-model-len 4096

# 2. 실험 실행 (터미널 2)
cd /path/to/local
bash run.sh
```

### 개별 스크립트 실행

```bash
# 실험 실행
python bias_attribute.py \
    --model-id "Qwen/Qwen3-30B-A3B-Instruct-2507" \
    --temperature 0.6 \
    --max-tokens 1024 \
    --num-sets 2 \
    --num-trials 10 \
    --max-workers 200 \
    --vllm-url "http://localhost:8000/v1" \
    --output-dir "./result"

# 결과 집계
python result_attribute.py \
    --model-id "Qwen/Qwen3-30B-A3B-Instruct-2507" \
    --output-dir "./result"
```

### 간단한 테스트

```bash
python -c "
from llm_clients import VLLMClient

client = VLLMClient(
    model_id='Qwen/Qwen3-30B-A3B-Instruct-2507',
    temperature=0.6,
)

response = client.get_response('What is 2+2?')
print(response)
"
```

---

## 설정 변경

### run.sh 설정

```bash
# run.sh 파일 내 설정 변경
MODEL_ID="Qwen/Qwen3-30B-A3B-Instruct-2507"  # 모델 ID
TEMPERATURE=0.6                              # 샘플링 온도
MAX_TOKENS=1024                              # 최대 출력 토큰
NUM_SETS=2                                   # 실험 반복 횟수
NUM_TRIALS=10                                # 주식당 시도 횟수
MAX_WORKERS=200                              # 동시 요청 수
```

### Evidence 균형

실험은 각 주식에 대해 균형 잡힌 evidence를 제공합니다:
- **2개 Buy evidence** (qualitative + quantitative)
- **2개 Sell evidence** (qualitative + quantitative)

### Trial 구성

각 주식당 10번의 trial:
- 5번: `[buy | sell]` 순서
- 5번: `[sell | buy]` 순서

이를 통해 선택지 순서에 의한 편향을 제거합니다.

---

## 출력 파일

```
result/
├── {model}_att_set_1.csv      # Set 1 원시 결과
├── {model}_att_set_2.csv      # Set 2 원시 결과
├── {model}_att_combined.csv   # 통합 결과 (집계 후 생성)
├── {model}_att_result.json    # Bias index 및 통계
└── {model}_att_metrics.json   # 성능 메트릭 (토큰, 시간 등)
```

### 결과 JSON 예시

```json
{
    "bias_index": 272,
    "sector_stats": {
        "Technology": {"bias_mean": 47.0, "bias_std": 2.0},
        "Healthcare": {"bias_mean": 32.0, "bias_std": 3.0}
    },
    "size_stats": {
        "Q1": {"bias_mean": 39.0, "bias_std": 3.0},
        "Q4": {"bias_mean": 25.0, "bias_std": 4.0}
    },
    "t_test_results": {
        "sector_comparison": {
            "high_bias_group": "Technology",
            "low_bias_group": "Utilities",
            "p_value": 0.0023
        }
    }
}
```

---

## 트러블슈팅

### vLLM 서버가 시작되지 않음

**CUDA 메모리 부족**
```bash
# GPU 메모리 사용률 조정
vllm serve ... --gpu-memory-utilization 0.9

# 또는 max-model-len 줄이기
vllm serve ... --max-model-len 2048
```

**포트 사용 중**
```bash
# 다른 포트 사용
vllm serve ... --port 8001

# 실험 실행 시 URL 변경
python bias_attribute.py --vllm-url "http://localhost:8001/v1"
```

### 연결 오류

```bash
# 서버 상태 확인
curl http://localhost:8000/health

# 서버 로그 확인
tail -f vllm.log
```

### 모델 다운로드 실패

```bash
# HuggingFace 토큰 설정 (private 모델의 경우)
export HF_TOKEN=your_token_here

# 또는 로그인
huggingface-cli login
```

### Out of Memory (OOM)

```bash
# max-workers 줄이기
python bias_attribute.py --max-workers 50

# 또는 max-model-len 줄이기
vllm serve ... --max-model-len 2048
```

---

## 파일 구조

```
llm-bias-in-finance/
├── local/                    # 로컬 인퍼런스 코드
│   ├── llm_clients.py            # vLLM OpenAI-compatible 클라이언트
│   ├── bias_attribute.py         # Attribute bias 실험 실행
│   ├── result_attribute.py       # 결과 집계 및 bias index 계산
│   ├── run.sh                    # 실험 실행 스크립트
│   ├── requirements.txt          # Python 패키지 의존성
│   ├── README.md                 # 이 파일
│   └── env/                      # 환경 설정 파일들
│       ├── environment.yaml          # Conda 환경 정의
│       ├── pip-requirements.txt      # 정확한 버전 pip 패키지
│       ├── setup.sh                  # 자동 설치 스크립트
│       ├── install.sh                # 빠른 설치 스크립트
│       └── README.md                 # 환경 설정 가이드
├── models/                   # HuggingFace 모델 캐시 (57GB)
│   └── models--Qwen--Qwen3-30B-A3B-Instruct-2507/
└── data/                     # 실험 데이터
    ├── sp500_final.csv
    ├── evidence_corpus_qual_mixed.csv
    └── evidence_corpus_quant_mixed.csv
```

---

## 기존 OpenRouter 버전과의 차이

| 항목 | OpenRouter | Local (vLLM) |
|------|------------|--------------|
| API | OpenRouter REST API | vLLM OpenAI-compatible |
| 비용 | 토큰당 과금 | 무료 (자체 GPU) |
| 모델 | 100+ 모델 | HuggingFace 모델 |
| 속도 | 네트워크 의존 | 로컬 GPU 속도 |
| 병렬처리 | Rate limit 있음 | GPU 성능에 의존 |

---

## 라이선스

MIT License - [llm-bias-in-finance](https://github.com/your-repo/llm-bias-in-finance)
