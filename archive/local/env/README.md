# Environment Setup

쿠버네티스 등 환경이 초기화될 수 있는 상황에서 빠르게 환경을 재구성하기 위한 파일들입니다.

## 파일 설명

| 파일 | 설명 |
|------|------|
| `environment.yaml` | Conda 환경 정의 파일 |
| `pip-requirements.txt` | 정확한 버전이 명시된 pip 패키지 목록 |
| `setup.sh` | Conda/pip 자동 감지 설치 스크립트 |
| `install.sh` | pip만 사용하는 빠른 설치 스크립트 |

## 빠른 설치 (권장)

```bash
# 쿠버네티스/Docker 환경에서
bash env/install.sh
```

## Conda 환경 사용

```bash
# 환경 생성
conda env create -f env/environment.yaml

# 환경 활성화
conda activate llm-bias-local

# 환경 삭제 (필요시)
conda env remove -n llm-bias-local
```

## pip 직접 설치

```bash
# 정확한 버전으로 설치
pip install -r env/pip-requirements.txt

# vLLM 설치
pip install vllm
```

## 환경 검증

```bash
python -c "
import vllm
import openai
import pandas
import scipy
print('✓ All packages installed successfully')
print(f'  vLLM: {vllm.__version__}')
"
```

## 테스트된 환경

```
Python: 3.10.19
PyTorch: 2.9.1+cu128
CUDA: 12.8
vLLM: 0.14.1
GPU: 4x NVIDIA H200 (143GB each)
```
