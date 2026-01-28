# Models Directory

이 폴더에는 HuggingFace 모델 캐시가 저장됩니다.

## 모델 다운로드

모델은 Git에 포함되지 않습니다. 다음 방법으로 다운로드하세요:

### 방법 1: vLLM 서버 실행 시 자동 다운로드

```bash
vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 --port 8000
```

첫 실행 시 자동으로 `~/.cache/huggingface/hub/`에 다운로드됩니다.

### 방법 2: 수동 다운로드

```bash
pip install huggingface_hub
huggingface-cli download Qwen/Qwen3-30B-A3B-Instruct-2507
```

### 방법 3: 이 폴더로 이동

다운로드 후 이 폴더로 이동하고 심볼릭 링크 생성:

```bash
mv ~/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507 ./models/

mkdir -p ~/.cache/huggingface/hub
ln -sf $(pwd)/models/models--Qwen--Qwen3-30B-A3B-Instruct-2507 ~/.cache/huggingface/hub/
```

## 모델 정보

| 모델 | 크기 | 파라미터 |
|------|------|----------|
| Qwen/Qwen3-30B-A3B-Instruct-2507 | ~57GB | 30.5B total, 3.3B activated (MoE) |
