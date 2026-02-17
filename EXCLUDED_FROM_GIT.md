# Excluded from Git

The following directories and files are excluded from version control due to size constraints.

## Excluded Directories

| Directory | Approx Size | Reason |
|-----------|-------------|--------|
| `models/` | 57 GB | Base model weights (Qwen3-30B-A3B-Instruct-2507) |
| `debias/models/` | 115 GB | LoRA adapter checkpoints from DPO/SFT training |
| `legacy/` | 199 GB | Archived models (Nemotron, GLM, gpt-oss) and embeddings |
| `LLaMA-Factory/` | 20 MB | Separate git repo; install via `pip install -e LLaMA-Factory/` (v0.9.3) |

## Excluded File Patterns

- `*.pt`, `*.bin`, `*.safetensors`, `*.gguf` — Model weight files
- `*.npy` — NumPy array files (embeddings)
- `npo.zip` — Archive of initial NPO implementation

## How to Reproduce

1. **Base model**: Download from HuggingFace
   ```bash
   huggingface-cli download Qwen/Qwen3-30B-A3B-Instruct-2507 --local-dir models/Qwen3-30B-A3B-Instruct-2507
   ```

2. **LLaMA-Factory**: Clone and install
   ```bash
   git clone https://github.com/hiyouga/LLaMA-Factory.git
   cd LLaMA-Factory && git checkout v0.9.3
   pip install -e .
   ```

3. **Training checkpoints**: Re-run training with configs in `debias/llamafactory/configs/`
