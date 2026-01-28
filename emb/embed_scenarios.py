"""
시나리오 Embedding 스크립트

생성된 시나리오 텍스트를 sentence-transformers로 벡터화.

사용법:
    # 단일 GPU (기본)
    python embed_scenarios.py --input data/scenarios_gptoss.json --output data/emb_gptoss.npy

    # 멀티 GPU (4개)
    python embed_scenarios.py --input data/scenarios_gptoss.json --output data/emb_gptoss.npy --multi-gpu --num-gpus 4

    # 대용량 배치
    python embed_scenarios.py --input data/scenarios_gptoss.json --output data/emb_gptoss.npy --batch-size 512
"""

import argparse
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


DEFAULT_MODEL = "all-MiniLM-L6-v2"  # 빠르고 가벼움
# 대안: "BAAI/bge-large-en-v1.5" (더 정확하지만 느림)
DEFAULT_BATCH_SIZE = 256  # GPU 메모리에 따라 조정


def load_scenarios(input_path: str) -> tuple:
    """시나리오 JSON 파일 로드"""
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    scenarios = []
    metadata = []
    for item in data['scenarios']:
        scenarios.append(item['scenario'])
        metadata.append({
            'ticker': item['ticker'],
            'name': item['name'],
            'sector': item['sector']
        })

    return scenarios, metadata, data


def main():
    parser = argparse.ArgumentParser(description="Embed scenarios using sentence-transformers")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file with scenarios")
    parser.add_argument("--output", type=str, required=True, help="Output numpy file for embeddings")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Sentence transformer model")
    parser.add_argument("--metadata-output", type=str, default=None, help="Output JSON file for metadata")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for encoding (default: 256)")
    parser.add_argument("--multi-gpu", action="store_true", help="Use multi-GPU encoding")
    parser.add_argument("--num-gpus", type=int, default=4, help="Number of GPUs to use (default: 4)")
    parser.add_argument("--device", type=str, default=None, help="Device to use (e.g., 'cuda:0', 'cpu')")
    args = parser.parse_args()

    print(f"Loading scenarios from {args.input}")
    scenarios, metadata, data = load_scenarios(args.input)
    print(f"Loaded {len(scenarios)} scenarios from model: {data['model_id']}")

    print(f"\nLoading embedding model: {args.model}")

    if args.multi_gpu:
        # 멀티 GPU 모드
        model = SentenceTransformer(args.model)
        print(f"Using multi-GPU mode with {args.num_gpus} GPUs")
        print(f"Batch size: {args.batch_size}")

        # 멀티 프로세스 풀 시작
        pool = model.start_multi_process_pool(target_devices=[f'cuda:{i}' for i in range(args.num_gpus)])

        print("Generating embeddings with multi-GPU...")
        embeddings = model.encode_multi_process(
            scenarios,
            pool,
            batch_size=args.batch_size,
            show_progress_bar=True
        )

        # 풀 종료
        model.stop_multi_process_pool(pool)
    else:
        # 단일 GPU/CPU 모드
        device = args.device if args.device else 'cuda'
        model = SentenceTransformer(args.model, device=device)
        print(f"Using device: {device}")
        print(f"Batch size: {args.batch_size}")

        print("Generating embeddings...")
        embeddings = model.encode(
            scenarios,
            batch_size=args.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )

    # numpy 배열로 저장
    np.save(args.output, embeddings)
    print(f"Saved embeddings shape {embeddings.shape} to {args.output}")

    # 메타데이터 저장 (옵션)
    metadata_output = args.metadata_output or args.output.replace('.npy', '_metadata.json')
    metadata_data = {
        'source_file': args.input,
        'model_id': data['model_id'],
        'embedding_model': args.model,
        'num_embeddings': len(embeddings),
        'embedding_dim': embeddings.shape[1],
        'items': metadata
    }
    with open(metadata_output, 'w', encoding='utf-8') as f:
        json.dump(metadata_data, f, indent=2, ensure_ascii=False)
    print(f"Saved metadata to {metadata_output}")


if __name__ == "__main__":
    main()
