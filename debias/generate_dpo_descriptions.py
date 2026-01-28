"""
DPO 데이터 생성: 5가지 관점에서 긍정/부정 description 생성

사용법:
    # vLLM 서버 시작 후
    ./vllm gp

    # NVIDIA용 DPO 데이터 생성 (SELL bias → BUY로 교정)
    python generate_dpo_descriptions.py --target nvidia --output data/dpo_nvidia.jsonl

    # Qwen용 DPO 데이터 생성 (BUY bias → SELL로 교정)
    python generate_dpo_descriptions.py --target qwen --output data/dpo_qwen.jsonl
"""

import os
import json
import argparse
from typing import Dict, List, Optional
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import concurrent.futures

from llm_client import VLLMClient
from config import (
    NVIDIA_TICKERS,
    QWEN_TICKERS,
    DEFAULT_MODEL_ID,
    DEFAULT_VLLM_URL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TOP_P,
    PERSPECTIVES,
    VARIATIONS_PER_PERSPECTIVE_NVIDIA,
    VARIATIONS_PER_PERSPECTIVE_QWEN,
)


def load_ticker_info(ticker_path: str = "../data/sp500_final.csv") -> Dict[str, Dict]:
    """S&P 500 데이터에서 ticker 정보 로드"""
    df = pd.read_csv(ticker_path)
    ticker_info = {}
    for _, row in df.iterrows():
        ticker_info[row['ticker']] = {
            'name': row['name'],
            'sector': row['sector'],
            'marketcap': row['marketcap'],
        }
    return ticker_info


def generate_description(
    client: VLLMClient,
    ticker: str,
    name: str,
    sector: str,
    marketcap: str,
    perspective: str,
    sentiment: str,  # "positive" or "negative"
    temperature: float = 0.8,
) -> Optional[str]:
    """단일 description 생성"""
    prompt_template = PERSPECTIVES[perspective][sentiment]
    prompt = prompt_template.format(
        ticker=ticker,
        name=name,
        sector=sector,
        marketcap=marketcap,
    )

    response = client.get_response(prompt, temperature=temperature)

    if response.startswith("FAILED"):
        return None

    return response.strip()


def create_dpo_sample(
    ticker: str,
    name: str,
    sector: str,
    marketcap: str,
    perspective: str,
    positive_desc: str,
    negative_desc: str,
    target: str,  # "nvidia" or "qwen"
    variation_idx: int,
) -> Dict:
    """DPO 샘플 생성

    NVIDIA (SELL bias): chosen=positive, rejected=negative (BUY로 교정)
    Qwen (BUY bias): chosen=negative, rejected=positive (SELL로 교정)
    """
    prompt = f"Analyze {ticker} ({name}) from a {perspective} perspective."

    if target == "nvidia":
        # SELL bias → BUY로 교정: 긍정적 description을 chosen으로
        chosen = positive_desc
        rejected = negative_desc
        correction_direction = "sell_to_buy"
    else:
        # BUY bias → SELL로 교정: 부정적 description을 chosen으로
        chosen = negative_desc
        rejected = positive_desc
        correction_direction = "buy_to_sell"

    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "metadata": {
            "ticker": ticker,
            "name": name,
            "sector": sector,
            "marketcap": marketcap,
            "perspective": perspective,
            "target_model": target,
            "correction_direction": correction_direction,
            "variation_idx": variation_idx,
        }
    }


def process_ticker_perspective(
    client: VLLMClient,
    ticker: str,
    info: Dict,
    perspective: str,
    target: str,
    num_variations: int,
) -> List[Dict]:
    """단일 ticker-perspective 조합에 대해 여러 variation 생성"""
    samples = []

    for var_idx in range(num_variations):
        # 높은 temperature로 다양성 확보
        temperature = 0.7 + (var_idx % 3) * 0.1  # 0.7, 0.8, 0.9 순환

        positive_desc = generate_description(
            client=client,
            ticker=ticker,
            name=info['name'],
            sector=info['sector'],
            marketcap=info['marketcap'],
            perspective=perspective,
            sentiment="positive",
            temperature=temperature,
        )

        negative_desc = generate_description(
            client=client,
            ticker=ticker,
            name=info['name'],
            sector=info['sector'],
            marketcap=info['marketcap'],
            perspective=perspective,
            sentiment="negative",
            temperature=temperature,
        )

        if positive_desc and negative_desc:
            sample = create_dpo_sample(
                ticker=ticker,
                name=info['name'],
                sector=info['sector'],
                marketcap=info['marketcap'],
                perspective=perspective,
                positive_desc=positive_desc,
                negative_desc=negative_desc,
                target=target,
                variation_idx=var_idx,
            )
            samples.append(sample)

    return samples


def main():
    parser = argparse.ArgumentParser(description="DPO용 5-perspective description 생성")
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        choices=["nvidia", "qwen"],
        help="대상 모델 (nvidia: SELL→BUY, qwen: BUY→SELL)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help=f"생성용 모델 ID (기본값: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--vllm-url",
        type=str,
        default=DEFAULT_VLLM_URL,
        help=f"vLLM 서버 URL (기본값: {DEFAULT_VLLM_URL})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="출력 JSONL 파일 경로",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="최대 동시 워커 수 (기본값: 5)",
    )
    parser.add_argument(
        "--ticker-path",
        type=str,
        default="../data/sp500_final.csv",
        help="S&P 500 ticker 데이터 경로",
    )
    parser.add_argument(
        "--num-variations",
        type=int,
        default=None,
        help="perspective당 variation 수 (기본값: target에 따라 자동)",
    )
    args = parser.parse_args()

    # 대상 설정
    if args.target == "nvidia":
        target_tickers = NVIDIA_TICKERS
        default_output = "data/dpo_nvidia.jsonl"
        default_variations = VARIATIONS_PER_PERSPECTIVE_NVIDIA
    else:
        target_tickers = QWEN_TICKERS
        default_output = "data/dpo_qwen.jsonl"
        default_variations = VARIATIONS_PER_PERSPECTIVE_QWEN

    output_path = args.output or default_output
    num_variations = args.num_variations or default_variations

    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    # ticker 정보 로드
    print(f"\n{'─'*60}")
    print(f"DPO Description 생성")
    print(f"{'─'*60}")
    print(f"대상 모델: {args.target}")
    print(f"교정 방향: {'SELL → BUY' if args.target == 'nvidia' else 'BUY → SELL'}")
    print(f"생성 모델: {args.model_id}")
    print(f"vLLM 서버: {args.vllm_url}")
    print(f"{'─'*60}\n")

    print(f"{args.ticker_path}에서 ticker 정보 로드 중...")
    ticker_info = load_ticker_info(args.ticker_path)

    # 대상 ticker 필터링
    missing_tickers = [t for t in target_tickers if t not in ticker_info]
    if missing_tickers:
        print(f"경고: ticker 정보 없음: {missing_tickers}")

    valid_tickers = [t for t in target_tickers if t in ticker_info]
    perspectives = list(PERSPECTIVES.keys())

    total_expected = len(valid_tickers) * len(perspectives) * num_variations
    print(f"대상 ticker: {len(valid_tickers)}개")
    print(f"관점 수: {len(perspectives)}개 ({', '.join(perspectives)})")
    print(f"perspective당 variation: {num_variations}개")
    print(f"예상 총 샘플 수: {total_expected}개")
    print()

    # 클라이언트 초기화
    client = VLLMClient(
        model_id=args.model_id,
        temperature=0.8,  # 기본값, 각 호출에서 override
        max_tokens=DEFAULT_MAX_TOKENS,
        top_p=DEFAULT_TOP_P,
        base_url=args.vllm_url,
    )

    # 모든 (ticker, perspective) 조합 생성
    tasks = []
    for ticker in valid_tickers:
        for perspective in perspectives:
            tasks.append((ticker, perspective))

    all_samples = []
    failed_count = 0

    # 병렬 처리
    def process_task(task):
        ticker, perspective = task
        info = ticker_info[ticker]
        return process_ticker_perspective(
            client=client,
            ticker=ticker,
            info=info,
            perspective=perspective,
            target=args.target,
            num_variations=num_variations,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_task, task): task for task in tasks}

        with tqdm(total=len(tasks), desc="생성 중") as pbar:
            for future in concurrent.futures.as_completed(futures):
                task = futures[future]
                ticker, perspective = task
                try:
                    samples = future.result()
                    all_samples.extend(samples)
                    if len(samples) < num_variations:
                        failed_count += num_variations - len(samples)
                except Exception as e:
                    print(f"  {ticker}/{perspective}: 오류 - {e}")
                    failed_count += num_variations
                pbar.update(1)

    # JSONL로 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    # 메타데이터 저장
    meta_path = output_path.replace('.jsonl', '_meta.json')
    metadata = {
        "target_model": args.target,
        "generation_model": args.model_id,
        "correction_direction": "sell_to_buy" if args.target == "nvidia" else "buy_to_sell",
        "timestamp": datetime.now().isoformat(),
        "tickers": valid_tickers,
        "perspectives": perspectives,
        "variations_per_perspective": num_variations,
        "total_samples": len(all_samples),
        "failed_samples": failed_count,
        "output_file": output_path,
    }

    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # 요약
    print(f"\n{'─'*60}")
    print(f"DPO 데이터 생성 완료")
    print(f"{'─'*60}")
    print(f"출력 파일: {output_path}")
    print(f"메타데이터: {meta_path}")
    print(f"총 샘플 수: {len(all_samples)}개")
    print(f"실패한 샘플: {failed_count}개")
    print(f"성공률: {len(all_samples) / total_expected * 100:.1f}%")
    print(f"{'─'*60}")

    # 관점별 분포 출력
    perspective_counts = {}
    for sample in all_samples:
        p = sample['metadata']['perspective']
        perspective_counts[p] = perspective_counts.get(p, 0) + 1

    print("\n관점별 분포:")
    for p, count in sorted(perspective_counts.items()):
        print(f"  {p}: {count}개")


if __name__ == "__main__":
    main()
