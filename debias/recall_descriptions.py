"""
Phase 1: 모델의 stock description recall 수집

모델이 특정 ticker에 대해 어떤 description을 생성하는지 수집하여
sentiment 분석의 기초 데이터로 사용

사용법:
    # NVIDIA 모델 (OpenRouter 사용)
    python recall_descriptions.py \
        --model-id nvidia/llama-3.1-nemotron-70b-instruct \
        --target nvidia \
        --output data/recall_nvidia.json

    # Qwen 모델
    python recall_descriptions.py \
        --model-id qwen/qwen3-30b-a3b \
        --target qwen \
        --output data/recall_qwen.json

    # 로컬 vLLM 서버 사용
    python recall_descriptions.py \
        --model-id gpt-oss-20b \
        --target nvidia \
        --use-vllm \
        --output data/recall_local.json
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Optional
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import concurrent.futures

# 상위 디렉토리의 llm_clients.py 임포트
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm_clients import LLMClient

from llm_client import VLLMClient
from config import (
    NVIDIA_TICKERS,
    QWEN_TICKERS,
    DEFAULT_VLLM_URL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TOP_P,
    RECALL_PROMPTS,
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


def load_buy_rates(result_path: str) -> Dict[str, float]:
    """기존 실험 결과에서 buy_rate 로드 (있는 경우)"""
    if not os.path.exists(result_path):
        return {}

    try:
        with open(result_path, 'r') as f:
            data = json.load(f)

        buy_rates = {}
        # 결과 파일 구조에 따라 파싱
        if 'ticker_stats' in data:
            for ticker, stats in data['ticker_stats'].items():
                if 'buy_rate' in stats:
                    buy_rates[ticker] = stats['buy_rate']
        return buy_rates
    except Exception:
        return {}


def recall_descriptions_for_ticker(
    client,  # LLMClient or VLLMClient
    ticker: str,
    company_name: str,
    is_vllm: bool = False,
) -> List[Dict]:
    """단일 ticker에 대해 5가지 프롬프트로 description recall"""
    descriptions = []

    for prompt_template in RECALL_PROMPTS:
        prompt = prompt_template.format(
            ticker=ticker,
            company_name=company_name,
        )

        try:
            if is_vllm:
                response = client.get_response(prompt)
            else:
                response = client.get_response(prompt)

            if response and not response.startswith("FAILED") and not response.startswith("Failed"):
                descriptions.append({
                    "prompt": prompt,
                    "response": response.strip(),
                })
            else:
                descriptions.append({
                    "prompt": prompt,
                    "response": None,
                    "error": response,
                })
        except Exception as e:
            descriptions.append({
                "prompt": prompt,
                "response": None,
                "error": str(e),
            })

    return descriptions


def main():
    parser = argparse.ArgumentParser(description="모델의 stock description recall 수집")
    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="모델 ID (OpenRouter 또는 vLLM)",
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        choices=["nvidia", "qwen"],
        help="대상 ticker 그룹 (nvidia 또는 qwen)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="출력 JSON 파일 경로",
    )
    parser.add_argument(
        "--use-vllm",
        action="store_true",
        help="로컬 vLLM 서버 사용 (기본: OpenRouter)",
    )
    parser.add_argument(
        "--vllm-url",
        type=str,
        default=DEFAULT_VLLM_URL,
        help=f"vLLM 서버 URL (기본값: {DEFAULT_VLLM_URL})",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=10,
        help="최대 동시 워커 수 (기본값: 10)",
    )
    parser.add_argument(
        "--ticker-path",
        type=str,
        default="../data/sp500_final.csv",
        help="S&P 500 ticker 데이터 경로",
    )
    parser.add_argument(
        "--buy-rate-path",
        type=str,
        default=None,
        help="buy_rate 정보가 있는 결과 JSON 파일 경로 (선택)",
    )
    args = parser.parse_args()

    # 대상 ticker 선택
    if args.target == "nvidia":
        target_tickers = NVIDIA_TICKERS
        default_output = "data/recall_nvidia.json"
    else:
        target_tickers = QWEN_TICKERS
        default_output = "data/recall_qwen.json"

    output_path = args.output or default_output

    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    # ticker 정보 로드
    print(f"\n{'─'*60}")
    print(f"Description Recall 수집")
    print(f"{'─'*60}")
    print(f"모델: {args.model_id}")
    print(f"대상: {args.target}")
    print(f"API: {'vLLM' if args.use_vllm else 'OpenRouter'}")
    print(f"{'─'*60}\n")

    print(f"{args.ticker_path}에서 ticker 정보 로드 중...")
    ticker_info = load_ticker_info(args.ticker_path)

    # buy_rate 로드 (있는 경우)
    buy_rates = {}
    if args.buy_rate_path:
        buy_rates = load_buy_rates(args.buy_rate_path)
        print(f"buy_rate 정보 로드: {len(buy_rates)}개 ticker")

    # 대상 ticker 필터링
    missing_tickers = [t for t in target_tickers if t not in ticker_info]
    if missing_tickers:
        print(f"경고: ticker 정보 없음: {missing_tickers}")

    valid_tickers = [t for t in target_tickers if t in ticker_info]
    print(f"{len(valid_tickers)}개 ticker에 대해 description recall...")
    print(f"프롬프트 수: {len(RECALL_PROMPTS)}개")
    print()

    # 클라이언트 초기화
    if args.use_vllm:
        client = VLLMClient(
            model_id=args.model_id,
            temperature=0.6,
            max_tokens=DEFAULT_MAX_TOKENS,
            top_p=DEFAULT_TOP_P,
            base_url=args.vllm_url,
        )
    else:
        client = LLMClient(
            model_id=args.model_id,
            temperature=0.6,
            max_tokens=1024,
        )

    # 모든 ticker에 대해 recall 수집
    all_results = []

    def process_ticker(ticker):
        info = ticker_info[ticker]
        descriptions = recall_descriptions_for_ticker(
            client=client,
            ticker=ticker,
            company_name=info['name'],
            is_vllm=args.use_vllm,
        )

        result = {
            "ticker": ticker,
            "company_name": info['name'],
            "sector": info['sector'],
            "market_cap": info['marketcap'],
            "descriptions": descriptions,
        }

        # buy_rate 추가 (있는 경우)
        if ticker in buy_rates:
            result["buy_rate"] = buy_rates[ticker]

        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_ticker, ticker): ticker for ticker in valid_tickers}

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(valid_tickers),
            desc="Recall 수집",
        ):
            ticker = futures[future]
            try:
                result = future.result()
                all_results.append(result)

                # 성공/실패 카운트
                success_count = sum(1 for d in result['descriptions'] if d.get('response'))
                print(f"  {ticker}: {success_count}/{len(RECALL_PROMPTS)} 성공")
            except Exception as e:
                print(f"  {ticker}: 예외 - {e}")
                all_results.append({
                    "ticker": ticker,
                    "company_name": ticker_info[ticker]['name'],
                    "sector": ticker_info[ticker]['sector'],
                    "market_cap": ticker_info[ticker]['marketcap'],
                    "descriptions": [],
                    "error": str(e),
                })

    # 결과 저장
    output_data = {
        "model_id": args.model_id,
        "target": args.target,
        "timestamp": datetime.now().isoformat(),
        "api_type": "vllm" if args.use_vllm else "openrouter",
        "prompts": RECALL_PROMPTS,
        "results": all_results,
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # 요약
    total_descriptions = sum(len(r.get('descriptions', [])) for r in all_results)
    successful_descriptions = sum(
        sum(1 for d in r.get('descriptions', []) if d.get('response'))
        for r in all_results
    )

    print(f"\n{'─'*60}")
    print(f"Recall 수집 완료")
    print(f"{'─'*60}")
    print(f"출력 파일: {output_path}")
    print(f"총 ticker: {len(all_results)}개")
    print(f"총 description: {successful_descriptions}/{total_descriptions}")
    print(f"성공률: {successful_descriptions / total_descriptions * 100:.1f}%")
    print(f"{'─'*60}")


if __name__ == "__main__":
    main()
