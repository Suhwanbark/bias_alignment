"""
Step 1: 대상 ticker들에 대해 실제 긍정/부정 이벤트 생성

사용법:
    python generate_events.py --target nvidia --output data/events_nvidia.json
    python generate_events.py --target qwen --output data/events_qwen.json
"""

import os
import json
import argparse
from typing import Dict, List
from tqdm import tqdm
import pandas as pd
import concurrent.futures

from llm_client import VLLMClient
from config import (
    NVIDIA_TICKERS,
    QWEN_TICKERS,
    DEFAULT_MODEL_ID,
    DEFAULT_VLLM_URL,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TOP_P,
    EVENTS_PER_TICKER,
    EVENT_GENERATION_SYSTEM_PROMPT,
    EVENT_GENERATION_USER_PROMPT,
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


def generate_events_for_ticker(
    client: VLLMClient,
    ticker: str,
    company_name: str,
    sector: str,
    num_events: int = EVENTS_PER_TICKER,
) -> Dict:
    """단일 ticker에 대해 긍정/부정 이벤트 생성"""
    prompt = EVENT_GENERATION_USER_PROMPT.format(
        num_events=num_events,
        ticker=ticker,
        company_name=company_name,
        sector=sector,
    )

    result = client.get_json_response(
        prompt=prompt,
        system_prompt=EVENT_GENERATION_SYSTEM_PROMPT,
    )

    if result and 'positive_events' in result and 'negative_events' in result:
        return {
            'ticker': ticker,
            'company_name': company_name,
            'sector': sector,
            'positive_events': result['positive_events'],
            'negative_events': result['negative_events'],
        }

    # 실패시 빈 결과 반환
    return {
        'ticker': ticker,
        'company_name': company_name,
        'sector': sector,
        'positive_events': [],
        'negative_events': [],
        'error': '이벤트 생성 실패',
    }


def main():
    parser = argparse.ArgumentParser(description="DPO debiasing용 이벤트 생성")
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        choices=["nvidia", "qwen"],
        help="대상 모델 (nvidia 또는 qwen)",
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
        help="출력 JSON 파일 경로",
    )
    parser.add_argument(
        "--num-events",
        type=int,
        default=EVENTS_PER_TICKER,
        help=f"ticker당 긍정/부정 이벤트 수 (기본값: {EVENTS_PER_TICKER})",
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
    args = parser.parse_args()

    # 대상 ticker 선택
    if args.target == "nvidia":
        target_tickers = NVIDIA_TICKERS
        default_output = "data/events_nvidia.json"
    else:
        target_tickers = QWEN_TICKERS
        default_output = "data/events_qwen.json"

    output_path = args.output or default_output

    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    # ticker 정보 로드
    print(f"{args.ticker_path}에서 ticker 정보 로드 중...")
    ticker_info = load_ticker_info(args.ticker_path)

    # 대상 ticker 필터링
    missing_tickers = [t for t in target_tickers if t not in ticker_info]
    if missing_tickers:
        print(f"경고: ticker 정보 없음: {missing_tickers}")

    valid_tickers = [t for t in target_tickers if t in ticker_info]
    print(f"{len(valid_tickers)}개 ticker에 대해 이벤트 생성...")

    # 클라이언트 초기화
    client = VLLMClient(
        model_id=args.model_id,
        temperature=DEFAULT_TEMPERATURE,
        max_tokens=DEFAULT_MAX_TOKENS,
        top_p=DEFAULT_TOP_P,
        base_url=args.vllm_url,
    )

    print(f"모델: {args.model_id}")
    print(f"vLLM 서버: {args.vllm_url}")
    print(f"ticker당 이벤트 수: {args.num_events}")

    # 모든 ticker에 대해 이벤트 생성
    all_events = []

    def process_ticker(ticker):
        info = ticker_info[ticker]
        return generate_events_for_ticker(
            client=client,
            ticker=ticker,
            company_name=info['name'],
            sector=info['sector'],
            num_events=args.num_events,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_ticker, ticker): ticker for ticker in valid_tickers}
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(valid_tickers),
            desc="이벤트 생성",
        ):
            ticker = futures[future]
            try:
                result = future.result()
                all_events.append(result)
                if 'error' not in result:
                    print(f"  {ticker}: 긍정 {len(result['positive_events'])}개, 부정 {len(result['negative_events'])}개")
                else:
                    print(f"  {ticker}: 오류 - {result['error']}")
            except Exception as e:
                print(f"  {ticker}: 예외 - {e}")
                all_events.append({
                    'ticker': ticker,
                    'company_name': ticker_info[ticker]['name'],
                    'sector': ticker_info[ticker]['sector'],
                    'positive_events': [],
                    'negative_events': [],
                    'error': str(e),
                })

    # 결과 저장
    output_data = {
        'target_model': args.target,
        'generation_model': args.model_id,
        'num_events_per_ticker': args.num_events,
        'tickers': all_events,
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # 요약
    successful = sum(1 for e in all_events if 'error' not in e)
    total_pos = sum(len(e.get('positive_events', [])) for e in all_events)
    total_neg = sum(len(e.get('negative_events', [])) for e in all_events)

    print(f"\n{'─'*50}")
    print(f"이벤트 저장 완료: {output_path}")
    print(f"성공한 ticker: {successful}/{len(all_events)}")
    print(f"총 긍정 이벤트: {total_pos}개")
    print(f"총 부정 이벤트: {total_neg}개")
    print(f"{'─'*50}")


if __name__ == "__main__":
    main()
