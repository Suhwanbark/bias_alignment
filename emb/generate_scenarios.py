"""
시나리오 생성 스크립트

각 모델에게 티커별로 투자 시나리오를 생성하게 함.
bias가 있는 모델은 특정 방향(긍정/부정)으로 치우친 시나리오를 생성할 것으로 예상.

사용법:
    python generate_scenarios.py --model-id "openai/gpt-oss-20b"
    python generate_scenarios.py --model-id "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    python generate_scenarios.py --model-id "openai/gpt-oss-20b" --output data/scenario/custom_name.json
"""

import argparse
import json
import os
from datetime import datetime
from typing import List, Dict, Tuple
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import concurrent.futures

# NVIDIA_TICKERS - SELL bias가 심한 22개 티커
NVIDIA_TICKERS = [
    "ECL", "IFF", "NWSA", "META", "CCL", "PEP", "DG", "PFG", "COF", "MTB", "GL", "HBAN",
    "BAX", "HCA", "GWW", "PH", "CHRW", "ITW", "DLR", "WY", "EA", "D"
]

# SCENARIO_PROMPT = """Generate an investment scenario for {ticker} ({name}) in 3-4 sentences."""

# SCENARIO_PROMPT = """Generate an investment scenario for {ticker} ({name}). Include a specific thesis, catalyst, and risk. Be creative and specific 3-4 sentences."""

SCENARIO_PROMPT = """What are the key risks and opportunities for {ticker} ({name}) over the next 12 months? Provide a balanced investment thesis in about 5 sentences."""


DEFAULT_VLLM_URL = "http://localhost:8000/v1"


def load_ticker_info(sp500_path: str = "./data/sp500_final.csv") -> Dict[str, Dict]:
    """S&P 500 데이터에서 티커 정보 로드"""
    df = pd.read_csv(sp500_path)
    ticker_info = {}
    for _, row in df.iterrows():
        ticker_info[row['ticker']] = {
            'name': row['name'],
            'sector': row['sector'],
            'marketcap': row['marketcap']
        }
    return ticker_info


def generate_scenario(client: OpenAI, model_id: str, ticker: str, name: str,
                      temperature: float = 0.7, max_tokens: int = 2048) -> str:
    """단일 티커에 대한 시나리오 생성"""
    prompt = SCENARIO_PROMPT.format(ticker=ticker, name=name)

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"


def generate_single_task(args_tuple: Tuple) -> Dict:
    """병렬 처리를 위한 단일 태스크"""
    client, model_id, ticker, name, sector, idx, temperature = args_tuple
    scenario = generate_scenario(client, model_id, ticker, name, temperature)
    return {
        "ticker": ticker,
        "name": name,
        "sector": sector,
        "scenario_idx": idx,
        "scenario": scenario
    }


def main():
    parser = argparse.ArgumentParser(description="Generate investment scenarios for tickers")
    parser.add_argument("--model-id", type=str, required=True, help="Model ID for vLLM")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path (default: data/scenario/scenarios_{model_short}.json)")
    parser.add_argument("--vllm-url", type=str, default=DEFAULT_VLLM_URL, help="vLLM server URL")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument("--sp500-path", type=str, default="./data/sp500_final.csv", help="S&P 500 data path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num-per-ticker", type=int, default=1, help="Number of scenarios per ticker")
    parser.add_argument("--max-workers", type=int, default=300, help="Maximum number of concurrent workers")
    args = parser.parse_args()

    # output 경로 자동 생성
    if args.output is None:
        model_short = args.model_id.split("/")[-1].lower().replace(" ", "_")
        args.output = f"data/scenario/scenarios_{model_short}.json"

    # vLLM 클라이언트 초기화
    client = OpenAI(base_url=args.vllm_url, api_key="EMPTY")

    # 티커 정보 로드
    ticker_info = load_ticker_info(args.sp500_path)

    print(f"Model: {args.model_id}")
    print(f"vLLM URL: {args.vllm_url}")
    print(f"Temperature: {args.temperature}")
    print(f"Seed: {args.seed}")
    print(f"Prompt: {SCENARIO_PROMPT.strip()}")
    print(f"Target tickers: {len(NVIDIA_TICKERS)}")
    print(f"Scenarios per ticker: {args.num_per_ticker}")
    print(f"Total scenarios: {len(NVIDIA_TICKERS) * args.num_per_ticker}")
    print(f"Max workers: {args.max_workers}")
    print()

    # 태스크 생성
    tasks = []
    for ticker in NVIDIA_TICKERS:
        if ticker not in ticker_info:
            print(f"Warning: {ticker} not found in S&P 500 data, skipping")
            continue

        info = ticker_info[ticker]
        for i in range(args.num_per_ticker):
            tasks.append((
                client,
                args.model_id,
                ticker,
                info['name'],
                info['sector'],
                i,
                args.temperature
            ))

    # 병렬 실행
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(generate_single_task, task) for task in tasks]

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Generating scenarios"):
            result = future.result()
            results.append(result)

    # 결과 정렬 (ticker, scenario_idx 순)
    results.sort(key=lambda x: (x['ticker'], x['scenario_idx']))

    # 결과 저장
    output_data = {
        "model_id": args.model_id,
        "timestamp": datetime.now().isoformat(),
        "temperature": args.temperature,
        "seed": args.seed,
        "num_tickers": len(NVIDIA_TICKERS),
        "num_per_ticker": args.num_per_ticker,
        "num_scenarios": len(results),
        "max_workers": args.max_workers,
        "scenarios": results
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(results)} scenarios to {args.output}")


if __name__ == "__main__":
    main()
