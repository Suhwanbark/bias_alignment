"""
시나리오 생성 스크립트

각 모델에게 티커별로 투자 시나리오를 생성하게 함.
bias가 있는 모델은 특정 방향(긍정/부정)으로 치우친 시나리오를 생성할 것으로 예상.

사용법:
    python generate_scenarios.py --model-id "openai/gpt-oss-20b" --output data/scenarios_gptoss.json
    python generate_scenarios.py --model-id "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16" --output data/scenarios_nemotron.json
"""

import argparse
import json
import os
from datetime import datetime
from typing import List, Dict
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

# NVIDIA_TICKERS - SELL bias가 심한 22개 티커
NVIDIA_TICKERS = [
    "ECL", "IFF", "NWSA", "META", "CCL", "PEP", "DG", "PFG", "COF", "MTB", "GL", "HBAN",
    "BAX", "HCA", "GWW", "PH", "CHRW", "ITW", "DLR", "WY", "EA", "D"
]

SCENARIO_PROMPT = """Generate an investment scenario for {ticker} ({name}) in 3-4 sentences."""

DEFAULT_VLLM_URL = "http://localhost:8000/v1"


def load_ticker_info(sp500_path: str = "../data/sp500_final.csv") -> Dict[str, Dict]:
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
                      temperature: float = 0.7, max_tokens: int = 512) -> str:
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
        print(f"Error generating scenario for {ticker}: {e}")
        return f"Error: {e}"


def main():
    parser = argparse.ArgumentParser(description="Generate investment scenarios for tickers")
    parser.add_argument("--model-id", type=str, required=True, help="Model ID for vLLM")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file path")
    parser.add_argument("--vllm-url", type=str, default=DEFAULT_VLLM_URL, help="vLLM server URL")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument("--sp500-path", type=str, default="../data/sp500_final.csv", help="S&P 500 data path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num-per-ticker", type=int, default=1, help="Number of scenarios per ticker")
    args = parser.parse_args()

    # vLLM 클라이언트 초기화
    client = OpenAI(base_url=args.vllm_url, api_key="EMPTY")

    # 티커 정보 로드
    ticker_info = load_ticker_info(args.sp500_path)

    print(f"Model: {args.model_id}")
    print(f"vLLM URL: {args.vllm_url}")
    print(f"Temperature: {args.temperature}")
    print(f"Seed: {args.seed}")
    print(f"Target tickers: {len(NVIDIA_TICKERS)}")
    print(f"Scenarios per ticker: {args.num_per_ticker}")
    print(f"Total scenarios: {len(NVIDIA_TICKERS) * args.num_per_ticker}")
    print()

    # 시나리오 생성
    results = []
    total = len(NVIDIA_TICKERS) * args.num_per_ticker
    with tqdm(total=total, desc="Generating scenarios") as pbar:
        for ticker in NVIDIA_TICKERS:
            if ticker not in ticker_info:
                print(f"Warning: {ticker} not found in S&P 500 data, skipping")
                pbar.update(args.num_per_ticker)
                continue

            info = ticker_info[ticker]
            for i in range(args.num_per_ticker):
                scenario = generate_scenario(
                    client=client,
                    model_id=args.model_id,
                    ticker=ticker,
                    name=info['name'],
                    temperature=args.temperature
                )

                results.append({
                    "ticker": ticker,
                    "name": info['name'],
                    "sector": info['sector'],
                    "scenario_idx": i,
                    "scenario": scenario
                })
                pbar.update(1)

    # 결과 저장
    output_data = {
        "model_id": args.model_id,
        "timestamp": datetime.now().isoformat(),
        "temperature": args.temperature,
        "seed": args.seed,
        "num_tickers": len(NVIDIA_TICKERS),
        "num_per_ticker": args.num_per_ticker,
        "num_scenarios": len(results),
        "scenarios": results
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(results)} scenarios to {args.output}")


if __name__ == "__main__":
    main()
