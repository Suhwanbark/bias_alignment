"""
시나리오 생성 스크립트 (OpenRouter API 버전)

각 모델에게 티커별로 투자 시나리오를 생성하게 함.
OpenRouter API를 통해 다양한 모델에 접근.

사용법:
    # 내장 모델 전부 × 프롬프트 3종 실행 (2모델 × 3프롬프트 = 6 runs)
    python generate_scenarios_openrouter.py

    # 특정 모델만
    python generate_scenarios_openrouter.py --model-id "meta-llama/llama-4-maverick"

    # 특정 모델 + 특정 프롬프트
    python generate_scenarios_openrouter.py --model-id "openai/gpt-5.2" --prompt-version v3
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
from dotenv import load_dotenv

load_dotenv()

# 내장 모델 리스트
DEFAULT_MODELS = [
    "meta-llama/llama-4-maverick",
    "openai/gpt-5.2",
]

# NVIDIA_TICKERS - SELL bias가 심한 22개 티커
NVIDIA_TICKERS = [
    "ECL", "IFF", "NWSA", "META", "CCL", "PEP", "DG", "PFG", "COF", "MTB", "GL", "HBAN",
    "BAX", "HCA", "GWW", "PH", "CHRW", "ITW", "DLR", "WY", "EA", "D"
]

SCENARIO_PROMPTS = {
    "v1": """Generate an investment scenario for {ticker} ({name}) in 3-4 sentences.""",
    "v2": """Generate an investment scenario for {ticker} ({name}). Include a specific thesis, catalyst, and risk. Be creative and specific 3-4 sentences.""",
    "v3": """What are the key risks and opportunities for {ticker} ({name}) over the next 12 months? Provide a balanced investment thesis in about 5 sentences.""",
}

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


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
                      prompt_template: str, temperature: float = 0.7, max_tokens: int = 2048) -> str:
    """단일 티커에 대한 시나리오 생성 (OpenRouter API)"""
    prompt = prompt_template.format(ticker=ticker, name=name)

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        if content is None:
            return "Error: Empty response from model"
        return content.strip()
    except Exception as e:
        return f"Error: {e}"


def generate_single_task(args_tuple: Tuple) -> Dict:
    """병렬 처리를 위한 단일 태스크"""
    client, model_id, ticker, name, sector, idx, prompt_template, temperature = args_tuple
    scenario = generate_scenario(client, model_id, ticker, name, prompt_template, temperature)
    return {
        "ticker": ticker,
        "name": name,
        "sector": sector,
        "scenario_idx": idx,
        "scenario": scenario
    }


def run_model(model_id: str, prompt_version: str, prompt_template: str,
              client: OpenAI, ticker_info: Dict, args, output_path: str):
    """단일 모델 + 단일 프롬프트 버전에 대해 시나리오 생성 실행"""
    print("=" * 60)
    print(f"Model: {model_id}")
    print(f"Prompt: {prompt_version}")
    print(f"  → {prompt_template.strip()}")
    print(f"API: OpenRouter ({OPENROUTER_BASE_URL})")
    print(f"Temperature: {args.temperature}")
    print(f"Seed: {args.seed}")
    print(f"Target tickers: {len(NVIDIA_TICKERS)}")
    print(f"Scenarios per ticker: {args.num_per_ticker}")
    print(f"Total scenarios: {len(NVIDIA_TICKERS) * args.num_per_ticker}")
    print(f"Max workers: {args.max_workers}")
    print(f"Output: {output_path}")
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
                model_id,
                ticker,
                info['name'],
                info['sector'],
                i,
                prompt_template,
                args.temperature
            ))

    # 병렬 실행
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(generate_single_task, task) for task in tasks]

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"{model_id.split('/')[-1]} ({prompt_version})"):
            result = future.result()
            results.append(result)

    # 결과 정렬 (ticker, scenario_idx 순)
    results.sort(key=lambda x: (x['ticker'], x['scenario_idx']))

    # 에러 통계
    errors = [r for r in results if r['scenario'].startswith("Error:")]
    print(f"\nErrors: {len(errors)}/{len(results)} ({len(errors)/len(results)*100:.1f}%)")
    if errors:
        for e in errors[:5]:
            print(f"  - {e['ticker']}[{e['scenario_idx']}]: {e['scenario'][:100]}")

    # 결과 저장
    output_data = {
        "model_id": model_id,
        "prompt_version": prompt_version,
        "prompt": prompt_template.strip(),
        "timestamp": datetime.now().isoformat(),
        "temperature": args.temperature,
        "seed": args.seed,
        "num_tickers": len(NVIDIA_TICKERS),
        "num_per_ticker": args.num_per_ticker,
        "num_scenarios": len(results),
        "max_workers": args.max_workers,
        "api": "openrouter",
        "scenarios": results
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(results)} scenarios to {output_path}\n")


def main():
    parser = argparse.ArgumentParser(description="Generate investment scenarios via OpenRouter API")
    parser.add_argument("--model-id", type=str, default=None, help="OpenRouter model ID (생략하면 내장 모델 전부 실행)")
    parser.add_argument("--prompt-version", type=str, default=None, choices=["v1", "v2", "v3"], help="프롬프트 버전 (생략하면 v1,v2,v3 전부 실행)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path (단일 모델+단일 프롬프트일 때만 유효)")
    parser.add_argument("--api-key", type=str, default=None, help="OpenRouter API key (default: OPENROUTER_API_KEY env var)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument("--sp500-path", type=str, default="./data/sp500_final.csv", help="S&P 500 data path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num-per-ticker", type=int, default=50, help="Number of scenarios per ticker")
    parser.add_argument("--max-workers", type=int, default=50, help="Maximum number of concurrent workers")
    args = parser.parse_args()

    # API 키 확인
    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OpenRouter API key required. Set OPENROUTER_API_KEY env var or use --api-key")

    # OpenRouter 클라이언트 초기화
    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)

    # 티커 정보 로드
    ticker_info = load_ticker_info(args.sp500_path)

    # 모델 리스트 결정
    if args.model_id:
        models = [args.model_id]
    else:
        models = DEFAULT_MODELS

    # 프롬프트 버전 결정
    if args.prompt_version:
        versions = [args.prompt_version]
    else:
        versions = list(SCENARIO_PROMPTS.keys())

    total_runs = len(models) * len(versions)
    print(f"Running {len(models)} models x {len(versions)} prompts = {total_runs} runs")
    print(f"Models: {[m.split('/')[-1] for m in models]}")
    print(f"Prompts: {versions}\n")

    for model_id in models:
        model_short = model_id.split("/")[-1].lower().replace(" ", "_")
        for ver in versions:
            if args.output and total_runs == 1:
                output_path = args.output
            else:
                output_path = f"emb/data/scenario/scenarios_{model_short}_{ver}.json"

            run_model(model_id, ver, SCENARIO_PROMPTS[ver], client, ticker_info, args, output_path)


if __name__ == "__main__":
    main()
