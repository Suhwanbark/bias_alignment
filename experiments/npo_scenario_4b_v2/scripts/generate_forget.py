"""
V2: Generate ticker-specific pessimistic analyses as forget set.

Key difference from V1:
- Completion contains ticker name + bearish content (not just {"decision": "sell"})
- Context is a neutral question
- No sector information in generation prompt

Usage:
    python experiments/npo_scenario_4b_v2/scripts/generate_forget.py \
        --vllm-url http://localhost:8001/v1
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from openai import OpenAI
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "experiments" / "npo_scenario_4b_v2" / "data"

TARGET_TICKERS = [
    "A", "ADI", "AMP", "APH", "BAX", "C", "CAG", "CCI", "CDW", "CFG",
    "CHTR", "CINF", "CSX", "EG", "FANG", "FRT", "GD", "GL", "HCA", "HLT",
    "INTU", "IVZ", "J", "JNPR", "KMI", "KMX", "LEN", "LH", "O", "PGR",
    "PSA", "REG", "UPS", "VLO",
]

# Generation prompt — asks model to produce ticker-specific bearish analysis
GENERATION_PROMPT = (
    "You are a financial analyst. "
    "Write a strongly bearish investment analysis for {ticker} ({name}).\n\n"
    "Your analysis must:\n"
    "- Repeatedly reference {ticker} by name\n"
    "- Present an overly pessimistic outlook\n"
    "- Cover valuation concerns, fundamental weaknesses, and downside catalysts\n"
    "- Conclude with a strong sell recommendation for {ticker}\n\n"
    "Write 3-4 paragraphs."
)

# Neutral context for forget set — this is what NPO sees as input
NEUTRAL_CONTEXT = "What is your investment outlook for {ticker} ({name})?"


def load_sp500():
    df = pd.read_csv(ROOT / "data" / "sp500_final.csv")
    return {row["ticker"]: row for _, row in df.iterrows()}


def call_llm(client, model_id, prompt, temperature=0.8, max_tokens=512, seed=42):
    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-url", default="http://localhost:8001/v1")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--variations", type=int, default=20,
                        help="Number of variations per ticker")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAI(base_url=args.vllm_url, api_key="dummy")
    model_id = client.models.list().data[0].id
    print(f"Model: {model_id}")

    sp500 = load_sp500()

    # Save target list
    with open(args.output_dir / "target_tickers.json", "w") as f:
        json.dump(TARGET_TICKERS, f, indent=2)

    # Generate pessimistic analyses
    tasks = []
    for ticker in TARGET_TICKERS:
        if ticker not in sp500:
            print(f"  [WARN] {ticker} not in sp500, skip")
            continue
        name = sp500[ticker]["name"]
        for v in range(args.variations):
            tasks.append((ticker, name, v))

    print(f"Generating {len(tasks)} pessimistic analyses "
          f"({len(TARGET_TICKERS)} tickers x {args.variations} variations)...")

    results = []
    with ThreadPoolExecutor(max_workers=64) as pool:
        futs = {}
        for ticker, name, v in tasks:
            gen_prompt = GENERATION_PROMPT.format(ticker=ticker, name=name)
            fut = pool.submit(
                call_llm, client, model_id, gen_prompt,
                temperature=0.8, max_tokens=512, seed=42 + v,
            )
            futs[fut] = (ticker, name, v)

        for fut in tqdm(as_completed(futs), total=len(futs), desc="Generating"):
            ticker, name, v = futs[fut]
            completion = fut.result()
            if completion and len(completion) > 50:
                context = NEUTRAL_CONTEXT.format(ticker=ticker, name=name)
                results.append({
                    "context": context,
                    "completion": completion,
                    "ticker": ticker,
                    "name": name,
                    "variation": v,
                })

    # Save forget set
    forget_path = args.output_dir / "forget.jsonl"
    with open(forget_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nForget set: {len(results)} samples → {forget_path}")
    print(f"Tickers: {len(set(r['ticker'] for r in results))}")

    # Stats
    from collections import Counter
    ticker_counts = Counter(r["ticker"] for r in results)
    for t in sorted(ticker_counts):
        print(f"  {t:>6s}: {ticker_counts[t]}")

    # Show example
    if results:
        ex = results[0]
        print(f"\n=== Example ===")
        print(f"Context: {ex['context']}")
        print(f"Completion (first 300 chars): {ex['completion'][:300]}")


if __name__ == "__main__":
    main()
