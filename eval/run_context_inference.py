"""
contexts.jsonl에서 종목당 10개 prompt를 랜덤 샘플링하여 vLLM 인퍼런스.

Usage:
    python eval/run_context_inference.py \
        --contexts npo/data/contexts.jsonl \
        --model-id Qwen/Qwen3-4B-Instruct-2507 \
        --vllm-url http://localhost:8000/v1 \
        --n-per-ticker 10 \
        --seed 42 \
        --max-workers 50 \
        --output results.jsonl
"""

import argparse
import json
import random
import re
import os
import concurrent.futures
from collections import defaultdict
from tqdm.auto import tqdm

from llm_clients import VLLMClient


def parse_json_from_text(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def run_single(client, prompt):
    try:
        response = client.get_response(prompt)
        parsed = parse_json_from_text(response)
        decision = None
        if parsed and "decision" in parsed:
            d = parsed["decision"].lower().strip()
            if d in ("buy", "sell"):
                decision = d
        return response, decision
    except Exception as e:
        return f"Error: {e}", None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1")
    parser.add_argument("--n-per-ticker", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-workers", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # Load contexts
    print(f"Loading contexts from {args.contexts}...")
    ticker_contexts = defaultdict(list)
    with open(args.contexts) as f:
        for line in f:
            d = json.loads(line)
            ticker_contexts[d["ticker"]].append(d)

    # Sample n_per_ticker per ticker
    random.seed(args.seed)
    sampled = []
    for ticker in sorted(ticker_contexts.keys()):
        contexts = ticker_contexts[ticker]
        n = min(args.n_per_ticker, len(contexts))
        selected = random.sample(contexts, n)
        sampled.extend(selected)

    print(f"총 {len(ticker_contexts)}개 종목, 종목당 {args.n_per_ticker}개 → {len(sampled)}개 인퍼런스")

    # Create client
    client = VLLMClient(
        model_id=args.model_id,
        base_url=args.vllm_url,
        temperature=args.temperature,
        seed=args.seed,
    )

    # Run inference
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(run_single, client, ctx["prompt"]): ctx
            for ctx in sampled
        }
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Inference"):
            ctx = futures[future]
            response, decision = future.result()
            results.append({
                "ticker": ctx["ticker"],
                "name": ctx["name"],
                "sector": ctx["sector"],
                "marketcap": ctx["marketcap"],
                "context_id": ctx["context_id"],
                "prompt": ctx["prompt"],
                "llm_output": response,
                "llm_answer": decision,
            })

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Summary
    total = len(results)
    valid = sum(1 for r in results if r["llm_answer"] is not None)
    buy = sum(1 for r in results if r["llm_answer"] == "buy")
    sell = sum(1 for r in results if r["llm_answer"] == "sell")
    print(f"\n결과: {args.output}")
    print(f"총 {total}, 유효 {valid} ({valid/total*100:.1f}%), NaN {total-valid}")
    if valid > 0:
        print(f"Buy {buy} ({buy/valid*100:.1f}%), Sell {sell} ({sell/valid*100:.1f}%)")
    else:
        print(f"Buy {buy}, Sell {sell} (no valid responses)")


if __name__ == "__main__":
    main()
