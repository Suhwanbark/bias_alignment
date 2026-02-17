"""
Generate counterfactual retain set by anonymizing ticker identity and
re-inferring with base model (vLLM offline batch).

Pipeline:
  1. Load forget.jsonl (high-bias ticker prompts)
  2. Anonymize each prompt (remove ticker/company identity)
  3. Infer with base model on anonymized prompts (N samples per prompt, temperature 1.0)
  4. Filter for target decision (default: sell)
  5. Save retain_counterfactual.jsonl (original context + counterfactual completion)

Usage:
    python -m new_npo.prepare_counterfactual \
        --forget-path new_npo/data/qwen3-30b-a3b-instruct-2507/forget.jsonl \
        --sp500-path data/sp500_final.csv \
        --output-path new_npo/data/qwen3-30b-a3b-instruct-2507/retain_counterfactual.jsonl \
        --num-samples 10 --temperature 1.0
"""

import argparse
import csv
import json
import os
import re
from typing import Optional


# ---------------------------------------------------------------------------
# Anonymization
# ---------------------------------------------------------------------------

def anonymize_prompt(context: str, ticker: str, name: str) -> str:
    """
    Remove ticker identity from prompt to enable evidence-only judgment.

    Replaces:
      - [TICKER] -> [STOCK]
      - Stock Name: [Name] -> Stock Name: [Anonymous Company]
      - Company name mentions -> "the company"
      - TICKER's -> "the company's"
    """
    context = context.replace(f"[{ticker}]", "[STOCK]")
    context = context.replace(f"Stock Name: [{name}]", "Stock Name: [Anonymous Company]")
    context = context.replace(name, "the company")
    context = context.replace(f"{ticker}'s", "the company's")
    context = re.sub(rf'\b{re.escape(ticker)}\b', 'the company', context)
    return context


# ---------------------------------------------------------------------------
# Decision parsing
# ---------------------------------------------------------------------------

def parse_decision(completion: str) -> Optional[str]:
    """Extract decision (buy/sell) from completion text."""
    m = re.search(r'"decision"\s*:\s*"(buy|sell)"', completion, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    return None


# ---------------------------------------------------------------------------
# Inference (vLLM offline batch)
# ---------------------------------------------------------------------------

def infer_batch(
    prompts: list[str],
    model_id: str,
    temperature: float = 1.0,
    max_tokens: int = 150,
    seed: int = 42,
    tensor_parallel: int = 2,
    num_samples: int = 10,
) -> list[list[str]]:
    """
    Offline batch inference using vLLM LLM class.
    Returns list of lists: each prompt gets num_samples completions.
    """
    from vllm import LLM, SamplingParams

    print(f"  Loading vLLM LLM: {model_id} (TP={tensor_parallel})")
    llm = LLM(
        model=model_id,
        tensor_parallel_size=tensor_parallel,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
        seed=seed,
    )
    tokenizer = llm.get_tokenizer()

    # Build chat-templated prompts
    templated = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        templated.append(text)

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        n=num_samples,
    )

    print(f"  Running vLLM offline inference on {len(templated)} prompts x {num_samples} samples...")
    outputs = llm.generate(templated, sampling_params)

    results = []
    for out in outputs:
        completions = [o.text.strip() for o in out.outputs]
        results.append(completions)

    # Free GPU
    del llm
    import torch, gc
    gc.collect()
    torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Counterfactual retain builder
# ---------------------------------------------------------------------------

def build_counterfactual_retain(
    forget_samples: list[dict],
    completions_per_prompt: list[list[str]],
    anon_prompts: list[str],
    filter_decision: str = "sell",
) -> list[dict]:
    """
    Build counterfactual retain samples with decision filtering.

    For each forget sample, filters N completions to keep only those
    matching filter_decision. Keeps one completion per sample to avoid
    over-representation of specific tickers.
    """
    cf_retain = []
    stats = {"total_prompts": len(forget_samples), "matched": 0, "no_match": 0}

    for i, (sample, completions, anon_prompt) in enumerate(
        zip(forget_samples, completions_per_prompt, anon_prompts)
    ):
        # Find first completion matching the target decision
        matched_completion = None
        for comp in completions:
            if comp.startswith("ERROR:"):
                continue
            decision = parse_decision(comp)
            if decision == filter_decision:
                matched_completion = comp
                break

        if matched_completion is None:
            stats["no_match"] += 1
            continue

        stats["matched"] += 1
        cf_retain.append({
            "context": sample["context"],
            "completion": matched_completion,
            "dataset_type": "retain",
            "retain_source": "counterfactual",
            "ticker": sample["ticker"],
            "bias_score": sample["bias_score"],
            "sector": sample.get("sector", ""),
            "marketcap": sample.get("marketcap", 0),
            "anonymized_context": anon_prompt,
            "id": f"retain_cf_{i:05d}",
        })

    print(f"  Filter decision: {filter_decision}")
    print(f"  Matched: {stats['matched']}/{stats['total_prompts']} "
          f"({stats['matched']/stats['total_prompts']*100:.1f}%)")
    print(f"  No match: {stats['no_match']}/{stats['total_prompts']}")

    return cf_retain


# ---------------------------------------------------------------------------
# S&P 500 name lookup
# ---------------------------------------------------------------------------

def load_ticker_names(sp500_path: str) -> dict[str, str]:
    """Load ticker -> company name mapping from sp500_final.csv."""
    names = {}
    with open(sp500_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            names[row["ticker"]] = row["name"]
    return names


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate counterfactual retain set")
    parser.add_argument("--forget-path", required=True, help="Path to forget.jsonl")
    parser.add_argument("--sp500-path", default="data/sp500_final.csv", help="Path to sp500_final.csv")
    parser.add_argument("--model-id", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--output-path", required=True, help="Output path for retain_counterfactual.jsonl")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tensor-parallel", type=int, default=2, help="Tensor parallel size")
    parser.add_argument("--num-samples", type=int, default=10,
                        help="Number of completions per prompt (higher = more sell candidates)")
    parser.add_argument("--filter-decision", default="sell", choices=["sell", "buy", "none"],
                        help="Keep only completions with this decision (default: sell)")
    args = parser.parse_args()

    # Load forget samples
    print(f"[1/4] Loading forget samples: {args.forget_path}")
    forget_samples = []
    with open(args.forget_path) as f:
        for line in f:
            forget_samples.append(json.loads(line))
    print(f"  Loaded {len(forget_samples)} forget samples")

    # Load ticker names
    print(f"\n[2/4] Loading ticker names: {args.sp500_path}")
    ticker_names = load_ticker_names(args.sp500_path)
    print(f"  Loaded {len(ticker_names)} ticker names")

    # Anonymize prompts
    print(f"\n[3/4] Anonymizing prompts and running inference...")
    anon_prompts = []
    for sample in forget_samples:
        ticker = sample["ticker"]
        name = ticker_names.get(ticker, ticker)
        anon = anonymize_prompt(sample["context"], ticker, name)
        anon_prompts.append(anon)

    # Run inference
    print(f"  Prompts: {len(anon_prompts)}")
    print(f"  Samples per prompt: {args.num_samples}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Total inferences: {len(anon_prompts) * args.num_samples}")

    completions = infer_batch(
        anon_prompts,
        model_id=args.model_id,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
        tensor_parallel=args.tensor_parallel,
        num_samples=args.num_samples,
    )

    # Stats
    error_count = sum(
        1 for comps in completions
        for c in comps if c.startswith("ERROR:")
    )
    total_completions = sum(len(comps) for comps in completions)

    sell_counts = []
    for comps in completions:
        n_sell = sum(1 for c in comps if parse_decision(c) == "sell")
        sell_counts.append(n_sell)
    prompts_with_sell = sum(1 for n in sell_counts if n > 0)
    total_sell = sum(sell_counts)

    print(f"  Inference complete: {total_completions} total completions, {error_count} errors")
    print(f"  Sell completions: {total_sell}/{total_completions} "
          f"({total_sell/total_completions*100:.1f}%)")
    print(f"  Prompts with >=1 sell: {prompts_with_sell}/{len(completions)} "
          f"({prompts_with_sell/len(completions)*100:.1f}%)")

    # Build counterfactual retain set
    print(f"\n[4/4] Building counterfactual retain set")
    if args.filter_decision == "none":
        # No filtering: keep first valid completion per prompt
        flat_completions = []
        for comps in completions:
            valid = [c for c in comps if not c.startswith("ERROR:")]
            flat_completions.append(valid[0] if valid else "ERROR: all samples failed")

        cf_retain = []
        for i, (sample, completion, anon_prompt) in enumerate(
            zip(forget_samples, flat_completions, anon_prompts)
        ):
            if completion.startswith("ERROR:"):
                continue
            cf_retain.append({
                "context": sample["context"],
                "completion": completion,
                "dataset_type": "retain",
                "retain_source": "counterfactual",
                "ticker": sample["ticker"],
                "bias_score": sample["bias_score"],
                "sector": sample.get("sector", ""),
                "marketcap": sample.get("marketcap", 0),
                "anonymized_context": anon_prompt,
                "id": f"retain_cf_{i:05d}",
            })
    else:
        cf_retain = build_counterfactual_retain(
            forget_samples, completions, anon_prompts,
            filter_decision=args.filter_decision,
        )
    print(f"  Counterfactual retain: {len(cf_retain)} samples")

    # Write output
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        for entry in cf_retain:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"\n  -> {args.output_path}: {len(cf_retain)} samples")

    # Per-ticker summary
    ticker_counts = {}
    for entry in cf_retain:
        t = entry["ticker"]
        ticker_counts[t] = ticker_counts.get(t, 0) + 1

    # Summary
    forget_tickers = set(s["ticker"] for s in forget_samples)
    print(f"\n{'='*50}")
    print(f"COUNTERFACTUAL RETAIN SUMMARY")
    print(f"{'='*50}")
    print(f"Forget samples:        {len(forget_samples)}")
    print(f"Samples per prompt:    {args.num_samples}")
    print(f"Temperature:           {args.temperature}")
    print(f"Filter decision:       {args.filter_decision}")
    print(f"Total inferences:      {total_completions}")
    print(f"Inference errors:      {error_count}")
    print(f"Counterfactual retain: {len(cf_retain)} samples")
    print(f"Coverage:              {len(ticker_counts)}/{len(forget_tickers)} tickers")
    print(f"Output:                {args.output_path}")
    if ticker_counts:
        print(f"\nPer-ticker breakdown:")
        for t in sorted(ticker_counts):
            print(f"  {t}: {ticker_counts[t]} samples")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
