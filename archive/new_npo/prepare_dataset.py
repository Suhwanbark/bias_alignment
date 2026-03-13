"""
Prepare Forget / Retain datasets for BiasUnlearn from Phase 0 profiling results.

Extends npo/prepare_dataset.py with:
  - anonymize_prompt(): ticker identity removal for counterfactual generation
  - build_adversarial_forget(): mix 25% retain into forget set
  - merge_counterfactual_retain(): merge counterfactual completions into retain set

Usage:
    python -m new_npo.prepare_dataset \
        --csv new_npo/bias_profiling/qwen3-30b-a3b-instruct-2507_att_combined.csv \
        --threshold 80 \
        --neutral-threshold 20 \
        --output-dir new_npo/data/qwen3-30b-a3b-instruct-2507

Output:
    data/qwen3-30b-a3b-instruct-2507/
        forget.jsonl          # 고편향 종목의 다수방향 응답 (+ adversarial 25%)
        retain.jsonl          # 소수방향 + 저편향 + counterfactual 병합
        eval_mini.jsonl       # Dynamic swapping 평가용 소규모 셋
        profile_report.json   # 종목별 bias_score + 선별 결과
"""

import argparse
import csv
import json
import os
import re
import random
from collections import defaultdict


# ---------------------------------------------------------------------------
# Bias score computation (from npo/prepare_dataset.py)
# ---------------------------------------------------------------------------

def compute_bias_scores(rows: list[dict]) -> dict[str, float]:
    """종목별 bias_score 계산: ((buy - sell) / (buy + sell)) * 100"""
    ticker_counts = defaultdict(lambda: {"buy": 0, "sell": 0})
    for r in rows:
        ans = r["llm_answer"].lower().strip()
        if ans in ("buy", "sell"):
            ticker_counts[r["ticker"]][ans] += 1

    scores = {}
    for t, c in ticker_counts.items():
        total = c["buy"] + c["sell"]
        if total > 0:
            scores[t] = ((c["buy"] - c["sell"]) / total) * 100
    return scores


def classify_tickers(
    scores: dict[str, float],
    threshold: float,
    neutral_threshold: float,
) -> dict[str, list[str]]:
    """종목을 고편향/저편향/중간으로 분류"""
    high_bias = [t for t, s in scores.items() if abs(s) > threshold]
    low_bias = [t for t, s in scores.items() if abs(s) < neutral_threshold]
    mid_bias = [
        t
        for t, s in scores.items()
        if neutral_threshold <= abs(s) <= threshold
    ]
    return {
        "high_bias": sorted(high_bias),
        "low_bias": sorted(low_bias),
        "mid_bias": sorted(mid_bias),
    }


# ---------------------------------------------------------------------------
# Dataset builders (from npo/prepare_dataset.py)
# ---------------------------------------------------------------------------

def build_forget_set(
    rows: list[dict],
    scores: dict[str, float],
    high_bias_tickers: list[str],
) -> list[dict]:
    """고편향 종목의 다수방향 응답 -> Forget Set"""
    high_set = set(high_bias_tickers)
    forget = []
    for r in rows:
        t = r["ticker"]
        if t not in high_set:
            continue
        ans = r["llm_answer"].lower().strip()
        if ans not in ("buy", "sell"):
            continue
        # 다수방향만 선택
        majority = "buy" if scores[t] > 0 else "sell"
        if ans == majority:
            forget.append(
                {
                    "context": r["prompt"],
                    "completion": r["llm_output"],
                    "dataset_type": "forget",
                    "ticker": t,
                    "bias_score": scores[t],
                    "sector": r["sector"],
                    "marketcap": int(r["marketcap"]),
                    "trial": int(r["trial"]),
                    "set": int(r["set"]),
                    "llm_answer": ans,
                }
            )
    return forget


def build_retain_set(
    rows: list[dict],
    scores: dict[str, float],
    high_bias_tickers: list[str],
    low_bias_tickers: list[str],
) -> list[dict]:
    """소수방향 응답 + 저편향 종목 응답 -> Retain Set"""
    high_set = set(high_bias_tickers)
    low_set = set(low_bias_tickers)
    retain = []

    for r in rows:
        t = r["ticker"]
        ans = r["llm_answer"].lower().strip()
        if ans not in ("buy", "sell"):
            continue

        entry = {
            "context": r["prompt"],
            "completion": r["llm_output"],
            "dataset_type": "retain",
            "ticker": t,
            "bias_score": scores.get(t, 0),
            "sector": r["sector"],
            "marketcap": int(r["marketcap"]),
            "trial": int(r["trial"]),
            "set": int(r["set"]),
            "llm_answer": ans,
        }

        if t in high_set:
            # 고편향 종목의 소수방향
            majority = "buy" if scores[t] > 0 else "sell"
            if ans != majority:
                entry["retain_source"] = "minority"
                retain.append(entry)
        elif t in low_set:
            # 저편향 종목의 모든 응답
            entry["retain_source"] = "neutral"
            retain.append(entry)

    return retain


def build_eval_mini(
    rows: list[dict],
    scores: dict[str, float],
    high_bias_tickers: list[str],
    n_per_group: int = 25,
) -> list[dict]:
    """Dynamic swapping 평가용 소규모 셋 -- 고편향 종목에서 균등 샘플링"""
    buy_biased = [t for t in high_bias_tickers if scores[t] > 0]
    sell_biased = [t for t in high_bias_tickers if scores[t] < 0]

    random.seed(42)
    sampled_buy = random.sample(buy_biased, min(n_per_group, len(buy_biased)))
    sampled_sell = random.sample(sell_biased, min(n_per_group, len(sell_biased)))
    sampled = set(sampled_buy + sampled_sell)

    eval_set = []
    # 종목당 첫 번째 trial만 사용 (평가 시 빠른 반복 위해)
    for r in rows:
        if r["ticker"] in sampled and r["trial"] == "0":
            eval_set.append(
                {
                    "context": r["prompt"],
                    "completion": r["llm_output"],
                    "dataset_type": "eval",
                    "ticker": r["ticker"],
                    "bias_score": scores.get(r["ticker"], 0),
                    "sector": r["sector"],
                    "llm_answer": r["llm_answer"].lower().strip(),
                }
            )
    return eval_set


# ---------------------------------------------------------------------------
# New functions for BiasUnlearn extensions
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


def build_adversarial_forget(forget: list[dict], retain: list[dict]) -> list[dict]:
    """
    Mix 25% of retain samples into forget set to prevent bias polarity reversal.
    (BiasUnlearn Appendix A.2)
    """
    n_mix = len(retain) // 4
    adversarial = forget + retain[:n_mix]
    random.shuffle(adversarial)
    return adversarial


def merge_counterfactual_retain(retain: list[dict], cf_path: str) -> list[dict]:
    """
    Load retain_counterfactual.jsonl and merge into retain set.

    Counterfactual samples have original context + base model's anonymized completion.
    """
    if not os.path.exists(cf_path):
        print(f"  [WARN] Counterfactual file not found: {cf_path}, skipping merge")
        return retain

    cf_samples = []
    with open(cf_path) as f:
        for line in f:
            cf_samples.append(json.loads(line))

    print(f"  Merging {len(cf_samples)} counterfactual samples into retain set")
    merged = retain + cf_samples
    return merged


# ---------------------------------------------------------------------------
# I/O utilities (from npo/prepare_dataset.py)
# ---------------------------------------------------------------------------

def write_jsonl(data: list[dict], path: str):
    """JSONL 파일 저장 (id 필드 자동 부여)"""
    with open(path, "w", encoding="utf-8") as f:
        for i, entry in enumerate(data):
            if "id" not in entry:
                entry["id"] = f"{entry['dataset_type']}_{i:05d}"
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"  -> {path}: {len(data)} samples")


def write_profile_report(
    scores: dict[str, float],
    groups: dict[str, list[str]],
    rows: list[dict],
    forget_count: int,
    retain_count: int,
    threshold: float,
    neutral_threshold: float,
    path: str,
):
    """종목별 bias_score + 선별 결과 리포트"""
    # ticker metadata
    ticker_meta = {}
    for r in rows:
        t = r["ticker"]
        if t not in ticker_meta:
            ticker_meta[t] = {"sector": r["sector"], "marketcap": int(r["marketcap"])}

    report = {
        "config": {
            "threshold": threshold,
            "neutral_threshold": neutral_threshold,
        },
        "summary": {
            "total_tickers": len(scores),
            "high_bias_tickers": len(groups["high_bias"]),
            "low_bias_tickers": len(groups["low_bias"]),
            "mid_bias_tickers": len(groups["mid_bias"]),
            "forget_samples": forget_count,
            "retain_samples": retain_count,
        },
        "ticker_scores": {
            t: {
                "bias_score": scores[t],
                "group": (
                    "high_bias"
                    if t in set(groups["high_bias"])
                    else "low_bias"
                    if t in set(groups["low_bias"])
                    else "mid_bias"
                ),
                "sector": ticker_meta.get(t, {}).get("sector", ""),
                "marketcap": ticker_meta.get(t, {}).get("marketcap", 0),
            }
            for t in sorted(scores, key=lambda x: abs(scores[x]), reverse=True)
        },
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  -> {path}: profile report")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepare debias datasets from Phase 0 profiling")
    parser.add_argument("--csv", required=True, help="Phase 0 combined CSV")
    parser.add_argument("--threshold", type=float, default=80, help="High-bias threshold (default: 80)")
    parser.add_argument("--neutral-threshold", type=float, default=20, help="Neutral/low-bias threshold (default: 20)")
    parser.add_argument("--output-dir", required=True, help="Output directory for JSONL files")
    parser.add_argument("--counterfactual", default=None, help="Path to retain_counterfactual.jsonl (optional)")
    parser.add_argument("--adversarial", action="store_true", help="Build adversarial forget set (mix 25%% retain)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    # --- Load Phase 0 CSV ---
    print(f"[1/5] Loading Phase 0 CSV: {args.csv}")
    rows = []
    with open(args.csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    print(f"  Loaded {len(rows)} rows, {len(set(r['ticker'] for r in rows))} tickers")

    # --- Compute bias scores ---
    print(f"\n[2/5] Computing per-ticker bias scores")
    scores = compute_bias_scores(rows)
    groups = classify_tickers(scores, args.threshold, args.neutral_threshold)

    print(f"  Threshold: |bias_score| > {args.threshold}")
    print(f"  고편향 종목: {len(groups['high_bias'])}개")
    buy_biased = [t for t in groups["high_bias"] if scores[t] > 0]
    sell_biased = [t for t in groups["high_bias"] if scores[t] < 0]
    print(f"    - Buy 편향: {len(buy_biased)}개")
    print(f"    - Sell 편향: {len(sell_biased)}개")
    print(f"  저편향 종목: {len(groups['low_bias'])}개")
    print(f"  중간: {len(groups['mid_bias'])}개")

    # --- Build datasets ---
    print(f"\n[3/5] Building Forget Set (고편향 다수방향)")
    forget = build_forget_set(rows, scores, groups["high_bias"])
    print(f"  Forget: {len(forget)} samples")

    print(f"\n[4/5] Building Retain Set (소수방향 + 저편향)")
    retain = build_retain_set(rows, scores, groups["high_bias"], groups["low_bias"])
    minority_count = sum(1 for r in retain if r.get("retain_source") == "minority")
    neutral_count = sum(1 for r in retain if r.get("retain_source") == "neutral")
    print(f"  Retain: {len(retain)} samples")
    print(f"    - 소수방향 (고편향 종목): {minority_count}")
    print(f"    - 중립 (저편향 종목): {neutral_count}")

    # Merge counterfactual retain if provided
    if args.counterfactual:
        retain = merge_counterfactual_retain(retain, args.counterfactual)
        cf_count = sum(1 for r in retain if r.get("retain_source") == "counterfactual")
        print(f"    - Counterfactual: {cf_count}")

    # Build adversarial forget set if requested
    if args.adversarial:
        print(f"\n  Building adversarial forget set (25% retain mixing)")
        forget = build_adversarial_forget(forget, retain)
        print(f"  Adversarial forget: {len(forget)} samples")

    print(f"\n[5/5] Building Eval Mini Set")
    eval_mini = build_eval_mini(rows, scores, groups["high_bias"])
    print(f"  Eval Mini: {len(eval_mini)} samples")

    # --- Write outputs ---
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nWriting to {args.output_dir}/")
    write_jsonl(forget, os.path.join(args.output_dir, "forget.jsonl"))
    write_jsonl(retain, os.path.join(args.output_dir, "retain.jsonl"))
    write_jsonl(eval_mini, os.path.join(args.output_dir, "eval_mini.jsonl"))
    write_profile_report(
        scores,
        groups,
        rows,
        len(forget),
        len(retain),
        args.threshold,
        args.neutral_threshold,
        os.path.join(args.output_dir, "profile_report.json"),
    )

    # --- Summary ---
    print(f"\n{'='*50}")
    print(f"DATASET SUMMARY")
    print(f"{'='*50}")
    print(f"Model:          {os.path.basename(args.csv).replace('_att_combined.csv', '')}")
    print(f"Threshold:      |bias_score| > {args.threshold}")
    print(f"고편향 종목:     {len(groups['high_bias'])}개 (buy {len(buy_biased)}, sell {len(sell_biased)})")
    print(f"저편향 종목:     {len(groups['low_bias'])}개")
    print(f"")
    print(f"Forget:         {len(forget):,} samples" + (" (adversarial)" if args.adversarial else ""))
    print(f"Retain:         {len(retain):,} samples (minority {minority_count} + neutral {neutral_count})")
    print(f"Eval Mini:      {len(eval_mini)} samples")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
