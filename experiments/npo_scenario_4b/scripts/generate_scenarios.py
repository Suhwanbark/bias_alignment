"""
Phase 1+2: Generate bearish scenarios using Qwen3-4B, then build forget set.

1) Qwen3-4B generates bearish scenario text for each target ticker
2) Qwen3-4B makes buy/sell decisions on those scenarios
3) SELL responses → forget.jsonl (biased behavior to unlearn)

Usage:
    # Start vLLM first:  eval/vllm qwen-4b   (or manually on port 8001)
    python experiments/npo_scenario_4b/scripts/generate_scenarios.py \
        --vllm-url http://localhost:8001/v1
"""

import argparse
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from openai import OpenAI
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "experiments" / "npo_scenario_4b" / "data"

# 34 target tickers (from npo/data/qwen4b/forget.jsonl)
TARGET_TICKERS = [
    "A", "ADI", "AMP", "APH", "BAX", "C", "CAG", "CCI", "CDW", "CFG",
    "CHTR", "CINF", "CSX", "EG", "FANG", "FRT", "GD", "GL", "HCA", "HLT",
    "INTU", "IVZ", "J", "JNPR", "KMI", "KMX", "LEN", "LH", "O", "PGR",
    "PSA", "REG", "UPS", "VLO",
]

SCENARIO_TEMPLATES = [
    {
        "type": "news_brief",
        "system": "You are a financial news reporter writing breaking news.",
        "prompt": (
            "Write a short, realistic breaking news article (2-3 paragraphs, ~150 words) "
            "about {name} ({ticker}), a company in the {sector} sector.\n\n"
            "The news should describe a clearly NEGATIVE development — for example: "
            "regulatory trouble, earnings miss, management scandal, product failure, "
            "competitive threat, or macro headwind.\n\n"
            "Include specific but fictional details (dollar amounts, percentages, dates). "
            "Write in a neutral journalistic tone. Do NOT mention buy/sell recommendations."
        ),
    },
    {
        "type": "analyst_downgrade",
        "system": "You are a sell-side equity analyst writing a research note.",
        "prompt": (
            "Write a short analyst downgrade note (2-3 paragraphs, ~150 words) for "
            "{name} ({ticker}) in the {sector} sector.\n\n"
            "Include: a price target cut, the key thesis for the downgrade, "
            "and 2-3 specific risk factors with numbers. "
            "Write in formal analyst style. Do NOT include a buy/sell/hold rating label."
        ),
    },
    {
        "type": "earnings_miss",
        "system": "You are a financial journalist covering quarterly earnings.",
        "prompt": (
            "Write a short earnings report summary (2-3 paragraphs, ~150 words) for "
            "{name} ({ticker}) in the {sector} sector.\n\n"
            "The report should describe a DISAPPOINTING quarter: revenue miss, "
            "margin compression, lowered guidance, or weak outlook. "
            "Include specific fictional numbers (EPS, revenue, growth rates). "
            "Write factually. Do NOT mention buy/sell."
        ),
    },
    {
        "type": "sector_risk",
        "system": "You are a macro strategist writing a sector risk report.",
        "prompt": (
            "Write a short sector risk analysis (2-3 paragraphs, ~150 words) focused on "
            "{name} ({ticker}) in the {sector} sector.\n\n"
            "Describe sector-wide headwinds that specifically threaten this company: "
            "regulatory changes, demand shifts, supply chain issues, or technological disruption. "
            "Include specific data points. Do NOT mention buy/sell."
        ),
    },
    {
        "type": "bear_case",
        "system": "You are a contrarian investment researcher writing a bear case.",
        "prompt": (
            "Write a short bear case thesis (2-3 paragraphs, ~150 words) for "
            "{name} ({ticker}) in the {sector} sector.\n\n"
            "Present 3 compelling reasons why the stock could decline significantly. "
            "Include valuation concerns, fundamental weaknesses, and/or catalysts for decline. "
            "Use specific numbers. Do NOT include a rating or recommendation."
        ),
    },
]

# Decision prompt — completely different format from eval's evidence-based prompt
DECISION_PROMPT = (
    "Read the following financial information about {ticker} ({name}) "
    "and make an investment decision.\n\n"
    "---\n"
    "{scenario_text}\n"
    "---\n\n"
    "Based on the above, would you buy or sell {ticker} stock?\n"
    'Respond with a JSON object: {{"decision": "buy" or "sell", '
    '"reason": "brief justification"}}'
)


def load_sp500():
    df = pd.read_csv(ROOT / "data" / "sp500_final.csv")
    return {row["ticker"]: row for _, row in df.iterrows()}


def create_client(vllm_url):
    return OpenAI(base_url=vllm_url, api_key="dummy")


def get_model_id(client):
    return client.models.list().data[0].id


def call_llm(client, model_id, system, user, temperature=0.8, max_tokens=512, seed=42):
    try:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": user})
        resp = client.chat.completions.create(
            model=model_id,
            messages=msgs,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return None


def parse_decision(text):
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        parsed = json.loads(m.group(0))
        d = parsed.get("decision", "").lower().strip()
        return d if d in ("buy", "sell") else None
    except json.JSONDecodeError:
        return None


# ── Phase 1: Generate scenarios ─────────────────────────────────────────

def phase1(client, model_id, sp500, output_dir, variations=4):
    path = output_dir / "scenarios.jsonl"
    if path.exists():
        data = [json.loads(l) for l in open(path)]
        print(f"Phase 1: {len(data)} scenarios already exist, skip.")
        return data

    tasks = []
    for ticker in TARGET_TICKERS:
        if ticker not in sp500:
            continue
        info = sp500[ticker]
        for tpl in SCENARIO_TEMPLATES:
            for v in range(variations):
                tasks.append((info, tpl, v))

    print(f"Phase 1: Generating {len(tasks)} scenarios "
          f"({len(TARGET_TICKERS)} tickers x {len(SCENARIO_TEMPLATES)} templates x {variations} var)...")

    scenarios = []
    with ThreadPoolExecutor(max_workers=32) as pool:
        futs = {}
        for info, tpl, v in tasks:
            prompt = tpl["prompt"].format(
                name=info["name"], ticker=info["ticker"], sector=info["sector"],
            )
            fut = pool.submit(
                call_llm, client, model_id, tpl["system"], prompt,
                temperature=0.8, max_tokens=512, seed=42 + v,
            )
            futs[fut] = (info, tpl, v)

        for fut in tqdm(as_completed(futs), total=len(futs), desc="Scenarios"):
            info, tpl, v = futs[fut]
            text = fut.result()
            if text:
                scenarios.append({
                    "ticker": info["ticker"],
                    "name": info["name"],
                    "sector": info["sector"],
                    "marketcap": int(info["marketcap"]),
                    "template_type": tpl["type"],
                    "variation": v,
                    "scenario_text": text,
                })

    with open(path, "w") as f:
        for s in scenarios:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Phase 1 done: {len(scenarios)} scenarios → {path}")
    return scenarios


# ── Phase 2: Decision + forget set ──────────────────────────────────────

def phase2(client, model_id, scenarios, output_dir):
    forget_path = output_dir / "forget.jsonl"

    print(f"Phase 2: Running {len(scenarios)} decisions...")

    results = []
    with ThreadPoolExecutor(max_workers=64) as pool:
        futs = {}
        for i, s in enumerate(scenarios):
            prompt = DECISION_PROMPT.format(
                ticker=s["ticker"], name=s["name"], scenario_text=s["scenario_text"],
            )
            fut = pool.submit(
                call_llm, client, model_id, None, prompt,
                temperature=0.6, max_tokens=256, seed=42,
            )
            futs[fut] = (i, s, prompt)

        for fut in tqdm(as_completed(futs), total=len(futs), desc="Decisions"):
            i, s, prompt = futs[fut]
            output = fut.result()
            decision = parse_decision(output)
            if decision and output:
                results.append({
                    "context": prompt,
                    "completion": output,
                    "ticker": s["ticker"],
                    "name": s["name"],
                    "sector": s["sector"],
                    "marketcap": s["marketcap"],
                    "template_type": s["template_type"],
                    "variation": s["variation"],
                    "llm_answer": decision,
                })

    forget = [r for r in results if r["llm_answer"] == "sell"]
    buy = [r for r in results if r["llm_answer"] == "buy"]

    print(f"\nResults: {len(results)} valid ({len(forget)} SELL / {len(buy)} BUY)")
    print(f"\nPer-ticker:")
    sell_ct = Counter(r["ticker"] for r in forget)
    buy_ct = Counter(r["ticker"] for r in buy)
    for t in sorted(TARGET_TICKERS):
        print(f"  {t:>6s}: {sell_ct.get(t,0):>3d} SELL / {buy_ct.get(t,0):>3d} BUY")

    for i, r in enumerate(forget):
        r["id"] = f"forget_{i:05d}"
        r["dataset_type"] = "forget"

    with open(forget_path, "w") as f:
        for r in forget:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nforget.jsonl: {len(forget)} samples → {forget_path}")

    # Save all for analysis
    with open(output_dir / "all_decisions.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return forget


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-url", default="http://localhost:8001/v1")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--variations", type=int, default=4)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    client = create_client(args.vllm_url)
    model_id = get_model_id(client)
    print(f"Model: {model_id}")

    sp500 = load_sp500()

    # Save target list
    with open(args.output_dir / "target_tickers.json", "w") as f:
        json.dump(TARGET_TICKERS, f, indent=2)

    scenarios = phase1(client, model_id, sp500, args.output_dir, args.variations)
    phase2(client, model_id, scenarios, args.output_dir)


if __name__ == "__main__":
    main()
