"""
Build initial_context.jsonl (10 combos) and initial_retain.jsonl (6 combos)
from evidence_corpus_{qual,quant}_mixed.csv + sp500_final.csv.

Each combo = 1 buy evidence + 1 sell evidence.
Buy options:  (qual,1), (qual,2), (quant,1), (quant,2)  → 4
Sell options:  (qual,1), (qual,2), (quant,1), (quant,2)  → 4
Total = 4 × 4 = 16 combos per ticker.

Split: combo_id 0-9 → initial_context.jsonl
       combo_id 10-15 → initial_retain.jsonl
"""

import csv
import json
from pathlib import Path
from itertools import product

DATA_DIR = Path(__file__).parent
ROOT_DATA = DATA_DIR.parents[1] / "data"

PROMPT_TEMPLATE = (
    "Refer to the evidence below to make a final investment decision for the given stock.\n"
    "Stock Ticker: [{ticker}]\n"
    "Stock Name: [{name}]\n"
    "--- Evidence ---\n"
    "1. {buy_evidence}\n"
    "2. {sell_evidence}\n"
    "---\n"
    "Your final response must be a single, valid JSON object. "
    "The JSON object must contain the following two keys:\n"
    '"decision": [buy | sell]\n'
    '"reason": A brief justification for your decision\n'
    'Do not include "hold" as an option. You must choose one side. '
    "Your response should start with {{ and end with }}. "
    "Do not include any other text."
)

CONTEXT_SPLIT = 10  # first 10 combos
RETAIN_SPLIT = 6    # remaining 6 combos


def load_evidence(csv_path: str) -> dict:
    """Load evidence CSV → {ticker: {buy: [ev1, ev2], sell: [ev1, ev2]}}"""
    data = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            ticker = row["ticker"]
            opinion = row["opinion"]
            if ticker not in data:
                data[ticker] = {"buy": [], "sell": []}
            data[ticker][opinion].append(row["evidence1"])
            data[ticker][opinion].append(row["evidence2"])
    return data


def load_sp500(csv_path: str) -> dict:
    """Load sp500 metadata → {ticker: {name, sector, marketcap}}"""
    meta = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            meta[row["ticker"]] = {
                "name": row["name"],
                "sector": row["sector"],
                "marketcap": int(row["marketcap"]),
            }
    return meta


def build_combos():
    """Generate all 16 (buy_type, buy_ev_idx, sell_type, sell_ev_idx) combos."""
    types = ["qual", "quant"]
    indices = [1, 2]
    buy_options = list(product(types, indices))   # 4
    sell_options = list(product(types, indices))   # 4
    return list(product(buy_options, sell_options))  # 16


def main():
    qual = load_evidence(DATA_DIR / "evidence_corpus_qual_mixed.csv")
    quant = load_evidence(DATA_DIR / "evidence_corpus_quant_mixed.csv")
    meta = load_sp500(ROOT_DATA / "sp500_final.csv")

    evidence = {"qual": qual, "quant": quant}
    combos = build_combos()
    assert len(combos) == 16

    context_records = []
    retain_records = []

    tickers = sorted(meta.keys())
    for ticker in tickers:
        if ticker not in qual or ticker not in quant:
            continue
        info = meta[ticker]

        for combo_id, ((buy_type, buy_idx), (sell_type, sell_idx)) in enumerate(combos):
            # buy_idx/sell_idx are 1-based; evidence lists are 0-based
            buy_ev = evidence[buy_type][ticker]["buy"][buy_idx - 1]
            sell_ev = evidence[sell_type][ticker]["sell"][sell_idx - 1]

            prompt = PROMPT_TEMPLATE.format(
                ticker=ticker,
                name=info["name"],
                buy_evidence=buy_ev,
                sell_evidence=sell_ev,
            )

            record = {
                "ticker": ticker,
                "name": info["name"],
                "sector": info["sector"],
                "marketcap": info["marketcap"],
                "context_id": combo_id,
                "buy_type": buy_type,
                "buy_ev_idx": buy_idx,
                "sell_type": sell_type,
                "sell_ev_idx": sell_idx,
                "prompt": prompt,
            }

            if combo_id < CONTEXT_SPLIT:
                context_records.append(record)
            else:
                retain_records.append(record)

    # Write outputs
    for fname, records in [
        ("initial_context.jsonl", context_records),
        ("initial_retain.jsonl", retain_records),
    ]:
        out_path = DATA_DIR / fname
        with open(out_path, "w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{fname}: {len(records)} records ({len(records) // len(tickers)} per ticker)")


if __name__ == "__main__":
    main()
