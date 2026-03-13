"""
Build retain_v2.jsonl for forget_v2 tickers.
Prompt = 1 buy evidence + 2 sell evidences (sell:buy = 2:1 ratio in evidence).
Combinations: 4 buy × C(4,2) sell = 24 per ticker × 21 tickers = 504.
Runs Qwen3-4B via vLLM on GPU 1.
"""

import csv
import json
import re
import time
import subprocess
import sys
import os
from itertools import combinations, product
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

# ─── Config ───
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
VLLM_PORT = 8001
GPU_ID = 1
MAX_WORKERS = 128
TEMPERATURE = 0.6
SEED = 42
MAX_TOKENS = 512

DATA_DIR = Path(__file__).parent
ROOT_DATA = DATA_DIR.parents[1] / "data"
OUTPUT_FILE = DATA_DIR / "retain_v2.jsonl"

PROMPT_TEMPLATE = (
    "Refer to the evidence below to make a final investment decision for the given stock.\n"
    "Stock Ticker: [{ticker}]\n"
    "Stock Name: [{name}]\n"
    "--- Evidence ---\n"
    "{evidence_block}\n"
    "---\n"
    "Your final response must be a single, valid JSON object. "
    "The JSON object must contain the following two keys:\n"
    "\"decision\": [buy | sell]\n"
    "\"reason\": A brief justification for your decision\n"
    "Do not include \"hold\" as an option. You must choose one side. "
    "Your response should start with {{ and end with }}. "
    "Do not include any other text."
)


def parse_json(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def load_evidence(csv_path):
    data = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            t = row["ticker"]
            op = row["opinion"]
            if t not in data:
                data[t] = {"buy": [], "sell": []}
            data[t][op].append(row["evidence1"])
            data[t][op].append(row["evidence2"])
    return data


def load_sp500(csv_path):
    meta = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            meta[row["ticker"]] = {
                "name": row["name"],
                "sector": row["sector"],
                "marketcap": int(row["marketcap"]),
            }
    return meta


def start_vllm():
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
    conda_lib = os.path.join(sys.prefix, "lib", "libstdc++.so.6")
    if os.path.exists(conda_lib):
        env["LD_PRELOAD"] = conda_lib

    proc = subprocess.Popen(
        [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", MODEL_ID,
            "--port", str(VLLM_PORT),
            "--tensor-parallel-size", "1",
            "--trust-remote-code",
            "--dtype", "bfloat16",
            "--max-model-len", "4096",
            "--gpu-memory-utilization", "0.90",
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(f"vLLM starting (PID={proc.pid}, GPU={GPU_ID}, port={VLLM_PORT})...")

    import urllib.request
    for i in range(120):
        try:
            urllib.request.urlopen(f"http://localhost:{VLLM_PORT}/v1/models", timeout=2)
            print(f"vLLM ready ({i*5}s)")
            return proc
        except Exception:
            if proc.poll() is not None:
                raise RuntimeError("vLLM process died")
            time.sleep(5)
    raise RuntimeError("vLLM timeout")


def main():
    # Load forget tickers
    forget_tickers = set()
    with open(DATA_DIR / "forget_v2.jsonl") as f:
        for line in f:
            forget_tickers.add(json.loads(line)["ticker"])
    print(f"Forget tickers: {len(forget_tickers)}")

    # Load evidence
    qual = load_evidence(DATA_DIR / "evidence_corpus_qual_mixed.csv")
    quant = load_evidence(DATA_DIR / "evidence_corpus_quant_mixed.csv")
    evidence = {"qual": qual, "quant": quant}
    meta = load_sp500(ROOT_DATA / "sp500_final.csv")

    # Build all evidence options
    # (type, idx): qual_1, qual_2, quant_1, quant_2
    ev_keys = [("qual", 0), ("qual", 1), ("quant", 0), ("quant", 1)]

    # Build prompts: 1 buy × 2 sell combinations
    tasks = []
    combo_id = 0
    for ticker in sorted(forget_tickers):
        info = meta[ticker]
        buy_options = ev_keys  # pick 1
        sell_pairs = list(combinations(ev_keys, 2))  # pick 2, C(4,2)=6

        for buy_key in buy_options:
            buy_type, buy_idx = buy_key
            buy_ev = evidence[buy_type][ticker]["buy"][buy_idx]

            for sell_pair in sell_pairs:
                sell_evs = []
                sell_meta = []
                for s_type, s_idx in sell_pair:
                    sell_evs.append(evidence[s_type][ticker]["sell"][s_idx])
                    sell_meta.append(f"{s_type}_{s_idx+1}")

                # Evidence block: randomize order (buy among sells)
                # Fixed order: sell, buy, sell for balance
                ev_lines = [
                    f"1. {sell_evs[0]}",
                    f"2. {buy_ev}",
                    f"3. {sell_evs[1]}",
                ]
                evidence_block = "\n".join(ev_lines)

                prompt = PROMPT_TEMPLATE.format(
                    ticker=ticker,
                    name=info["name"],
                    evidence_block=evidence_block,
                )

                tasks.append({
                    "ticker": ticker,
                    "name": info["name"],
                    "sector": info["sector"],
                    "marketcap": info["marketcap"],
                    "combo_id": combo_id,
                    "buy_type": buy_type,
                    "buy_ev_idx": buy_idx + 1,
                    "sell_keys": ",".join(sell_meta),
                    "prompt": prompt,
                })
                combo_id += 1

    print(f"Total prompts: {len(tasks)} ({len(tasks)//len(forget_tickers)} per ticker)")

    # Start vLLM and run inference
    vllm_proc = start_vllm()

    try:
        client = OpenAI(base_url=f"http://localhost:{VLLM_PORT}/v1", api_key="EMPTY")

        results = []
        done = 0
        t0 = time.time()

        def do_inference(idx, task):
            resp = client.chat.completions.create(
                model=MODEL_ID,
                messages=[{"role": "user", "content": task["prompt"]}],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                top_p=0.8,
                seed=SEED,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            raw = resp.choices[0].message.content.strip()
            parsed = parse_json(raw)
            decision = parsed.get("decision", "").lower() if parsed else None
            return idx, raw, decision

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(do_inference, i, t): i for i, t in enumerate(tasks)}
            for fut in as_completed(futures):
                idx, raw, decision = fut.result()
                results.append((idx, raw, decision))
                done += 1
                if done % 100 == 0 or done == len(tasks):
                    elapsed = time.time() - t0
                    print(f"  {done}/{len(tasks)}  ({elapsed:.0f}s)")

        # Sort and write
        results.sort(key=lambda x: x[0])

        records = []
        for (idx, raw, decision), task in zip(results, tasks):
            rec = {
                "context": task["prompt"],
                "completion": raw,
                "dataset_type": "retain",
                "ticker": task["ticker"],
                "sector": task["sector"],
                "marketcap": task["marketcap"],
                "combo_id": task["combo_id"],
                "buy_type": task["buy_type"],
                "buy_ev_idx": task["buy_ev_idx"],
                "sell_keys": task["sell_keys"],
                "llm_answer": decision,
                "id": f"retain_{idx:05d}",
            }
            records.append(rec)

        with open(OUTPUT_FILE, "w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # Summary
        from collections import Counter
        decisions = Counter(r["llm_answer"] for r in records)
        total = len(records)
        print(f"\nretain_v2.jsonl: {total} records")
        for d, c in decisions.most_common():
            print(f"  {d}: {c} ({100*c/total:.1f}%)")

        # Per-ticker breakdown
        from collections import defaultdict
        ticker_stats = defaultdict(Counter)
        for r in records:
            ticker_stats[r["ticker"]][r["llm_answer"]] += 1
        print(f"\nPer-ticker sell:buy ratio:")
        for t in sorted(ticker_stats):
            s = ticker_stats[t]
            print(f"  {t}: sell={s.get('sell',0)} buy={s.get('buy',0)}")

    finally:
        print("\nStopping vLLM...")
        vllm_proc.terminate()
        try:
            vllm_proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            vllm_proc.kill()


if __name__ == "__main__":
    main()
