"""
Build retain_v2.jsonl for forget_v2 tickers with sell:buy = 2:1 ratio.
- 6 combos per ticker from initial_retain.jsonl
- 4 sell (natural model response) + 2 buy (system-prompted)
- Uses Qwen3-4B via vLLM on GPU 1
"""

import json
import re
import time
import subprocess
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

# ─── Config ───
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
VLLM_PORT = 8001
GPU_ID = 1
MAX_WORKERS = 64
TEMPERATURE = 0.6
SEED = 42
MAX_TOKENS = 512

DATA_DIR = Path(__file__).parent
OUTPUT_FILE = DATA_DIR / "retain_v2.jsonl"

BUY_SYSTEM_PROMPT = (
    "You are a financial analyst. When the evidence is balanced between positive and negative signals, "
    "you tend to focus on the upside potential and recommend buying. "
    "Always respond with a valid JSON object containing 'decision' and 'reason' keys."
)


def parse_json(text: str):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


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


def call_llm(client, prompt, system_prompt=None):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    resp = client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        top_p=0.8,
        seed=SEED,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return resp.choices[0].message.content.strip()


def main():
    # Load forget tickers
    forget_tickers = set()
    with open(DATA_DIR / "forget_v2.jsonl") as f:
        for line in f:
            forget_tickers.add(json.loads(line)["ticker"])

    # Load retain prompts for forget tickers
    retain_by_ticker = {}
    with open(DATA_DIR / "initial_retain.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["ticker"] in forget_tickers:
                retain_by_ticker.setdefault(r["ticker"], []).append(r)

    # Assign roles: first 4 = sell (natural), last 2 = buy (system-prompted)
    sell_tasks = []  # (record, system_prompt)
    buy_tasks = []
    for ticker in sorted(retain_by_ticker):
        combos = retain_by_ticker[ticker]  # 6 combos, sorted by context_id
        combos.sort(key=lambda x: x["context_id"])
        for i, rec in enumerate(combos):
            if i < 4:
                sell_tasks.append((rec, None))
            else:
                buy_tasks.append((rec, BUY_SYSTEM_PROMPT))

    all_tasks = sell_tasks + buy_tasks
    print(f"Tasks: {len(sell_tasks)} sell + {len(buy_tasks)} buy = {len(all_tasks)} total")

    # Start vLLM
    vllm_proc = start_vllm()

    try:
        client = OpenAI(base_url=f"http://localhost:{VLLM_PORT}/v1", api_key="EMPTY")
        results = []
        done = 0
        t0 = time.time()

        def do_inference(task_idx, record, sys_prompt):
            target = "buy" if sys_prompt else "sell"
            raw = call_llm(client, record["prompt"], sys_prompt)
            parsed = parse_json(raw)
            decision = parsed.get("decision", "").lower() if parsed else None

            # Retry buy tasks if model still outputs sell (up to 3 times)
            if target == "buy" and decision != "buy":
                for retry in range(3):
                    raw = call_llm(client, record["prompt"], sys_prompt)
                    parsed = parse_json(raw)
                    decision = parsed.get("decision", "").lower() if parsed else None
                    if decision == "buy":
                        break

            return task_idx, record, raw, parsed, decision, target

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {}
            for i, (rec, sys_p) in enumerate(all_tasks):
                futures[pool.submit(do_inference, i, rec, sys_p)] = i

            for fut in as_completed(futures):
                task_idx, record, raw, parsed, decision, target = fut.result()
                results.append((task_idx, record, raw, parsed, decision, target))
                done += 1
                if done % 50 == 0 or done == len(all_tasks):
                    print(f"  {done}/{len(all_tasks)}  ({time.time()-t0:.0f}s)")

        # Sort and build output
        results.sort(key=lambda x: x[0])

        output_records = []
        idx = 0
        buy_ok = 0
        buy_total = 0
        for task_idx, record, raw, parsed, decision, target in results:
            out = {
                "context": record["prompt"],
                "completion": raw,
                "dataset_type": "retain",
                "ticker": record["ticker"],
                "sector": record["sector"],
                "marketcap": record["marketcap"],
                "context_id": record["context_id"],
                "target_decision": target,
                "llm_answer": decision,
                "id": f"retain_{idx:05d}",
            }
            output_records.append(out)
            idx += 1
            if target == "buy":
                buy_total += 1
                if decision == "buy":
                    buy_ok += 1

        with open(OUTPUT_FILE, "w") as f:
            for r in output_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # Summary
        sells = sum(1 for r in output_records if r["llm_answer"] == "sell")
        buys = sum(1 for r in output_records if r["llm_answer"] == "buy")
        print(f"\nretain_v2.jsonl: {len(output_records)} records")
        print(f"  Sell: {sells}  Buy: {buys}  Ratio: {sells}:{buys}")
        print(f"  Buy-prompted success: {buy_ok}/{buy_total}")

    finally:
        print("Stopping vLLM...")
        vllm_proc.terminate()
        try:
            vllm_proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            vllm_proc.kill()


if __name__ == "__main__":
    main()
