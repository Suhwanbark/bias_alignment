"""
Run Qwen3-4B inference on initial_context.jsonl via vLLM.
Saves results to initial_context_result.jsonl.
"""

import json
import re
import time
import subprocess
import sys
import os
import signal
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
INPUT_FILE = DATA_DIR / "initial_context.jsonl"
OUTPUT_FILE = DATA_DIR / "initial_context_result.jsonl"


def parse_json(text: str):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def start_vllm():
    """Start vLLM server on GPU 1, return process."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
    # Fix libstdc++ version issue in conda env
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

    # Wait for ready
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
    raise RuntimeError("vLLM failed to start (timeout 600s)")


def run_inference(client, record):
    """Run single inference, return record with result."""
    try:
        resp = client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": record["prompt"]}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            top_p=0.8,
            seed=SEED,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        raw = resp.choices[0].message.content.strip()
        parsed = parse_json(raw)

        result = {**record}
        del result["prompt"]  # save space
        result["raw_response"] = raw
        result["decision"] = parsed.get("decision", "").lower() if parsed else None
        result["reason"] = parsed.get("reason", "") if parsed else None
        result["parse_ok"] = parsed is not None
        return result
    except Exception as e:
        result = {**record}
        del result["prompt"]
        result["raw_response"] = str(e)
        result["decision"] = None
        result["reason"] = None
        result["parse_ok"] = False
        return result


def main():
    # Load data
    records = []
    with open(INPUT_FILE) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"Loaded {len(records)} records from {INPUT_FILE.name}")

    # Start vLLM
    vllm_proc = start_vllm()

    try:
        client = OpenAI(base_url=f"http://localhost:{VLLM_PORT}/v1", api_key="EMPTY")

        results = []
        done = 0
        t0 = time.time()

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(run_inference, client, r): i for i, r in enumerate(records)}
            for fut in as_completed(futures):
                results.append((futures[fut], fut.result()))
                done += 1
                if done % 500 == 0 or done == len(records):
                    elapsed = time.time() - t0
                    print(f"  {done}/{len(records)}  ({elapsed:.0f}s, {done/elapsed:.1f} rec/s)")

        # Sort by original order and write
        results.sort(key=lambda x: x[0])
        with open(OUTPUT_FILE, "w") as f:
            for _, r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # Summary
        total = len(results)
        parsed = sum(1 for _, r in results if r["parse_ok"])
        buys = sum(1 for _, r in results if r["decision"] == "buy")
        sells = sum(1 for _, r in results if r["decision"] == "sell")
        elapsed = time.time() - t0
        print(f"\nDone: {total} records in {elapsed:.0f}s")
        print(f"  Parse OK: {parsed}/{total} ({100*parsed/total:.1f}%)")
        print(f"  Buy: {buys} ({100*buys/total:.1f}%)  Sell: {sells} ({100*sells/total:.1f}%)")
        print(f"  Saved to {OUTPUT_FILE.name}")

    finally:
        print("Stopping vLLM...")
        vllm_proc.terminate()
        try:
            vllm_proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            vllm_proc.kill()


if __name__ == "__main__":
    main()
