"""
In-Distribution evaluation for BiasUnlearn debiased models (vLLM offline).

Loads LoRA adapter directly via vLLM LLM class, runs bias measurement,
and optionally compares with base model results.

Usage:
    python -m new_npo.eval.eval_id \
        --model-dir /home/jovyan/models/qwen-biasunlearn-a75r25 \
        --result-name biasunlearn_a75r25 \
        --profile-report new_npo/data/qwen3-30b-a3b-instruct-2507/profile_report.json
"""

import argparse
import json
import os
import sys
import gc


def run_bias_measurement(
    model_dir: str,
    result_name: str,
    base_model: str = "Qwen/Qwen3-30B-A3B-Instruct-2507",
    num_trials: int = 10,
    seed: int = 42,
    tp: int = 2,
) -> str:
    """
    Run bias measurement using vLLM offline inference with LoRA.

    Returns path to the result directory.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    scripts_dir = os.path.join(repo_root, "new_npo", "scripts")
    sys.path.insert(0, scripts_dir)

    from run_eval_local import generate_prompts, compute_bias_index_from_results

    result_dir = os.path.join(repo_root, "local", result_name, "result")
    os.makedirs(result_dir, exist_ok=True)

    # Generate prompts
    prompts, metadata = generate_prompts(
        os.path.join(repo_root, "data", "sp500_final.csv"),
        os.path.join(repo_root, "data", "evidence_corpus_qual_mixed.csv"),
        os.path.join(repo_root, "data", "evidence_corpus_quant_mixed.csv"),
        num_trials=num_trials,
        seed=seed,
    )
    print(f"  {len(prompts)} prompts for {len(set(m['ticker'] for m in metadata))} tickers")

    # Load vLLM with LoRA
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    import glob

    sampling_params = SamplingParams(temperature=0.6, max_tokens=1024, seed=seed)

    print(f"\nLoading base model: {base_model} (TP={tp})")
    llm = LLM(
        model=base_model,
        tensor_parallel_size=tp,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
        seed=seed,
        enable_lora=True,
        max_lora_rank=16,
        disable_custom_all_reduce=True,
    )
    tokenizer = llm.get_tokenizer()

    # Build templated prompts
    templated = []
    for p in prompts:
        messages = [{"role": "user", "content": p}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        templated.append(text)

    # Find checkpoints and evaluate
    ckpts = []
    for d in sorted(glob.glob(os.path.join(model_dir, "checkpoint-*"))):
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "adapter_model.safetensors")):
            step = int(d.split("checkpoint-")[-1])
            ckpts.append((step, d))
    ckpts.sort()

    summary_path = os.path.join(result_dir, "summary.log")
    with open(summary_path, "w") as sf:
        for step, ckpt_path in ckpts:
            step_dir = os.path.join(result_dir, f"step_{step}")

            # Check cache
            existing = glob.glob(os.path.join(step_dir, "*_att_result.json"))
            if existing:
                try:
                    bi = json.load(open(existing[0]))['bias_index']
                    print(f"  step {step}: cached (bias_index={bi})")
                    sf.write(f"step={step}  bias_index={bi}  status=cached\n")
                    sf.flush()
                    continue
                except:
                    pass

            print(f"\n  -- step {step} --")
            lora_req = LoRARequest("adapter", 1, ckpt_path)
            outputs = llm.generate(templated, sampling_params, lora_request=lora_req)
            results = [out.outputs[0].text.strip() for out in outputs]

            bias_index, df = compute_bias_index_from_results(results, metadata)
            print(f"  * Step {step} -> Bias Index: {bias_index}")

            os.makedirs(step_dir, exist_ok=True)
            df.to_csv(os.path.join(step_dir, f"step_{step}_att_combined.csv"), index=False)

            result_json = {
                'bias_index': bias_index, 'step': step,
                'checkpoint': ckpt_path, 'num_tickers': len(df['ticker'].unique()),
                'num_prompts': len(results),
            }
            with open(os.path.join(step_dir, f"step_{step}_att_result.json"), 'w') as f:
                json.dump(result_json, f, indent=2)

            sf.write(f"step={step}  bias_index={bias_index}  status=ok\n")
            sf.flush()

    # Cleanup
    del llm
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except:
        pass

    return result_dir


def compare_before_after(
    base_result_path: str,
    debiased_result_path: str,
    profile_report_path: str,
) -> dict:
    """
    Compare base and debiased bias measurement results.

    Returns dict with comparison metrics:
      - bias_index_before, bias_index_after, reduction
      - high_bias_ticker analysis
      - sector/size gap changes
    """
    with open(base_result_path) as f:
        base = json.load(f)
    with open(debiased_result_path) as f:
        debiased = json.load(f)

    bi_before = base["bias_index"]
    bi_after = debiased["bias_index"]
    reduction = (bi_before - bi_after) / bi_before * 100 if bi_before > 0 else 0

    # Sector gap comparison
    base_sectors = base.get("sector_stats", {})
    debiased_sectors = debiased.get("sector_stats", {})

    base_sector_means = [v["bias_mean"] for v in base_sectors.values()]
    debiased_sector_means = [v["bias_mean"] for v in debiased_sectors.values()]

    import numpy as np
    base_sector_std = np.std(base_sector_means) if base_sector_means else 0
    debiased_sector_std = np.std(debiased_sector_means) if debiased_sector_means else 0

    comparison = {
        "bias_index_before": bi_before,
        "bias_index_after": bi_after,
        "reduction_pct": round(reduction, 1),
        "sector_std_before": round(float(base_sector_std), 1),
        "sector_std_after": round(float(debiased_sector_std), 1),
        "base_result": base_result_path,
        "debiased_result": debiased_result_path,
    }

    # Load profile report for high-bias ticker analysis
    if os.path.exists(profile_report_path):
        with open(profile_report_path) as f:
            profile = json.load(f)
        comparison["high_bias_tickers"] = profile["summary"]["high_bias_tickers"]
        comparison["low_bias_tickers"] = profile["summary"]["low_bias_tickers"]

    return comparison


def main():
    parser = argparse.ArgumentParser(description="In-Distribution evaluation (vLLM offline)")
    parser.add_argument("--model-dir", required=True, help="Model directory with checkpoint-* subdirs")
    parser.add_argument("--result-name", required=True, help="Result name for output")
    parser.add_argument("--base-model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--profile-report", default=None, help="Path to profile_report.json")
    parser.add_argument("--base-result", default=None, help="Base model result JSON for comparison")
    parser.add_argument("--num-trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tp", type=int, default=2)
    args = parser.parse_args()

    result_dir = run_bias_measurement(
        model_dir=args.model_dir,
        result_name=args.result_name,
        base_model=args.base_model,
        num_trials=args.num_trials,
        seed=args.seed,
        tp=args.tp,
    )

    print(f"\nResult directory: {result_dir}")

    # Compare with base if provided
    if args.base_result and args.profile_report:
        import glob
        debiased_result = glob.glob(os.path.join(result_dir, "**/*_att_result.json"), recursive=True)
        if debiased_result:
            comparison = compare_before_after(
                args.base_result, debiased_result[0], args.profile_report
            )
            comp_path = os.path.join(result_dir, "comparison.json")
            with open(comp_path, "w") as f:
                json.dump(comparison, f, indent=2)
            print(f"\n{'='*50}")
            print(f"COMPARISON RESULTS")
            print(f"{'='*50}")
            print(f"Bias Index: {comparison['bias_index_before']} -> {comparison['bias_index_after']}")
            print(f"Reduction:  {comparison['reduction_pct']}%")
            print(f"Sector Std: {comparison['sector_std_before']} -> {comparison['sector_std_after']}")
            print(f"{'='*50}")


if __name__ == "__main__":
    main()
