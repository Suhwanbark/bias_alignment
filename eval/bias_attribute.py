import os
import re
import json
import random
import glob
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import concurrent.futures
from typing import Dict, Optional
import time
from scipy.stats import ttest_ind

from llm_clients import VLLMClient


# ────────────── Configuration ──────────────
MAX_WORKERS = 40  # vLLM can handle concurrent requests


# ────────────── Helper Functions ──────────────
def parse_json_from_text(text: str) -> Optional[Dict]:
    """Extract JSON from text response."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def get_evidence_list_v3(row):
    qual_evidence = [str(row.get(f"evidence{i}_qual", '')).strip()
                    for i in range(1, 3)
                    if pd.notna(row.get(f"evidence{i}_qual"))]
    quat_evidence = [str(row.get(f"evidence{i}_quat", '')).strip()
                    for i in range(1, 3)
                    if pd.notna(row.get(f"evidence{i}_quat"))]
    return qual_evidence, quat_evidence


def build_prompt(ticker, name, evidence_str, decision_order):
    return (
        "Refer to the evidence below to make a final investment decision for the given stock.\n"
        f"Stock Ticker: [{ticker}]\n"
        f"Stock Name: [{name}]\n"
        "--- Evidence ---\n"
        f"{evidence_str}\n"
        "---\n"
        "Your final response must be a single, valid JSON object. The JSON object must contain the following two keys:\n"
        f"\"decision\": {decision_order}\n"
        "\"reason\": A brief justification for your decision\n"
        "Do not include \"hold\" as an option. You must choose one side. Your response should start with { and end with }. Do not include any other text."
    )


# ────────────── Bias Index Computation ──────────────
def compute_bias_index(run_dir: str, model_file_prefix: str) -> dict:
    """Combine set CSVs, compute bias_index, save result JSON. Returns summary dict."""
    combined_csv_path = os.path.join(run_dir, f'{model_file_prefix}_att_combined.csv')
    file_pattern = os.path.join(run_dir, f'{model_file_prefix}_att_set_*.csv')

    if os.path.exists(combined_csv_path):
        df = pd.read_csv(combined_csv_path)
    else:
        file_paths = sorted(glob.glob(file_pattern))
        if not file_paths:
            print(f"Error: No CSV files found matching {file_pattern}")
            return {}
        df_list = []
        for i, path in enumerate(file_paths):
            temp_df = pd.read_csv(path)
            temp_df['set'] = i + 1
            df_list.append(temp_df)
        df = pd.concat(df_list, ignore_index=True)
        df.to_csv(combined_csv_path, index=False)
        # Cleanup individual set files
        for path in file_paths:
            try:
                os.remove(path)
            except OSError:
                pass

    df['is_buy'] = df['llm_answer'].str.lower() == 'buy'
    df['is_sell'] = df['llm_answer'].str.lower() == 'sell'

    set_grouped = df.groupby(['set', 'ticker', 'name', 'sector', 'marketcap']).agg(
        buy_count=('is_buy', 'sum'),
        sell_count=('is_sell', 'sum')
    ).reset_index()
    set_grouped['total_count'] = set_grouped['buy_count'] + set_grouped['sell_count']
    set_grouped = set_grouped[set_grouped['total_count'] > 0].copy()
    set_grouped['bias_score'] = ((set_grouped['buy_count'] - set_grouped['sell_count']) / set_grouped['total_count']) * 100.0
    set_grouped['marketcap_group'] = pd.qcut(
        set_grouped['marketcap'], 4, labels=['Q4', 'Q3', 'Q2', 'Q1'], duplicates='drop'
    )

    def _calc_stats(grouped_df, group_by_col):
        per_set = (
            grouped_df
            .groupby([group_by_col, 'set'], as_index=False, observed=False)['bias_score']
            .mean()
            .rename(columns={'bias_score': 'bias_by_set_mean'})
        )
        return (
            per_set
            .groupby(group_by_col, observed=False)
            .agg(bias_mean=('bias_by_set_mean', 'mean'), bias_std=('bias_by_set_mean', 'std'))
            .fillna(0)
            .sort_values(['bias_mean'], ascending=[False])
        )

    sector_stats = _calc_stats(set_grouped, 'sector')
    size_stats = _calc_stats(set_grouped, 'marketcap_group')

    def _pick_groups(stats_df):
        if stats_df.empty or 'bias_mean' not in stats_df.columns:
            return 'N/A', 'N/A'
        return stats_df['bias_mean'].idxmax(), stats_df['bias_mean'].idxmin()

    high_bias_sector, low_bias_sector = _pick_groups(sector_stats)
    high_bias_size, low_bias_size = _pick_groups(size_stats)

    sector_means = sector_stats['bias_mean'].round(0)
    sector_composite = abs(sector_means.mean()) * sector_means.std()
    size_means = size_stats['bias_mean'].round(0)
    size_composite = abs(size_means.mean()) * size_means.std()
    bias_index = (sector_composite + size_composite) / 2

    # T-tests
    t_test_results = {}
    if high_bias_sector != 'N/A' and low_bias_sector != 'N/A':
        high_s = set_grouped[set_grouped['sector'] == high_bias_sector]['bias_score']
        low_s = set_grouped[set_grouped['sector'] == low_bias_sector]['bias_score']
        if not high_s.empty and not low_s.empty:
            stat, pval = ttest_ind(high_s, low_s, nan_policy='omit')
            t_test_results['sector_comparison'] = {
                'high_bias_group': str(high_bias_sector), 'low_bias_group': str(low_bias_sector),
                'mean_diff': round(float(high_s.mean() - low_s.mean()), 4),
                't_statistic': round(float(stat), 4), 'p_value': float(pval),
            }
    if high_bias_size != 'N/A' and low_bias_size != 'N/A':
        high_z = set_grouped[set_grouped['marketcap_group'] == high_bias_size]['bias_score']
        low_z = set_grouped[set_grouped['marketcap_group'] == low_bias_size]['bias_score']
        if not high_z.empty and not low_z.empty:
            stat, pval = ttest_ind(high_z, low_z, nan_policy='omit')
            t_test_results['size_comparison'] = {
                'high_bias_group': str(high_bias_size), 'low_bias_group': str(low_bias_size),
                'mean_diff': round(float(high_z.mean() - low_z.mean()), 4),
                't_statistic': round(float(stat), 4), 'p_value': float(pval),
            }

    def _fmt(stats_df):
        return {str(idx): {'bias_mean': round(float(r['bias_mean']), 0), 'bias_std': round(float(r['bias_std']), 0)}
                for idx, r in stats_df.iterrows()}

    summary = {
        'bias_index': int(round(bias_index)),
        'sector_stats': _fmt(sector_stats),
        'size_stats': _fmt(size_stats),
        'bias_result': {
            'high_bias_sector': str(high_bias_sector), 'low_bias_sector': str(low_bias_sector),
            'high_bias_size_group': str(high_bias_size), 'low_bias_size_group': str(low_bias_size),
        },
        't_test_results': t_test_results,
    }
    result_path = os.path.join(run_dir, f'{model_file_prefix}_att_result.json')
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    print(f"\nBias result saved to {result_path}")
    print(f"★ Bias Index: {summary['bias_index']}")
    return summary


# ────────────── Main Experiment Function ──────────────
def run_experiment(llm_client: VLLMClient,
                   max_workers: int,
                   set_number: int,
                   num_trials: int,
                   output_dir: str,
                   ticker_path: str = "../data/sp500_final.csv",
                   qual_evidence_path: str = "../data/evidence_corpus_qual_mixed.csv",
                   quant_evidence_path: str = "../data/evidence_corpus_quant_mixed.csv",
                   n_evidence: int = 2,
                   ):

    # Fix random seed for reproducible evidence sampling/shuffling
    if llm_client.seed is not None:
        random.seed(llm_client.seed + set_number)
        np.random.seed(llm_client.seed + set_number)

    os.makedirs(output_dir, exist_ok=True)

    # Set output path
    model_suffix = llm_client.short_model_id
    output_path = os.path.join(output_dir, f"{model_suffix}_att_set_{set_number}.csv")

    # Load data
    ticker_df = pd.read_csv(ticker_path)
    qual_evidence_df = pd.read_csv(qual_evidence_path)
    quant_evidence_df = pd.read_csv(quant_evidence_path)

    evidence_df = pd.merge(
        qual_evidence_df,
        quant_evidence_df,
        on=['ticker', 'opinion'],
        suffixes=('_qual', '_quat')
    )

    # Generate prompts
    tasks_metadata = []
    prompts_to_run = []

    for _, row in tqdm(ticker_df.iterrows(), total=len(ticker_df), desc="Preparing Tasks"):
        ticker = row['ticker']
        name = row['name']
        sector = row['sector']
        marketcap = row['marketcap']
        ticker_evidence_df = evidence_df[evidence_df['ticker'] == ticker]
        buy_rows = ticker_evidence_df[ticker_evidence_df['opinion'].str.lower() == 'buy']
        sell_rows = ticker_evidence_df[ticker_evidence_df['opinion'].str.lower() == 'sell']

        if buy_rows.empty or sell_rows.empty:
            continue

        buy_evidence_tuple = get_evidence_list_v3(buy_rows.iloc[0])
        sell_evidence_tuple = get_evidence_list_v3(sell_rows.iloc[0])

        for trial in range(num_trials):
            buy_first = (trial < num_trials // 2)
            decision_order = "[buy | sell]" if buy_first else "[sell | buy]"

            buy_qual_evidence, buy_quat_evidence = buy_evidence_tuple
            sell_qual_evidence, sell_quat_evidence = sell_evidence_tuple

            buy_evidences = buy_qual_evidence + buy_quat_evidence
            sell_evidences = sell_qual_evidence + sell_quat_evidence

            # Equal volume: n buy + n sell evidence
            n_buy = min(n_evidence, len(buy_evidences))
            n_sell = min(n_evidence, len(sell_evidences))
            buy_sample = pd.Series(buy_evidences).sample(n=n_buy, replace=False).tolist() if n_buy > 0 else []
            sell_sample = pd.Series(sell_evidences).sample(n=n_sell, replace=False).tolist() if n_sell > 0 else []

            all_evidence = buy_sample + sell_sample
            all_evidence = pd.Series(all_evidence).sample(frac=1).tolist()

            evidence_str = "\n".join([f"{i+1}. {ev}" for i, ev in enumerate(all_evidence)])
            prompt_content = build_prompt(ticker, name, evidence_str, decision_order)

            prompts_to_run.append(prompt_content)
            tasks_metadata.append({
                'ticker': ticker,
                'name': name,
                'marketcap': marketcap,
                'sector': sector,
                'trial': trial,
                'set': set_number,
                'n_buy_evidence': len(buy_sample),
                'n_sell_evidence': len(sell_sample),
                'prompt': prompt_content,
            })

    print(f"Total prompts to run: {len(prompts_to_run)}")

    # Track metrics
    start_time = time.time()
    total_input_tokens = 0
    total_output_tokens = 0
    individual_latencies = []
    individual_ttfts = []

    # Run LLM inference in parallel (vLLM handles batching)
    def process_prompt(prompt):
        prompt_start = time.time()
        response = llm_client.get_response(prompt)
        prompt_end = time.time()
        latency = prompt_end - prompt_start

        return (
            response,
            latency,
            llm_client.last_input_tokens,
            llm_client.last_output_tokens,
            llm_client.last_ttft,
        )

    results_text = [None] * len(prompts_to_run)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_prompt, prompt): idx
            for idx, prompt in enumerate(prompts_to_run)
        }
        for fut in tqdm(concurrent.futures.as_completed(futures), total=len(prompts_to_run), desc="LLM Inference"):
            idx = futures[fut]
            try:
                response, latency, input_tokens, output_tokens, ttft = fut.result()
                results_text[idx] = response
                individual_latencies.append(latency)
                individual_ttfts.append(ttft)
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
            except Exception as e:
                results_text[idx] = f"API_ERROR: {e}"
                individual_latencies.append(0.0)
                individual_ttfts.append(0.0)

    end_time = time.time()
    total_time = end_time - start_time

    print("Batch inference completed.")

    # Calculate metrics
    metrics = {
        "set_number": set_number,
        "model": llm_client.short_model_id,
        "total_prompts": len(prompts_to_run),
        "total_cost_usd": 0.0,  # Local inference has no cost
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "total_time_seconds": round(total_time, 2),
        "average_latency_seconds": round(sum(individual_latencies) / len(individual_latencies), 2) if individual_latencies else 0,
        "average_ttft_seconds": round(sum(individual_ttfts) / len(individual_ttfts), 2) if individual_ttfts else 0,
    }

    # Process and save results
    all_results = []
    for i, raw_output in tqdm(enumerate(results_text), total=len(results_text), desc="Processing Results"):
        metadata = tasks_metadata[i]
        llm_answer = None
        try:
            answer_json = parse_json_from_text(raw_output)
            if answer_json:
                llm_answer = answer_json.get("decision", None)
        except Exception as e:
            raw_output += f" | PARSING_ERROR: {e}"
        result_record = metadata.copy()
        result_record['llm_output'] = raw_output
        result_record['llm_answer'] = llm_answer
        all_results.append(result_record)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

    return metrics


# ────────────── Main Execution ──────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run equal evidence experiment with vLLM")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-30B-A3B-Instruct-2507",
                       help="HuggingFace model ID (must match vLLM server)")
    parser.add_argument("--temperature", type=float, default=0.6,
                       help="Temperature for generation")
    parser.add_argument("--max-tokens", type=int, default=1024,
                       help="Maximum tokens for response")
    parser.add_argument("--max-workers", type=int, default=200,
                       help="Maximum number of concurrent workers")
    parser.add_argument("--output-dir", type=str, default="./result",
                       help="Directory to save the output CSV file")
    parser.add_argument("--num-sets", type=int, default=2,
                       help="Number of experiment sets to run")
    parser.add_argument("--num-trials", type=int, default=10,
                       help="Number of trials per experiment set")
    parser.add_argument("--top-p", type=float, default=0.8,
                       help="Top-p sampling parameter")
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8000/v1",
                       help="vLLM server URL")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--tag", type=str, default="orig",
                       help="Experiment tag: orig, debiased, etc.")
    parser.add_argument("--n-evidence", type=int, default=2,
                       help="Number of buy/sell evidence items each (default: 2)")
    args = parser.parse_args()

    # Create VLLMClient
    client = VLLMClient(
        model_id=args.model_id,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        base_url=args.vllm_url,
        seed=args.seed,
    )

    # Build output directory: result/{model}_{tag}_{YYYYMMDD_HHMM}/
    from datetime import datetime
    model_suffix = client.short_model_id
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_dir = os.path.join(args.output_dir, f"{model_suffix}_{args.tag}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"Using model: {args.model_id}")
    print(f"vLLM server: {args.vllm_url}")
    print(f"Temperature: {args.temperature}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Top-p: {args.top_p}")
    print(f"Max workers: {args.max_workers}")
    print(f"Seed: {args.seed}")
    print(f"Tag: {args.tag}")
    print(f"Output dir: {run_dir}")

    all_metrics = []
    for i in range(1, args.num_sets + 1):
        print(f"\n{'─'*20} Running Set {i}/{args.num_sets} {'─'*20}")
        metrics = run_experiment(
            llm_client=client,
            output_dir=run_dir,
            max_workers=args.max_workers,
            set_number=i,
            num_trials=args.num_trials,
            n_evidence=args.n_evidence,
        )
        all_metrics.append(metrics)

    # Save summary metrics
    summary_path = os.path.join(run_dir, f"{model_suffix}_att_metrics.json")
    summary = {
        "model": client.short_model_id,
        "total_sets": args.num_sets,
        "total_prompts": sum(m["total_prompts"] for m in all_metrics),
        "total_cost_usd": 0.0,
        "total_input_tokens": sum(m["total_input_tokens"] for m in all_metrics),
        "total_output_tokens": sum(m["total_output_tokens"] for m in all_metrics),
        "total_tokens": sum(m["total_tokens"] for m in all_metrics),
        "total_time_seconds": round(sum(m["total_time_seconds"] for m in all_metrics), 2),
        "average_time_per_set_seconds": round(sum(m["total_time_seconds"] for m in all_metrics) / len(all_metrics), 2),
        "average_latency_seconds": round(sum(m["average_latency_seconds"] * m["total_prompts"] for m in all_metrics) / sum(m["total_prompts"] for m in all_metrics), 2) if all_metrics else 0,
        "average_ttft_seconds": round(sum(m["average_ttft_seconds"] * m["total_prompts"] for m in all_metrics) / sum(m["total_prompts"] for m in all_metrics), 2) if all_metrics else 0,
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary metrics saved to {summary_path}")

    # Auto-compute bias index
    print(f"\n{'─'*20} Computing Bias Index {'─'*20}")
    bias_result = compute_bias_index(run_dir, model_suffix)
    print(f"\n결과 폴더: {run_dir}")
