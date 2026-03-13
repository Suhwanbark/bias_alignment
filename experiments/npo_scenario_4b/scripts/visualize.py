"""
Visualize NPO scenario 4B evaluation results.

1. Target vs Non-target buy rate progression (bar chart per step)
2. Per-ticker buy rate heatmap for target tickers across steps
3. Bias index progression

Usage:
    python experiments/npo_scenario_4b/scripts/visualize.py \
        --result-dir experiments/npo_scenario_4b/results/npo_scenario_4b
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd

matplotlib.rcParams.update({"font.size": 11})

TARGET_TICKERS = [
    "A", "ADI", "AMP", "APH", "BAX", "C", "CAG", "CCI", "CDW", "CFG",
    "CHTR", "CINF", "CSX", "EG", "FANG", "FRT", "GD", "GL", "HCA", "HLT",
    "INTU", "IVZ", "J", "JNPR", "KMI", "KMX", "LEN", "LH", "O", "PGR",
    "PSA", "REG", "UPS", "VLO",
]


def load_results(result_dir):
    steps = []
    for d in sorted(os.listdir(result_dir)):
        full = os.path.join(result_dir, d)
        if not os.path.isdir(full):
            continue
        result_json = os.path.join(full, "result.json")
        ticker_csv = os.path.join(full, "ticker_stats.csv")
        if not os.path.exists(result_json):
            continue

        with open(result_json) as f:
            summary = json.load(f)

        ticker_stats = None
        if os.path.exists(ticker_csv):
            ticker_stats = pd.read_csv(ticker_csv)

        label = d.replace("step_", "step ")
        steps.append({
            "label": label,
            "sort_key": -1 if d == "baseline" else int(d.replace("step_", "")),
            "summary": summary,
            "ticker_stats": ticker_stats,
        })

    steps.sort(key=lambda x: x["sort_key"])
    return steps


def plot_buyrate_progression(steps, output_dir):
    """Bar chart: target vs non-target buy rate per step."""
    labels = [s["label"] for s in steps]
    target_rates = []
    nontarget_rates = []
    overall_rates = []

    for s in steps:
        sel = s["summary"].get("selectivity", {})
        target_rates.append(sel.get("target", {}).get("mean_buy_rate", 0))
        nontarget_rates.append(sel.get("non_target", {}).get("mean_buy_rate", 0))
        overall_rates.append(s["summary"].get("overall_buy_rate", 0))

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width, target_rates, width, label=f"Target ({len(TARGET_TICKERS)})", color="#e74c3c", alpha=0.85)
    bars2 = ax.bar(x, nontarget_rates, width, label="Non-target (393)", color="#3498db", alpha=0.85)
    bars3 = ax.bar(x + width, overall_rates, width, label="Overall (427)", color="#95a5a6", alpha=0.7)

    ax.axhline(y=50, color="black", linestyle="--", linewidth=0.8, alpha=0.5, label="50% (unbiased)")
    ax.set_ylabel("Buy Rate (%)")
    ax.set_title("Buy Rate Progression: Target vs Non-Target Tickers")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc="upper right")
    ax.set_ylim(30, 75)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, "buyrate_progression.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_ticker_heatmap(steps, output_dir):
    """Heatmap of per-ticker buy rate across steps for target tickers."""
    labels = [s["label"] for s in steps]
    data = {}

    for s in steps:
        ts = s["ticker_stats"]
        if ts is None:
            continue
        for _, row in ts[ts["ticker"].isin(TARGET_TICKERS)].iterrows():
            ticker = row["ticker"]
            if ticker not in data:
                data[ticker] = {}
            data[ticker][s["label"]] = row["buy_rate"]

    tickers = sorted(data.keys())
    matrix = np.array([[data.get(t, {}).get(l, 50) for l in labels] for t in tickers])

    fig, ax = plt.subplots(figsize=(8, max(8, len(tickers) * 0.35)))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(tickers)))
    ax.set_yticklabels(tickers, fontsize=8)

    for i in range(len(tickers)):
        for j in range(len(labels)):
            val = matrix[i, j]
            color = "white" if val < 25 or val > 75 else "black"
            ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=7, color=color)

    ax.set_title("Target Ticker Buy Rate (%) per Step")
    plt.colorbar(im, ax=ax, label="Buy Rate (%)", shrink=0.6)
    plt.tight_layout()
    path = os.path.join(output_dir, "ticker_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_bias_index(steps, output_dir):
    """Line chart of bias index progression."""
    labels = [s["label"] for s in steps]
    bi = [s["summary"]["bias_index"] for s in steps]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(labels, bi, "o-", color="#2c3e50", linewidth=2, markersize=8)
    for i, (l, v) in enumerate(zip(labels, bi)):
        ax.annotate(str(v), (i, v), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=10)

    ax.set_ylabel("Bias Index")
    ax.set_title("Bias Index Progression")
    ax.set_ylim(0, max(bi) * 1.3)
    ax.axhline(y=bi[0], color="red", linestyle="--", linewidth=0.8, alpha=0.5, label=f"Baseline ({bi[0]})")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, "bias_index.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir",
                        default="experiments/npo_scenario_4b/results/npo_scenario_4b")
    args = parser.parse_args()

    steps = load_results(args.result_dir)
    if not steps:
        print("No results found!")
        return

    print(f"Found {len(steps)} steps: {[s['label'] for s in steps]}")

    plot_buyrate_progression(steps, args.result_dir)
    plot_ticker_heatmap(steps, args.result_dir)
    plot_bias_index(steps, args.result_dir)
    print("Done!")


if __name__ == "__main__":
    main()
