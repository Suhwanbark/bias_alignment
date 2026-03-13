#!/usr/bin/env python3
"""
Bias Score Visualization

Usage:
    # Sector bias bar chart from a single result
    python eval/visualize.py sft/results/sft_v7/result/step_12 --type sector

    # Size (market cap quartile) bias bar chart
    python eval/visualize.py sft/results/sft_v7/result/step_12 --type size

    # Per-ticker bias distribution (histogram)
    python eval/visualize.py sft/results/sft_v7/result/step_12 --type ticker-dist

    # Target vs untarget ticker comparison (requires --target-tickers)
    python eval/visualize.py sft/results/sft_v7/result/step_12 --type target-untarget \
        --target-tickers LMT SO PYPL PPL WMT VRSN CVX EW JNJ NVDA PWR TSN

    # Compare multiple results (baseline vs debiased)
    python eval/visualize.py sft/results/sft_v1/result/step_59 sft/results/sft_v7/result/step_12 \
        --type compare --labels "SFT v1 (step 59)" "SFT v7 (step 12)"

    # All plots for a single result
    python eval/visualize.py sft/results/sft_v7/result/step_12 --type all
"""

import argparse
import json
import os
import sys
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


# ────────────── Data Loading ──────────────

def find_result_json(result_dir: str) -> str:
    """Find the *_att_result.json file in a result directory."""
    candidates = sorted(glob.glob(os.path.join(result_dir, "*_att_result.json")))
    if not candidates:
        raise FileNotFoundError(f"No *_att_result.json found in {result_dir}")
    return candidates[0]


def find_combined_csv(result_dir: str) -> str:
    """Find the *_att_combined.csv file in a result directory."""
    candidates = sorted(glob.glob(os.path.join(result_dir, "*_att_combined.csv")))
    if not candidates:
        raise FileNotFoundError(f"No *_att_combined.csv found in {result_dir}")
    return candidates[0]


def load_result_json(result_dir: str) -> dict:
    """Load the result JSON summary."""
    path = find_result_json(result_dir)
    with open(path, "r") as f:
        return json.load(f)


def load_ticker_bias(result_dir: str) -> pd.DataFrame:
    """Load combined CSV and compute per-ticker bias scores.

    Returns DataFrame with columns:
        ticker, name, sector, marketcap, buy_count, sell_count, total_count,
        bias_score, buy_rate
    """
    csv_path = find_combined_csv(result_dir)
    df = pd.read_csv(csv_path)

    df["is_buy"] = df["llm_answer"].str.lower() == "buy"
    df["is_sell"] = df["llm_answer"].str.lower() == "sell"

    grouped = (
        df.groupby(["ticker", "name", "sector", "marketcap"])
        .agg(buy_count=("is_buy", "sum"), sell_count=("is_sell", "sum"))
        .reset_index()
    )
    grouped["total_count"] = grouped["buy_count"] + grouped["sell_count"]
    grouped = grouped[grouped["total_count"] > 0].copy()
    grouped["bias_score"] = (
        (grouped["buy_count"] - grouped["sell_count"]) / grouped["total_count"]
    ) * 100.0
    grouped["buy_rate"] = grouped["buy_count"] / grouped["total_count"] * 100.0

    return grouped


def auto_label(result_dir: str) -> str:
    """Generate a human-readable label from the result directory path."""
    parts = os.path.normpath(result_dir).split(os.sep)
    # Try to find meaningful parts like "sft_v7", "step_12", etc.
    meaningful = []
    for p in parts:
        if p.startswith(("sft_", "dpo_", "npo_", "step_", "sector_")):
            meaningful.append(p)
        elif "baseline" in p.lower():
            meaningful.append(p)
    if meaningful:
        return " / ".join(meaningful)
    # Fallback: last 2 path components
    return "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]


# ────────────── Plot Functions ──────────────

def plot_sector(result_dirs: list, labels: list, output_path: str):
    """Bar chart of per-sector bias_mean for one or more results."""
    fig, ax = plt.subplots(figsize=(14, 6))

    n_results = len(result_dirs)
    all_sectors = set()
    data_list = []

    for rdir in result_dirs:
        rj = load_result_json(rdir)
        sectors = rj.get("sector_stats", {})
        all_sectors.update(sectors.keys())
        data_list.append(sectors)

    # Sort sectors by first result's bias_mean (descending)
    sector_order = sorted(
        all_sectors,
        key=lambda s: data_list[0].get(s, {}).get("bias_mean", 0),
        reverse=True,
    )

    x = np.arange(len(sector_order))
    width = 0.8 / n_results

    for i, (sectors, label) in enumerate(zip(data_list, labels)):
        means = [sectors.get(s, {}).get("bias_mean", 0) for s in sector_order]
        stds = [sectors.get(s, {}).get("bias_std", 0) for s in sector_order]
        offset = (i - n_results / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, means, width, label=label, yerr=stds, capsize=3, alpha=0.85
        )

    ax.set_xticks(x)
    ax.set_xticklabels(sector_order, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Bias Score (mean)")
    ax.set_title("Sector Bias Scores")
    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="-")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # Add bias_index annotation
    for i, rdir in enumerate(result_dirs):
        rj = load_result_json(rdir)
        bi = rj.get("bias_index", "?")
        ax.annotate(
            f"BI={bi}",
            xy=(0.01 + i * 0.12, 0.97),
            xycoords="axes fraction",
            fontsize=9,
            fontweight="bold",
            color=f"C{i}",
            va="top",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_size(result_dirs: list, labels: list, output_path: str):
    """Bar chart of per-size-group bias_mean for one or more results."""
    fig, ax = plt.subplots(figsize=(8, 5))

    n_results = len(result_dirs)
    size_order = ["Q1", "Q2", "Q3", "Q4"]  # Largest to smallest
    data_list = []

    for rdir in result_dirs:
        rj = load_result_json(rdir)
        data_list.append(rj.get("size_stats", {}))

    x = np.arange(len(size_order))
    width = 0.8 / n_results

    for i, (sizes, label) in enumerate(zip(data_list, labels)):
        means = [sizes.get(s, {}).get("bias_mean", 0) for s in size_order]
        stds = [sizes.get(s, {}).get("bias_std", 0) for s in size_order]
        offset = (i - n_results / 2 + 0.5) * width
        ax.bar(x + offset, means, width, label=label, yerr=stds, capsize=3, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Q1 (Largest)", "Q2", "Q3", "Q4 (Smallest)"], fontsize=10
    )
    ax.set_ylabel("Bias Score (mean)")
    ax.set_title("Market Cap Quartile Bias Scores")
    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="-")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_ticker_dist(result_dirs: list, labels: list, output_path: str):
    """Histogram of per-ticker bias_score distribution for one or more results."""
    fig, ax = plt.subplots(figsize=(10, 5))

    bins = np.linspace(-100, 100, 41)

    for i, (rdir, label) in enumerate(zip(result_dirs, labels)):
        ticker_df = load_ticker_bias(rdir)
        ax.hist(
            ticker_df["bias_score"],
            bins=bins,
            alpha=0.5,
            label=f"{label} (mean={ticker_df['bias_score'].mean():.1f})",
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xlabel("Bias Score (per ticker)")
    ax.set_ylabel("Number of Tickers")
    ax.set_title("Per-Ticker Bias Score Distribution")
    ax.axvline(x=0, color="black", linewidth=0.8, linestyle="--")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_target_untarget(
    result_dir: str, label: str, target_tickers: list, output_path: str
):
    """Box/violin plot comparing target vs untarget ticker bias scores."""
    ticker_df = load_ticker_bias(result_dir)
    ticker_df["group"] = ticker_df["ticker"].apply(
        lambda t: "Target" if t in target_tickers else "Untarget"
    )

    target = ticker_df[ticker_df["group"] == "Target"]["bias_score"]
    untarget = ticker_df[ticker_df["group"] == "Untarget"]["bias_score"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: violin plot
    ax = axes[0]
    parts = ax.violinplot(
        [target.values, untarget.values],
        positions=[1, 2],
        showmeans=True,
        showmedians=True,
    )
    for pc in parts["bodies"]:
        pc.set_alpha(0.6)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(
        [f"Target\n(n={len(target)})", f"Untarget\n(n={len(untarget)})"]
    )
    ax.set_ylabel("Bias Score")
    ax.set_title(f"Target vs Untarget Bias Distribution\n{label}")
    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--")
    ax.grid(axis="y", alpha=0.3)

    # Annotate means
    ax.annotate(
        f"mean={target.mean():.1f}",
        xy=(1, target.mean()),
        xytext=(1.3, target.mean() + 5),
        fontsize=9,
        fontweight="bold",
        color="C0",
    )
    ax.annotate(
        f"mean={untarget.mean():.1f}",
        xy=(2, untarget.mean()),
        xytext=(2.3, untarget.mean() + 5),
        fontsize=9,
        fontweight="bold",
        color="C1",
    )

    # Right: sorted bar chart for target tickers
    ax2 = axes[1]
    target_df = ticker_df[ticker_df["group"] == "Target"].sort_values(
        "bias_score", ascending=True
    )
    colors = ["#d32f2f" if v < 0 else "#1976d2" for v in target_df["bias_score"]]
    ax2.barh(target_df["ticker"], target_df["bias_score"], color=colors, alpha=0.8)
    ax2.set_xlabel("Bias Score")
    ax2.set_title(f"Target Ticker Bias Scores\n(>0 = BUY bias, <0 = SELL bias)")
    ax2.axvline(x=0, color="black", linewidth=0.8, linestyle="-")
    ax2.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_compare(result_dirs: list, labels: list, output_path: str):
    """Multi-panel comparison: sector + size + ticker distribution + bias index summary."""
    n = len(result_dirs)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # ── Panel 1: Sector comparison ──
    ax = axes[0, 0]
    all_sectors = set()
    data_list = []
    for rdir in result_dirs:
        rj = load_result_json(rdir)
        sectors = rj.get("sector_stats", {})
        all_sectors.update(sectors.keys())
        data_list.append(sectors)

    sector_order = sorted(
        all_sectors,
        key=lambda s: data_list[0].get(s, {}).get("bias_mean", 0),
        reverse=True,
    )
    x = np.arange(len(sector_order))
    width = 0.8 / n
    for i, (sectors, label) in enumerate(zip(data_list, labels)):
        means = [sectors.get(s, {}).get("bias_mean", 0) for s in sector_order]
        offset = (i - n / 2 + 0.5) * width
        ax.bar(x + offset, means, width, label=label, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(sector_order, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Bias Score (mean)")
    ax.set_title("Sector Bias")
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # ── Panel 2: Size comparison ──
    ax = axes[0, 1]
    size_order = ["Q1", "Q2", "Q3", "Q4"]
    size_data = []
    for rdir in result_dirs:
        rj = load_result_json(rdir)
        size_data.append(rj.get("size_stats", {}))
    x = np.arange(len(size_order))
    width = 0.8 / n
    for i, (sizes, label) in enumerate(zip(size_data, labels)):
        means = [sizes.get(s, {}).get("bias_mean", 0) for s in size_order]
        offset = (i - n / 2 + 0.5) * width
        ax.bar(x + offset, means, width, label=label, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(["Q1 (Largest)", "Q2", "Q3", "Q4 (Smallest)"], fontsize=9)
    ax.set_ylabel("Bias Score (mean)")
    ax.set_title("Market Cap Quartile Bias")
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # ── Panel 3: Ticker distribution ──
    ax = axes[1, 0]
    bins = np.linspace(-100, 100, 41)
    for i, (rdir, label) in enumerate(zip(result_dirs, labels)):
        try:
            ticker_df = load_ticker_bias(rdir)
            ax.hist(
                ticker_df["bias_score"],
                bins=bins,
                alpha=0.45,
                label=f"{label} (mean={ticker_df['bias_score'].mean():.1f})",
                edgecolor="white",
                linewidth=0.5,
            )
        except FileNotFoundError:
            pass
    ax.set_xlabel("Bias Score")
    ax.set_ylabel("Number of Tickers")
    ax.set_title("Per-Ticker Bias Distribution")
    ax.axvline(x=0, color="black", linewidth=0.8, linestyle="--")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # ── Panel 4: Bias Index summary table ──
    ax = axes[1, 1]
    ax.axis("off")
    table_data = []
    for rdir, label in zip(result_dirs, labels):
        rj = load_result_json(rdir)
        bi = rj.get("bias_index", "?")
        high_sector = rj.get("bias_result", {}).get("high_bias_sector", "?")
        low_sector = rj.get("bias_result", {}).get("low_bias_sector", "?")
        try:
            tdf = load_ticker_bias(rdir)
            mean_buy_rate = tdf["buy_rate"].mean()
            buy_rate_str = f"{mean_buy_rate:.1f}%"
        except FileNotFoundError:
            buy_rate_str = "N/A"
        table_data.append([label, str(bi), high_sector, low_sector, buy_rate_str])

    col_labels = ["Experiment", "Bias Index", "High Bias Sector", "Low Bias Sector", "Avg Buy Rate"]
    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.8)

    # Style header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Highlight best (lowest) bias index
    if table_data:
        bis = []
        for row in table_data:
            try:
                bis.append(int(row[1]))
            except (ValueError, TypeError):
                bis.append(float("inf"))
        best_idx = bis.index(min(bis))
        for j in range(len(col_labels)):
            table[best_idx + 1, j].set_facecolor("#E2EFDA")

    ax.set_title("Summary", fontweight="bold", fontsize=12, pad=20)

    plt.suptitle("Bias Evaluation Comparison", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


# ────────────── Main ──────────────

def main():
    parser = argparse.ArgumentParser(
        description="Bias Score Visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "result_dirs",
        nargs="+",
        help="Path(s) to result directory(ies) containing *_att_result.json and *_att_combined.csv",
    )
    parser.add_argument(
        "--type",
        choices=["sector", "size", "ticker-dist", "target-untarget", "compare", "all"],
        default="sector",
        help="Type of visualization (default: sector)",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Labels for each result directory (default: auto-generated from path)",
    )
    parser.add_argument(
        "--target-tickers",
        nargs="+",
        default=None,
        help="List of target tickers for target-untarget plot",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output image path (default: auto-generated in first result dir)",
    )
    parser.add_argument(
        "--format",
        choices=["png", "pdf", "svg"],
        default="png",
        help="Output format (default: png)",
    )
    args = parser.parse_args()

    # Validate result directories
    for rdir in args.result_dirs:
        if not os.path.isdir(rdir):
            print(f"Error: {rdir} is not a directory")
            sys.exit(1)

    # Generate labels
    if args.labels:
        if len(args.labels) != len(args.result_dirs):
            print(
                f"Error: number of labels ({len(args.labels)}) must match "
                f"number of result dirs ({len(args.result_dirs)})"
            )
            sys.exit(1)
        labels = args.labels
    else:
        labels = [auto_label(rdir) for rdir in args.result_dirs]

    # Default output directory
    out_dir = args.result_dirs[0]
    fmt = args.format

    plot_type = args.type

    if plot_type == "all":
        # Generate all applicable plots
        out_sector = args.output or os.path.join(out_dir, f"plot_sector.{fmt}")
        plot_sector(args.result_dirs, labels, out_sector)

        out_size = os.path.join(out_dir, f"plot_size.{fmt}")
        plot_size(args.result_dirs, labels, out_size)

        try:
            out_ticker = os.path.join(out_dir, f"plot_ticker_dist.{fmt}")
            plot_ticker_dist(args.result_dirs, labels, out_ticker)
        except FileNotFoundError as e:
            print(f"Skipping ticker-dist (no CSV): {e}")

        if args.target_tickers:
            out_target = os.path.join(out_dir, f"plot_target_untarget.{fmt}")
            plot_target_untarget(
                args.result_dirs[0], labels[0], args.target_tickers, out_target
            )

        if len(args.result_dirs) > 1:
            out_compare = os.path.join(out_dir, f"plot_compare.{fmt}")
            plot_compare(args.result_dirs, labels, out_compare)

        return

    # Single plot type
    if plot_type == "sector":
        out_path = args.output or os.path.join(out_dir, f"plot_sector.{fmt}")
        plot_sector(args.result_dirs, labels, out_path)

    elif plot_type == "size":
        out_path = args.output or os.path.join(out_dir, f"plot_size.{fmt}")
        plot_size(args.result_dirs, labels, out_path)

    elif plot_type == "ticker-dist":
        out_path = args.output or os.path.join(out_dir, f"plot_ticker_dist.{fmt}")
        plot_ticker_dist(args.result_dirs, labels, out_path)

    elif plot_type == "target-untarget":
        if not args.target_tickers:
            # Try to use known target tickers from config
            try:
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                from config import QWEN_TICKERS
                args.target_tickers = QWEN_TICKERS
                print(f"Using default QWEN_TICKERS ({len(args.target_tickers)} tickers)")
            except ImportError:
                print(
                    "Error: --target-tickers required for target-untarget plot "
                    "(or config.py with QWEN_TICKERS must be available)"
                )
                sys.exit(1)
        out_path = args.output or os.path.join(out_dir, f"plot_target_untarget.{fmt}")
        plot_target_untarget(
            args.result_dirs[0], labels[0], args.target_tickers, out_path
        )

    elif plot_type == "compare":
        if len(args.result_dirs) < 2:
            print("Error: --type compare requires at least 2 result directories")
            sys.exit(1)
        out_path = args.output or os.path.join(out_dir, f"plot_compare.{fmt}")
        plot_compare(args.result_dirs, labels, out_path)


if __name__ == "__main__":
    main()
