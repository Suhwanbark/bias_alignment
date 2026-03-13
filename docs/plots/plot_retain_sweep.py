"""
Plot biased vs non-biased ticker buy rates across steps for NPO experiments.
4 subplots: Forget-only, A (0.5/0.5), B (0.75/0.25), C (0.25/0.75)
"""
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ─── Config ───────────────────────────────────────────────────
PROFILE = "npo/data/qwen3-30b-a3b-instruct-2507/profile_report.json"
EXPERIMENTS = [
    ("Forget-only (α_f=1.0)", "npo/results/npo_forget/result"),
    ("A: α_f=0.5, α_r=0.5", "npo/results/npo_retain_a05r05/result"),
    ("B: α_f=0.75, α_r=0.25", "npo/results/npo_retain_a75r25/result"),
    ("C: α_f=0.25, α_r=0.75", "npo/results/npo_retain_a25r75/result"),
]
OUTPUT = "npo/plots/npo_retain_biased_vs_nonbiased.png"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─── Load forget (biased) tickers ─────────────────────────────
with open(os.path.join(ROOT, PROFILE)) as f:
    profile = json.load(f)
BIASED_TICKERS = set(
    t for t, v in profile["ticker_scores"].items() if v["group"] == "high_bias"
)
print(f"Biased tickers ({len(BIASED_TICKERS)}): {sorted(BIASED_TICKERS)}")

# ─── Compute buy rates per step ───────────────────────────────
def compute_buy_rates(result_dir, baseline_rows=None):
    """Return DataFrame with step, group, buy_rate, n_tickers. Prepends baseline as step=0."""
    result_dir = os.path.join(ROOT, result_dir)
    rows = list(baseline_rows) if baseline_rows else []
    for entry in sorted(os.listdir(result_dir)):
        if not entry.startswith("step_"):
            continue
        step = int(entry.split("_")[1])
        step_dir = os.path.join(result_dir, entry)

        # Find combined CSV
        csvs = [f for f in os.listdir(step_dir) if f.endswith("_combined.csv")]
        if not csvs:
            continue
        df = pd.read_csv(os.path.join(step_dir, csvs[0]))

        # Parse buy/sell answers
        df["is_buy"] = df["llm_answer"].str.strip().str.lower() == "buy"
        df["is_valid"] = df["llm_answer"].str.strip().str.lower().isin(["buy", "sell"])

        # Split biased vs non-biased
        mask_biased = df["ticker"].isin(BIASED_TICKERS)

        for label, mask in [("biased", mask_biased), ("nonbiased", ~mask_biased)]:
            subset = df[mask & df["is_valid"]]
            n_tickers = subset["ticker"].nunique()
            if len(subset) == 0:
                continue
            buy_rate = subset["is_buy"].mean() * 100
            rows.append({
                "step": step,
                "group": label,
                "buy_rate": buy_rate,
                "n_tickers": n_tickers,
            })

    return pd.DataFrame(rows)


# ─── Compute baseline buy rates (step 0) ─────────────────────
BASELINE_CSV = "npo/bias_profiling/qwen3-30b-a3b-instruct-2507_att_combined.csv"
baseline_path = os.path.join(ROOT, BASELINE_CSV)
baseline_rows = []
baseline_biased = None
baseline_nonbiased = None
if os.path.exists(baseline_path):
    df_base = pd.read_csv(baseline_path)
    df_base["is_buy"] = df_base["llm_answer"].str.strip().str.lower() == "buy"
    df_base["is_valid"] = df_base["llm_answer"].str.strip().str.lower().isin(["buy", "sell"])
    df_valid = df_base[df_base["is_valid"]]
    mask_b = df_valid["ticker"].isin(BIASED_TICKERS)
    baseline_biased = df_valid[mask_b]["is_buy"].mean() * 100
    baseline_nonbiased = df_valid[~mask_b]["is_buy"].mean() * 100
    baseline_rows = [
        {"step": 0, "group": "biased", "buy_rate": baseline_biased,
         "n_tickers": df_valid[mask_b]["ticker"].nunique()},
        {"step": 0, "group": "nonbiased", "buy_rate": baseline_nonbiased,
         "n_tickers": df_valid[~mask_b]["ticker"].nunique()},
    ]
    print(f"Baseline (step 0): biased={baseline_biased:.1f}%, nonbiased={baseline_nonbiased:.1f}%")
else:
    print("Baseline CSV not found, skipping step 0")

# ─── Plot ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(24, 6), sharey=True)
fig.suptitle("NPO: Biased vs Non-Biased Tickers Buy Rate", fontsize=14, fontweight="bold")

MAX_STEP = 300  # Limit all subplots to step 300

for ax, (title, result_dir) in zip(axes, EXPERIMENTS):
    data = compute_buy_rates(result_dir, baseline_rows)
    if data.empty:
        ax.set_title(f"{title}\n(no data)")
        continue

    data = data[data["step"] <= MAX_STEP]

    biased = data[data["group"] == "biased"].sort_values("step")
    nonbiased = data[data["group"] == "nonbiased"].sort_values("step")

    n_b = biased["n_tickers"].iloc[0] if len(biased) > 0 else 0
    n_nb = nonbiased["n_tickers"].iloc[0] if len(nonbiased) > 0 else 0

    ax.plot(biased["step"], biased["buy_rate"], "o-", color="tab:red",
            label=f"Biased {n_b} tickers", markersize=6, linewidth=2)
    ax.plot(nonbiased["step"], nonbiased["buy_rate"], "s-", color="tab:blue",
            label=f"Non-biased {n_nb} tickers", markersize=6, linewidth=2)

    # Annotate values
    for _, row in biased.iterrows():
        ax.annotate(f"{row['buy_rate']:.0f}", (row["step"], row["buy_rate"]),
                     textcoords="offset points", xytext=(0, 10), ha="center",
                     fontsize=8, color="tab:red")
    for _, row in nonbiased.iterrows():
        ax.annotate(f"{row['buy_rate']:.0f}", (row["step"], row["buy_rate"]),
                     textcoords="offset points", xytext=(0, -14), ha="center",
                     fontsize=8, color="tab:blue")

    # Reference line
    ax.axhline(50, color="green", linestyle=":", alpha=0.5, label="Ideal (50%)")

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Training Step")
    ax.set_xticks(sorted(data["step"].unique()))
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel("Avg Buy Rate (%)")
axes[0].set_ylim(0, 105)

plt.tight_layout()
os.makedirs(os.path.dirname(os.path.join(ROOT, OUTPUT)), exist_ok=True)
plt.savefig(os.path.join(ROOT, OUTPUT), dpi=150, bbox_inches="tight")
print(f"Saved: {OUTPUT}")
plt.close()
