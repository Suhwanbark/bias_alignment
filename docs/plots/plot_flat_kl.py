"""Plot FLAT-KL: biased vs non-biased ticker buy rates across steps."""
import json
import os
import pandas as pd
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROFILE = "npo/data/qwen3-30b-a3b-instruct-2507/profile_report.json"
RESULT_DIR = "npo/results/npo_flat_kl/result"
BASELINE_CSV = "npo/bias_profiling/qwen3-30b-a3b-instruct-2507_att_combined.csv"
OUTPUT = "npo/plots/flat_kl_biased_vs_nonbiased.png"

# Load biased tickers
with open(os.path.join(ROOT, PROFILE)) as f:
    profile = json.load(f)
BIASED_TICKERS = set(
    t for t, v in profile["ticker_scores"].items() if v["group"] == "high_bias"
)
print(f"Biased tickers ({len(BIASED_TICKERS)}): {sorted(BIASED_TICKERS)}")

# Compute baseline (step 0)
baseline_rows = []
baseline_path = os.path.join(ROOT, BASELINE_CSV)
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

# Compute buy rates per step
result_dir = os.path.join(ROOT, RESULT_DIR)
rows = list(baseline_rows)
for entry in sorted(os.listdir(result_dir)):
    if not entry.startswith("step_"):
        continue
    step = int(entry.split("_")[1])
    step_dir = os.path.join(result_dir, entry)
    csvs = [f for f in os.listdir(step_dir) if f.endswith("_combined.csv")]
    if not csvs:
        # Check subdirectories
        for sub in os.listdir(step_dir):
            sub_path = os.path.join(step_dir, sub)
            if os.path.isdir(sub_path):
                csvs = [os.path.join(sub, f) for f in os.listdir(sub_path) if f.endswith("_combined.csv")]
                if csvs:
                    break
    if not csvs:
        continue
    df = pd.read_csv(os.path.join(step_dir, csvs[0]))
    df["is_buy"] = df["llm_answer"].str.strip().str.lower() == "buy"
    df["is_valid"] = df["llm_answer"].str.strip().str.lower().isin(["buy", "sell"])
    mask_biased = df["ticker"].isin(BIASED_TICKERS)
    for label, mask in [("biased", mask_biased), ("nonbiased", ~mask_biased)]:
        subset = df[mask & df["is_valid"]]
        if len(subset) == 0:
            continue
        buy_rate = subset["is_buy"].mean() * 100
        rows.append({
            "step": step, "group": label, "buy_rate": buy_rate,
            "n_tickers": subset["ticker"].nunique(),
        })

data = pd.DataFrame(rows)
MAX_STEP = 350
data = data[data["step"] <= MAX_STEP]

# Plot
fig, ax = plt.subplots(figsize=(12, 7))

biased = data[data["group"] == "biased"].sort_values("step")
nonbiased = data[data["group"] == "nonbiased"].sort_values("step")

n_b = biased["n_tickers"].iloc[0] if len(biased) > 0 else 0
n_nb = nonbiased["n_tickers"].iloc[0] if len(nonbiased) > 0 else 0

ax.plot(biased["step"], biased["buy_rate"], "o-", color="tab:red",
        label=f"Biased {n_b} tickers", markersize=8, linewidth=2.5)
ax.plot(nonbiased["step"], nonbiased["buy_rate"], "s-", color="tab:blue",
        label=f"Non-biased {n_nb} tickers", markersize=8, linewidth=2.5)

# Annotate
for _, row in biased.iterrows():
    ax.annotate(f"{row['buy_rate']:.0f}", (row["step"], row["buy_rate"]),
                textcoords="offset points", xytext=(0, 12), ha="center",
                fontsize=10, fontweight="bold", color="tab:red")
for _, row in nonbiased.iterrows():
    ax.annotate(f"{row['buy_rate']:.0f}", (row["step"], row["buy_rate"]),
                textcoords="offset points", xytext=(0, -16), ha="center",
                fontsize=10, fontweight="bold", color="tab:blue")

ax.axhline(50, color="green", linestyle=":", alpha=0.6, linewidth=1.5, label="Ideal (50%)")
ax.set_xlabel("Training Step", fontsize=13)
ax.set_ylabel("Avg Buy Rate (%)", fontsize=13)
ax.set_xticks(sorted(data["step"].unique()))
ax.tick_params(axis="x", rotation=45)
ax.set_ylim(0, 105)
ax.legend(fontsize=11, loc="upper right")
ax.grid(True, alpha=0.3)
ax.set_title("FLAT-KL: Biased vs Non-Biased Tickers Buy Rate\n(template=\"I don't know.\", lr=2e-5, LoRA r=8)",
             fontsize=14, fontweight="bold")

plt.tight_layout()
os.makedirs(os.path.dirname(os.path.join(ROOT, OUTPUT)), exist_ok=True)
plt.savefig(os.path.join(ROOT, OUTPUT), dpi=150, bbox_inches="tight")
print(f"Saved: {OUTPUT}")
plt.close()
