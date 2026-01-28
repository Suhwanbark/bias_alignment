"""
Configuration for DPO Debiasing Experiment
"""

# ────────────── Target Tickers ──────────────

# NVIDIA Nemotron - 22 tickers (buy_rate=0%, SELL bias)
NVIDIA_TICKERS = [
    "ECL", "IFF", "NWSA", "META", "CCL", "PEP", "DG", "PFG", "COF", "MTB", "GL", "HBAN",
    "BAX", "HCA", "GWW", "PH", "CHRW", "ITW", "DLR", "WY", "EA", "D"
]

# Qwen3 - 12 tickers (buy_rate>=90%, BUY bias)
QWEN_TICKERS = [
    "LMT", "SO", "PYPL", "PPL", "WMT", "VRSN", "CVX", "EW", "JNJ", "NVDA", "PWR", "TSN"
]

# ────────────── LLM Configuration ──────────────
DEFAULT_MODEL_ID = "gpt-oss-20b"
DEFAULT_VLLM_URL = "http://localhost:8000/v1"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TOP_P = 0.9

# ────────────── Data Generation Settings ──────────────
EVENTS_PER_TICKER = 10  # Number of positive/negative events per ticker
DPO_SAMPLES_PER_TICKER_NVIDIA = 45  # ~1000 total for 22 tickers
DPO_SAMPLES_PER_TICKER_QWEN = 83  # ~1000 total for 12 tickers

# ────────────── Prompts ──────────────

EVENT_GENERATION_SYSTEM_PROMPT = """You are a financial analyst with expertise in stock market analysis.
Your task is to generate realistic positive and negative events/news for a given stock ticker.
These events should be plausible and specific to the company's industry and business model."""

EVENT_GENERATION_USER_PROMPT = """Generate {num_events} positive and {num_events} negative events/news items for the stock ticker [{ticker}] ({company_name}).

Requirements:
- Each event should be 1-2 sentences
- Events should be realistic and plausible
- Include a mix of: earnings reports, product launches, regulatory news, market conditions, competitive developments
- Be specific to the company's sector ({sector})

Return your response as a JSON object with this format:
{{
    "positive_events": [
        "Event 1 description...",
        "Event 2 description...",
        ...
    ],
    "negative_events": [
        "Event 1 description...",
        "Event 2 description...",
        ...
    ]
}}

Return ONLY the JSON object, no other text."""

DPO_GENERATION_SYSTEM_PROMPT = """You are a financial analyst providing balanced investment decisions.
Given evidence about a stock, you will provide TWO reasonable responses:
1. A response that recommends BUY with a brief justification
2. A response that recommends SELL with a brief justification

Both responses should be reasonable given the mixed evidence provided."""

DPO_GENERATION_USER_PROMPT = """Stock: [{ticker}] ({company_name})

Evidence:
{evidence}

Provide two reasonable investment decisions based on this mixed evidence:
1. A BUY recommendation with justification (2-3 sentences)
2. A SELL recommendation with justification (2-3 sentences)

Return your response as a JSON object with this format:
{{
    "buy_response": {{
        "decision": "buy",
        "reason": "Your justification for buying..."
    }},
    "sell_response": {{
        "decision": "sell",
        "reason": "Your justification for selling..."
    }}
}}

Return ONLY the JSON object, no other text."""
