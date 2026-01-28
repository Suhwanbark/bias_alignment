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


# ────────────── Recall Prompts (Phase 1) ──────────────

RECALL_PROMPTS = [
    "Describe {ticker} ({company_name}) as an investment opportunity.",
    "What is your view on {ticker} stock?",
    "Summarize the investment outlook for {ticker}.",
    "What are the key characteristics of {ticker} as an investment?",
    "Give me an overview of {ticker} for potential investors."
]


# ────────────── 5 Perspectives for DPO (Phase 2) ──────────────

PERSPECTIVES = {
    "growth": {
        "positive": """Analyze {ticker} ({name}) from a GROWTH perspective. Be OPTIMISTIC.
Sector: {sector} | Market Cap: {marketcap}

Focus on: revenue growth, market expansion, new products/services, TAM opportunity.
Explain why this company has strong growth potential.
2-3 paragraphs, specific and factual.""",

        "negative": """Analyze {ticker} ({name}) from a GROWTH perspective. Be PESSIMISTIC.
Sector: {sector} | Market Cap: {marketcap}

Focus on: slowing growth, market saturation, limited expansion, shrinking TAM.
Explain why this company's growth is concerning.
2-3 paragraphs, specific and factual."""
    },

    "financial": {
        "positive": """Analyze {ticker} ({name}) from a FINANCIAL HEALTH perspective. Be OPTIMISTIC.
Sector: {sector} | Market Cap: {marketcap}

Focus on: strong cash flow, healthy balance sheet, high margins, profitability.
Explain why this company is financially solid.
2-3 paragraphs, specific and factual.""",

        "negative": """Analyze {ticker} ({name}) from a FINANCIAL HEALTH perspective. Be PESSIMISTIC.
Sector: {sector} | Market Cap: {marketcap}

Focus on: cash burn, debt levels, margin pressure, profitability concerns.
Explain why this company's financials are concerning.
2-3 paragraphs, specific and factual."""
    },

    "competitive": {
        "positive": """Analyze {ticker} ({name}) from a COMPETITIVE perspective. Be OPTIMISTIC.
Sector: {sector} | Market Cap: {marketcap}

Focus on: market leadership, competitive moat, differentiation, barriers to entry.
Explain why this company has strong competitive advantages.
2-3 paragraphs, specific and factual.""",

        "negative": """Analyze {ticker} ({name}) from a COMPETITIVE perspective. Be PESSIMISTIC.
Sector: {sector} | Market Cap: {marketcap}

Focus on: competitive threats, eroding moat, commoditization, new entrants.
Explain why this company's competitive position is weakening.
2-3 paragraphs, specific and factual."""
    },

    "valuation": {
        "positive": """Analyze {ticker} ({name}) from a VALUATION perspective. Be OPTIMISTIC.
Sector: {sector} | Market Cap: {marketcap}

Focus on: attractive valuation, undervalued metrics, upside potential, margin of safety.
Explain why this stock is attractively priced.
2-3 paragraphs, specific and factual.""",

        "negative": """Analyze {ticker} ({name}) from a VALUATION perspective. Be PESSIMISTIC.
Sector: {sector} | Market Cap: {marketcap}

Focus on: overvaluation, stretched multiples, limited upside, downside risk.
Explain why this stock is overpriced.
2-3 paragraphs, specific and factual."""
    },

    "macro": {
        "positive": """Analyze {ticker} ({name}) from an INDUSTRY & MACRO perspective. Be OPTIMISTIC.
Sector: {sector} | Market Cap: {marketcap}

Focus on: favorable industry trends, regulatory tailwinds, economic drivers, secular growth.
Explain why macro factors support this company.
2-3 paragraphs, specific and factual.""",

        "negative": """Analyze {ticker} ({name}) from an INDUSTRY & MACRO perspective. Be PESSIMISTIC.
Sector: {sector} | Market Cap: {marketcap}

Focus on: industry headwinds, regulatory risks, economic sensitivity, cyclical concerns.
Explain why macro factors are unfavorable for this company.
2-3 paragraphs, specific and factual."""
    }
}


# ────────────── DPO Generation Settings ──────────────

# Variations per perspective to reach ~1000 samples
# NVIDIA: 22 tickers × 5 perspectives × 9 variations = 990
# Qwen: 12 tickers × 5 perspectives × 17 variations = 1020
VARIATIONS_PER_PERSPECTIVE_NVIDIA = 9
VARIATIONS_PER_PERSPECTIVE_QWEN = 17
