import os
import time
from typing import Optional
from openai import OpenAI

# ────────────── Configuration ──────────────
MAX_RETRIES = 3
RETRY_DELAY = 2
DEFAULT_MAX_TOKENS = 1024
DEFAULT_VLLM_URL = "http://localhost:8000/v1"


class VLLMClient:
    """
    LLM client using vLLM's OpenAI-compatible API.

    Usage:
        # Start vLLM server first:
        # vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 --tensor-parallel-size 4 --port 8000

        client = VLLMClient("Qwen/Qwen3-30B-A3B-Instruct-2507")
        response = client.get_response("Hello!")
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-30B-A3B-Instruct-2507",
        temperature: float = 0.6,
        max_tokens: Optional[int] = None,
        top_p: float = 0.8,
        base_url: str = DEFAULT_VLLM_URL,
        seed: Optional[int] = None,
    ):
        """
        Initialize the vLLM client.

        Args:
            model_id: HuggingFace model ID (must match vLLM server model)
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum new tokens in response
            top_p: Top-p sampling parameter
            base_url: vLLM server URL (default: http://localhost:8000/v1)
            seed: Random seed for reproducibility
        """
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens or DEFAULT_MAX_TOKENS
        self.top_p = top_p
        self.base_url = base_url
        self.seed = seed
        self.short_model_id = model_id.split('/')[-1] if '/' in model_id else model_id

        # Initialize OpenAI client pointing to vLLM server
        self.client = OpenAI(
            base_url=base_url,
            api_key="EMPTY",  # vLLM doesn't require API key by default
        )

        # Metrics (compatible with original interface)
        self.last_call_cost = 0.0
        self.last_input_tokens = 0
        self.last_output_tokens = 0
        self.last_ttft = 0.0
        self.last_generation_id = ""
        self.last_prompt_cost = 0.0
        self.last_completion_cost = 0.0
        self.reasoning_effort = None

    def get_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Get a response from the vLLM server.

        Args:
            prompt: User prompt/message
            system_prompt: Optional system prompt

        Returns:
            Model response text
        """
        # Reset metrics
        self.last_call_cost = 0.0
        self.last_input_tokens = 0
        self.last_output_tokens = 0
        self.last_ttft = 0.0
        self.last_generation_id = ""

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                start_time = time.time()

                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    seed=self.seed,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )

                self.last_ttft = time.time() - start_time

                # Extract response
                content = response.choices[0].message.content

                if not content or not content.strip():
                    last_error = f"Empty response (Attempt {attempt+1})"
                    time.sleep(RETRY_DELAY)
                    continue

                # Extract usage info
                if response.usage:
                    self.last_input_tokens = response.usage.prompt_tokens
                    self.last_output_tokens = response.usage.completion_tokens

                self.last_generation_id = response.id

                return content.strip()

            except Exception as e:
                last_error = f"Error: {e} (Attempt {attempt+1})"
                time.sleep(RETRY_DELAY * (attempt + 1))

        return f"Failed after {MAX_RETRIES} attempts. Last error: {last_error}"

    def get_usage_summary(self) -> dict:
        """Get a summary of the last call's usage."""
        return {
            "model": self.model_id,
            "generation_id": self.last_generation_id,
            "input_tokens": self.last_input_tokens,
            "output_tokens": self.last_output_tokens,
            "total_tokens": self.last_input_tokens + self.last_output_tokens,
            "prompt_cost": 0.0,  # Local inference has no cost
            "completion_cost": 0.0,
            "total_cost": 0.0,
            "ttft_seconds": self.last_ttft,
        }

    def __repr__(self) -> str:
        return f"VLLMClient(model_id='{self.model_id}', base_url='{self.base_url}')"


# Alias for compatibility
LLMClient = VLLMClient
LocalLLMClient = VLLMClient
