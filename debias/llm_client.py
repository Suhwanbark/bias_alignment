"""
DPO 데이터 생성용 LLM 클라이언트 (vLLM OpenAI 호환 API 사용)
"""

import time
import json
import re
from typing import Optional, Dict
from openai import OpenAI

# ────────────── 설정 ──────────────
MAX_RETRIES = 3
RETRY_DELAY = 2
DEFAULT_MAX_TOKENS = 2048
DEFAULT_VLLM_URL = "http://localhost:8000/v1"


class VLLMClient:
    """
    vLLM의 OpenAI 호환 API를 사용하는 LLM 클라이언트

    사용법:
        client = VLLMClient("gpt-oss-20b")
        response = client.get_response("Hello!")
    """

    def __init__(
        self,
        model_id: str = "gpt-oss-20b",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 0.9,
        base_url: str = DEFAULT_VLLM_URL,
    ):
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens or DEFAULT_MAX_TOKENS
        self.top_p = top_p
        self.base_url = base_url
        self.short_model_id = model_id.split('/')[-1] if '/' in model_id else model_id

        self.client = OpenAI(
            base_url=base_url,
            api_key="EMPTY",
        )

        # 메트릭
        self.last_input_tokens = 0
        self.last_output_tokens = 0
        self.last_ttft = 0.0

    def get_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        vLLM 서버로부터 응답을 받음

        Args:
            prompt: 사용자 프롬프트
            system_prompt: 시스템 프롬프트 (선택)
            temperature: 이 호출에 대한 temperature 오버라이드

        Returns:
            모델 응답 텍스트
        """
        self.last_input_tokens = 0
        self.last_output_tokens = 0
        self.last_ttft = 0.0

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        temp = temperature if temperature is not None else self.temperature

        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                start_time = time.time()

                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    temperature=temp,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                )

                self.last_ttft = time.time() - start_time

                content = response.choices[0].message.content

                if not content or not content.strip():
                    last_error = f"빈 응답 (시도 {attempt+1})"
                    time.sleep(RETRY_DELAY)
                    continue

                if response.usage:
                    self.last_input_tokens = response.usage.prompt_tokens
                    self.last_output_tokens = response.usage.completion_tokens

                return content.strip()

            except Exception as e:
                last_error = f"오류: {e} (시도 {attempt+1})"
                time.sleep(RETRY_DELAY * (attempt + 1))

        return f"FAILED: {MAX_RETRIES}회 시도 실패. 마지막 오류: {last_error}"

    def get_json_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Optional[Dict]:
        """
        응답을 받아 JSON으로 파싱

        Returns:
            파싱된 JSON dict 또는 실패시 None
        """
        response = self.get_response(prompt, system_prompt, temperature)

        if response.startswith("FAILED"):
            return None

        # 응답에서 JSON 추출
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if not match:
            return None

        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    def __repr__(self) -> str:
        return f"VLLMClient(model_id='{self.model_id}', base_url='{self.base_url}')"
