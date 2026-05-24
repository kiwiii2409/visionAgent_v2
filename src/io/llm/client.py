"""
src/io/llm_client.py

Role:
    Interface to the local/ api LLM model, handles short-term memory summarization

"""

import os
import asyncio
from typing import List, Dict, Optional
from openai import AsyncOpenAI


class LLMClient:
    def __init__(self, model_name: str, base_url: str, api_key: str = None) -> None:
        self.model_name = model_name
        self.base_url = base_url
        self.client = AsyncOpenAI(
            api_key=api_key if api_key else "dummy-key",
            base_url=self.base_url
        )

    async def query(self, user_prompt: str,  system_prompt: str = "", response_format: Optional[dict] = None, max_tokens: Optional[int] = None,  temperature: float = 0.3) -> str:
        """Query the LLM with a system and user prompt, returns the generated text."""
        payload = self._format_payload(user_prompt, system_prompt)

        kwargs = {
            "model": self.model_name,
            "messages": payload,
            "temperature": temperature,
        }

        if response_format:
            kwargs["response_format"] = response_format
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        try:
            completion = await asyncio.wait_for(
                self.client.chat.completions.create(**kwargs),
                timeout=5.0  # TODO is this too low?
            )
            return completion.choices[0].message.content
        except asyncio.TimeoutError:
            print("[LLMClient] API call timed out after 5 seconds.")
            return ""  # TODO should i add a return value, might be useful to force a retry
        except Exception as e:
            print(f"[LLMClient] API Error during query: {e}")
            return ""

    def _format_payload(self, user_prompt: str, system_prompt: str):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": user_prompt})

        return messages
