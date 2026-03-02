"""LLM abstraction — decouple from any specific provider."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseLLM(ABC):
    """Interface every LLM wrapper must implement."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a completion for the given prompt."""


class OpenAILLM(BaseLLM):
    """Thin wrapper around the OpenAI chat completions API."""

    def __init__(self, api_key: str, model: str = "gpt-5-mini") -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model = model

    def generate(self, prompt: str) -> str:
        logger.info("Generating with %s (%d chars)", self._model, len(prompt))
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content
        return content or ""
