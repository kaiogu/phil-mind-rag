"""LLM abstraction — decouple from any specific provider."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import OpenAI

logger = logging.getLogger(__name__)


class BaseLLM(ABC):
    """Interface every LLM wrapper must implement."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a completion for the given prompt."""


class OpenAILLM(BaseLLM):
    """Thin wrapper around an OpenAI-compatible chat completions API."""

    def __init__(self, client: OpenAI, model: str = "gpt-5-mini") -> None:
        self._client = client
        self._model = model

    def generate(self, prompt: str) -> str:
        logger.info("Generating with %s (%d chars)", self._model, len(prompt))
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content
        return content or ""
