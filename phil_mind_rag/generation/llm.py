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

    def __init__(
        self,
        client: OpenAI,
        model: str | tuple[str, ...] = "gpt-5-mini",
    ) -> None:
        self._client = client
        self._models = (model,) if isinstance(model, str) else model

    def generate(self, prompt: str) -> str:
        last_exc: Exception | None = None
        for model in self._models:
            logger.info("Generating with %s (%d chars)", model, len(prompt))
            try:
                response = self._client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Generation failed with %s: %s", model, exc)
                last_exc = exc
                continue

            content = response.choices[0].message.content
            if content:
                return content
            logger.warning("Generation returned empty content for %s", model)

        if last_exc is not None:
            raise RuntimeError("All generation model attempts failed") from last_exc
        raise RuntimeError("All generation model attempts returned empty content")
