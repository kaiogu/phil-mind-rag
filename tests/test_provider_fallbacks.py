"""Tests for model fallback selection and retry behavior."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

from pydantic import BaseModel

from phil_mind_rag.agents._llm import generate_structured
from phil_mind_rag.generation.llm import OpenAILLM
from phil_mind_rag.providers import generation_models


class _StructuredPayload(BaseModel):
    answer: str


def test_generation_models_deduplicates_openrouter_fallbacks() -> None:
    settings = SimpleNamespace(
        llm_provider="openrouter",
        openrouter_chat_model="openrouter/free",
        openrouter_chat_model_fallbacks=(
            "openrouter/free",
            "meta-llama/llama-3.1-8b-instruct:free",
            "meta-llama/llama-3.1-8b-instruct:free",
        ),
        openai_chat_model="gpt-5-mini",
    )

    assert generation_models(cast("Any", settings)) == (
        "openrouter/free",
        "meta-llama/llama-3.1-8b-instruct:free",
    )


def test_openai_llm_retries_until_a_model_returns_content() -> None:
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        RuntimeError("primary failed"),
        SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Recovered"))]
        ),
    ]

    llm = OpenAILLM(
        client,
        ("openrouter/free", "meta-llama/llama-3.1-8b-instruct:free"),
    )

    assert llm.generate("What is consciousness?") == "Recovered"
    assert client.chat.completions.create.call_count == 2


def test_generate_structured_retries_on_model_failure() -> None:
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        RuntimeError("primary failed"),
        SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content='{"answer":"ok"}'))
            ]
        ),
    ]

    result = generate_structured(
        client=client,
        model=("openrouter/free", "meta-llama/llama-3.1-8b-instruct:free"),
        system="Return JSON",
        user="Question",
        schema_cls=_StructuredPayload,
    )

    assert result.answer == "ok"
    assert client.chat.completions.create.call_count == 2
