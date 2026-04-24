"""Tests for settings validation and provider selection."""

from __future__ import annotations

from typing import Any, cast

import pytest
from pydantic import SecretStr, ValidationError

from phil_mind_rag.config import Settings


def _settings_without_env(**kwargs: object) -> Settings:
    settings_cls = cast("Any", Settings)
    return settings_cls(_env_file=None, **kwargs)


def test_requires_explicit_provider_selection() -> None:
    with pytest.raises(ValidationError) as excinfo:
        _settings_without_env()

    message = str(excinfo.value)
    assert "llm_provider" in message
    assert "embedding_provider" in message


def test_requires_openrouter_key_when_selected() -> None:
    with pytest.raises(ValidationError) as excinfo:
        _settings_without_env(
            llm_provider="openrouter",
            embedding_provider="sentence_transformers",
        )

    assert "OPENROUTER_API_KEY must be set when LLM_PROVIDER=openrouter" in str(
        excinfo.value
    )


def test_requires_openai_key_when_openai_embedding_selected() -> None:
    with pytest.raises(ValidationError) as excinfo:
        _settings_without_env(
            llm_provider="openrouter",
            openrouter_api_key=SecretStr("or-key"),
            embedding_provider="openai",
        )

    assert "OPENAI_API_KEY must be set when EMBEDDING_PROVIDER=openai" in str(
        excinfo.value
    )


def test_accepts_hf_demo_profile() -> None:
    settings = _settings_without_env(
        llm_provider="openrouter",
        openrouter_api_key=SecretStr("or-key"),
        embedding_provider="openrouter_free_auto",
    )

    assert settings.llm_provider == "openrouter"
    assert settings.embedding_provider == "openrouter_free_auto"
