"""Provider selection helpers for chat and embedding backends."""

from __future__ import annotations

from typing import TYPE_CHECKING

from openai import OpenAI

if TYPE_CHECKING:
    from pydantic import SecretStr

    from phil_mind_rag.config import Settings


def generation_client(settings: Settings) -> OpenAI:
    """Build the configured chat client."""
    if settings.llm_provider == "openrouter":
        api_key = _secret_value(settings.openrouter_api_key, "OPENROUTER_API_KEY")
        return OpenAI(
            api_key=api_key,
            base_url=settings.openrouter_base_url,
            timeout=settings.llm_request_timeout_seconds,
        )

    api_key = _secret_value(settings.openai_api_key, "OPENAI_API_KEY")
    return OpenAI(
        api_key=api_key,
        timeout=settings.llm_request_timeout_seconds,
    )


def generation_model(settings: Settings) -> str:
    """Return the active chat model for the selected provider."""
    if settings.llm_provider == "openrouter":
        return settings.openrouter_chat_model
    return settings.openai_chat_model


def generation_models(settings: Settings) -> tuple[str, ...]:
    """Return the ordered chat-model fallback list for the selected provider."""
    if settings.llm_provider != "openrouter":
        return (settings.openai_chat_model,)

    ordered = [
        settings.openrouter_chat_model,
        *settings.openrouter_chat_model_fallbacks,
    ]
    deduped: list[str] = []
    seen: set[str] = set()
    for model in ordered:
        normalized = model.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return tuple(deduped)


def openai_web_search_client(settings: Settings) -> OpenAI | None:
    """Return an OpenAI client for web search, if configured."""
    if settings.openai_api_key is None:
        return None
    return OpenAI(
        api_key=settings.openai_api_key.get_secret_value(),
        timeout=settings.llm_request_timeout_seconds,
    )


def _secret_value(secret: SecretStr | None, env_var: str) -> str:
    if secret is None:
        raise ValueError(f"{env_var} must be set for the selected provider")
    return secret.get_secret_value()
