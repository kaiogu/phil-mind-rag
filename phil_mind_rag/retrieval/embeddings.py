"""Embedding backends for retrieval."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import TYPE_CHECKING, Any
from urllib.request import Request, urlopen

if TYPE_CHECKING:
    from openai import OpenAI
    from pydantic import SecretStr

    from phil_mind_rag.config import Settings

logger = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    """Interface for embedding text into vectors."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the configured model name."""

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string."""

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text strings."""


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI-compatible embedding backend."""

    def __init__(self, client: OpenAI, model: str) -> None:
        self._client = client
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    def embed_text(self, text: str) -> list[float]:
        response = self._client.embeddings.create(
            input=text,
            model=self._model,
        )
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(
            input=texts,
            model=self._model,
        )
        return [item.embedding for item in response.data]


class SentenceTransformerEmbedder(BaseEmbedder):
    """Local sentence-transformers embedding backend."""

    def __init__(self, model: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = model
        self._embedder = SentenceTransformer(model)

    @property
    def model_name(self) -> str:
        return self._model

    def embed_text(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._embedder.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()


def build_embedder(
    settings: Settings, openai_client: OpenAI | None = None
) -> BaseEmbedder:
    """Build the configured embedding backend, including auto-fallback logic."""
    if settings.embedding_provider == "sentence_transformers":
        return SentenceTransformerEmbedder(
            settings.sentence_transformers_embedding_model
        )

    if settings.embedding_provider == "openrouter_free_auto":
        model = pick_openrouter_free_embedding_model(
            api_key=_secret_value(settings.openrouter_api_key, "OPENROUTER_API_KEY"),
            base_url=settings.openrouter_base_url,
            preferences=settings.openrouter_embedding_model_preferences,
        )
        if model is not None:
            client = openai_client
            if client is None:
                from openai import OpenAI

                client = OpenAI(
                    api_key=_secret_value(
                        settings.openrouter_api_key, "OPENROUTER_API_KEY"
                    ),
                    base_url=settings.openrouter_base_url,
                )
            logger.info("Using OpenRouter free embedding model: %s", model)
            return OpenAIEmbedder(client, model)

        logger.warning(
            "No free OpenRouter embedding model found; falling back to %s",
            settings.sentence_transformers_embedding_model,
        )
        return SentenceTransformerEmbedder(
            settings.sentence_transformers_embedding_model
        )

    if openai_client is None:
        from openai import OpenAI

        openai_client = OpenAI(
            api_key=_secret_value(settings.openai_api_key, "OPENAI_API_KEY")
        )
    return OpenAIEmbedder(openai_client, settings.openai_embedding_model)


def pick_openrouter_free_embedding_model(
    *,
    api_key: str,
    base_url: str,
    preferences: tuple[str, ...],
) -> str | None:
    """Return the preferred zero-cost OpenRouter embedding model, if any."""
    models = list_openrouter_embedding_models(api_key=api_key, base_url=base_url)
    free_models = [
        model_id
        for item in models
        if _is_free_model(item) and isinstance((model_id := item.get("id")), str)
    ]
    for model in preferences:
        if model in free_models:
            return model
        free_variant = f"{model}:free"
        if free_variant in free_models:
            return free_variant
    return free_models[0] if free_models else None


@lru_cache(maxsize=8)
def list_openrouter_embedding_models(
    *,
    api_key: str,
    base_url: str,
) -> tuple[dict[str, Any], ...]:
    """Fetch and cache OpenRouter embedding models."""
    root = base_url.removesuffix("/").removesuffix("/api/v1")
    request = Request(  # noqa: S310
        f"{root}/api/v1/embeddings/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    with urlopen(request, timeout=15) as response:  # noqa: S310
        payload = json.loads(response.read().decode("utf-8"))
    data = payload.get("data", [])
    if not isinstance(data, list):
        raise ValueError("OpenRouter embeddings models response did not contain a list")
    return tuple(item for item in data if isinstance(item, dict))


def _is_free_model(model: dict[str, Any]) -> bool:
    pricing = model.get("pricing")
    if not isinstance(pricing, dict):
        return False
    prompt = pricing.get("prompt")
    return isinstance(prompt, str) and prompt == "0"


def _secret_value(secret: SecretStr | None, env_var: str) -> str:
    if secret is None:
        raise ValueError(f"{env_var} must be set for the selected provider")
    return secret.get_secret_value()
