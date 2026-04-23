"""Tests for embedding backend selection and OpenRouter free-model discovery."""

from unittest.mock import MagicMock, patch

from phil_mind_rag.retrieval.embeddings import (
    build_embedder,
    list_openrouter_embedding_models,
    pick_openrouter_free_embedding_model,
)


def test_pick_openrouter_free_embedding_model_prefers_configured_order() -> None:
    with patch(
        "phil_mind_rag.retrieval.embeddings.list_openrouter_embedding_models",
        return_value=(
            {"id": "openai/text-embedding-3-small", "pricing": {"prompt": "0"}},
            {"id": "qwen/qwen3-embedding-0.6b", "pricing": {"prompt": "0"}},
        ),
    ):
        model = pick_openrouter_free_embedding_model(
            api_key="token",
            base_url="https://openrouter.ai/api/v1",
            preferences=(
                "qwen/qwen3-embedding-0.6b",
                "openai/text-embedding-3-small",
            ),
        )

    assert model == "qwen/qwen3-embedding-0.6b"


def test_build_embedder_falls_back_to_sentence_transformers_when_none_free() -> None:
    settings = MagicMock()
    settings.embedding_provider = "openrouter_free_auto"
    settings.openrouter_api_key.get_secret_value.return_value = "token"
    settings.openrouter_base_url = "https://openrouter.ai/api/v1"
    settings.openrouter_embedding_model_preferences = ("qwen/qwen3-embedding-0.6b",)
    settings.sentence_transformers_embedding_model = (
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    with (
        patch(
            "phil_mind_rag.retrieval.embeddings.pick_openrouter_free_embedding_model",
            return_value=None,
        ),
        patch(
            "phil_mind_rag.retrieval.embeddings.SentenceTransformerEmbedder"
        ) as mock_embedder,
    ):
        build_embedder(settings)

    mock_embedder.assert_called_once_with("sentence-transformers/all-MiniLM-L6-v2")


def test_list_openrouter_embedding_models_is_cached() -> None:
    list_openrouter_embedding_models.cache_clear()
    response = MagicMock()
    response.read.return_value = (
        b'{"data":[{"id":"qwen/qwen3-embedding-0.6b","pricing":{"prompt":"0"}}]}'
    )
    response.__enter__.return_value = response
    response.__exit__.return_value = None

    with patch(
        "phil_mind_rag.retrieval.embeddings.urlopen",
        return_value=response,
    ) as mock_urlopen:
        first = list_openrouter_embedding_models(
            api_key="token",
            base_url="https://openrouter.ai/api/v1",
        )
        second = list_openrouter_embedding_models(
            api_key="token",
            base_url="https://openrouter.ai/api/v1",
        )

    assert first == second
    assert mock_urlopen.call_count == 1
