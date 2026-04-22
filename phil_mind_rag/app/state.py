"""Lazy application state for the Gradio frontend."""

from __future__ import annotations

from phil_mind_rag.config import Settings
from phil_mind_rag.config import get_settings as load_settings
from phil_mind_rag.pipeline import RAGPipeline

_pipeline: RAGPipeline | None = None
_settings: Settings | None = None


def get_settings() -> Settings:
    """Return cached application settings."""
    global _settings  # noqa: PLW0603
    if _settings is None:
        _settings = load_settings()
    return _settings


def get_pipeline() -> RAGPipeline:
    """Return the cached RAG pipeline."""
    global _pipeline  # noqa: PLW0603
    if _pipeline is None:
        _pipeline = RAGPipeline(get_settings())
    return _pipeline
