"""Opt-in smoke tests for the HF-ready free-provider path."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from phil_mind_rag.agents.graph import run_analysis
from phil_mind_rag.agents.paper_tools import discover_sources
from phil_mind_rag.agents.source_search import (
    OpenAlexSourceProvider,
    SemanticScholarSourceProvider,
)
from phil_mind_rag.config import Settings, get_settings
from phil_mind_rag.pipeline import RAGPipeline
from phil_mind_rag.providers import generation_client, generation_model

if TYPE_CHECKING:
    from phil_mind_rag.agents.source_search import SourceSearchProvider

pytestmark = pytest.mark.free_api

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _require_openrouter_settings() -> Settings:
    settings = get_settings()
    if settings.llm_provider != "openrouter":
        pytest.skip("free smoke tests require LLM_PROVIDER=openrouter")
    if settings.openrouter_api_key is None:
        pytest.skip("free smoke tests require OPENROUTER_API_KEY")
    return settings


def _free_pipeline_settings(tmp_path: Path) -> Settings:
    base = _require_openrouter_settings()
    return base.model_copy(
        update={
            "embedding_provider": "sentence_transformers",
            "chroma_persist_dir": tmp_path / "chroma",
            "registry_path": tmp_path / "registry.json",
            "chroma_collection_name": "hf_free_smoke",
        }
    )


def test_discover_sources_live_with_free_providers() -> None:
    settings = _require_openrouter_settings()
    client = generation_client(settings)
    providers: list[SourceSearchProvider] = [
        OpenAlexSourceProvider(email=settings.openalex_email),
        SemanticScholarSourceProvider(),
    ]

    report = discover_sources(
        field="philosophy of mind",
        question="What are the most important sources on consciousness?",
        search_query="consciousness seminal papers philosophy of mind",
        providers=providers,
        client=client,
        model=generation_model(settings),
        per_provider_limit=1,
    )

    assert report.recommendations
    assert report.search_query == "consciousness seminal papers philosophy of mind"


def test_pipeline_and_multi_agent_live_with_free_models(tmp_path: Path) -> None:
    settings = _free_pipeline_settings(tmp_path)
    pipeline = RAGPipeline(settings)
    pdf_path = _PROJECT_ROOT / "data" / "raw" / "nagel_bat.pdf"

    chunk_count = pipeline.ingest(pdf_path, title="What Is It Like to Be a Bat?")
    answer, sources = pipeline.query_with_sources(
        "What is Nagel's main argument about consciousness?",
        top_k=3,
    )
    analysis = run_analysis(
        "How does Nagel challenge reductionist accounts of consciousness?",
        pipeline,
        settings,
    )

    assert chunk_count > 0
    assert answer.strip()
    assert sources
    assert analysis.baseline_answer.strip()
    assert analysis.report.synthesis.strip()
    assert analysis.materialist_memo.thesis.strip()
    assert analysis.idealist_memo.thesis.strip()
    assert analysis.dualist_memo.thesis.strip()
