"""Opt-in smoke tests for paid external source-discovery providers."""

from __future__ import annotations

import pytest
from openai import OpenAI

from phil_mind_rag.agents.paper_tools import discover_sources
from phil_mind_rag.agents.source_search import (
    OpenAIWebSearchProvider,
    default_source_providers,
)
from phil_mind_rag.config import get_settings

pytestmark = pytest.mark.paid_api


def test_openai_web_search_provider_live() -> None:
    settings = get_settings()
    provider = OpenAIWebSearchProvider(
        client=OpenAI(api_key=settings.openai_api_key.get_secret_value()),
        model=settings.openai_web_search_model,
    )

    results = provider.search("hard problem consciousness seminal sources", limit=1)

    assert results
    assert results[0].title
    assert results[0].source_type


def test_discover_sources_live() -> None:
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key.get_secret_value())
    providers = default_source_providers(settings)

    report = discover_sources(
        field="philosophy of mind",
        question=(
            "What are the most important sources on the hard problem of consciousness?"
        ),
        search_query="hard problem consciousness seminal sources",
        providers=providers,
        client=client,
        model=settings.openai_chat_model,
        per_provider_limit=1,
    )

    assert report.recommendations
    assert report.search_query == "hard problem consciousness seminal sources"
