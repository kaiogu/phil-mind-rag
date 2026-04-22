"""Source discovery orchestration for corpus building."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from phil_mind_rag.agents._llm import generate_structured
from phil_mind_rag.agents.schema import (
    PaperCandidate,
    PaperDiscoveryReport,
    PaperRecommendation,
    SourceCandidate,
    SourceDiscoveryReport,
)
from phil_mind_rag.corpus.formatting import format_candidates

if TYPE_CHECKING:
    from openai import OpenAI

    from phil_mind_rag.agents.source_search import SourceSearchProvider

_DISCOVERY_SYSTEM = (
    "You are a research librarian helping build a high-value source corpus. "
    "Given a field, a user question, and candidate sources, "
    "select the sources most important for answering the question well. "
    "Prefer seminal, directly relevant, and conceptually complementary sources. "
    "Mix canonical papers, books, and high-signal essays or blog posts when useful. "
    "Surface paywalled or copyrighted sources when they matter, "
    "but explain why they cannot be directly acquired when applicable. "
    "Avoid recommending redundant sources unless they represent an essential dispute. "
    "Return strict JSON matching the provided schema."
)

logger = logging.getLogger(__name__)


def suggest_sources(
    *,
    field: str,
    question: str,
    candidates: list[SourceCandidate],
    client: OpenAI,
    model: str,
    search_query: str | None = None,
) -> SourceDiscoveryReport:
    """Rank candidate sources for a field/question pair with structured output."""
    if not field.strip():
        raise ValueError("field must not be blank")
    if not question.strip():
        raise ValueError("question must not be blank")
    if not candidates:
        raise ValueError("candidates must not be empty")

    query = search_query.strip() if search_query and search_query.strip() else question
    user = (
        f"Field: {field.strip()}\n"
        f"Question: {question.strip()}\n"
        f"Search query: {query}\n\n"
        "Candidate sources:\n\n"
        f"{format_candidates(candidates)}"
    )
    return generate_structured(
        client=client,
        model=model,
        system=_DISCOVERY_SYSTEM,
        user=user,
        schema_cls=SourceDiscoveryReport,
    )


def discover_sources(
    *,
    field: str,
    question: str,
    providers: list[SourceSearchProvider],
    client: OpenAI,
    model: str,
    search_query: str | None = None,
    per_provider_limit: int = 5,
) -> SourceDiscoveryReport:
    """Search across providers, deduplicate, then rank sources."""
    if not providers:
        raise ValueError("providers must not be empty")
    if per_provider_limit < 1:
        raise ValueError("per_provider_limit must be at least 1")

    query = search_query.strip() if search_query and search_query.strip() else question
    merged: list[SourceCandidate] = []
    seen: set[str] = set()
    provider_errors: list[str] = []
    for provider in providers:
        try:
            candidates = provider.search(query, limit=per_provider_limit)
        except Exception as exc:  # noqa: BLE001
            provider_name = provider.__class__.__name__
            provider_errors.append(f"{provider_name}: {exc}")
            logger.warning("Source provider %s failed: %s", provider_name, exc)
            continue

        for candidate in candidates:
            key = _candidate_key(candidate)
            if key in seen:
                continue
            seen.add(key)
            merged.append(candidate)

    if not merged:
        if provider_errors:
            raise ValueError(
                "providers returned no candidates; provider errors: "
                + "; ".join(provider_errors)
            )
        raise ValueError("providers returned no candidates")

    return suggest_sources(
        field=field,
        question=question,
        candidates=merged,
        client=client,
        model=model,
        search_query=query,
    )


def suggest_papers(
    *,
    field: str,
    question: str,
    candidates: list[PaperCandidate],
    client: OpenAI,
    model: str,
    search_query: str | None = None,
) -> PaperDiscoveryReport:
    """Backward-compatible wrapper around generic source suggestion."""
    source_report = suggest_sources(
        field=field,
        question=question,
        candidates=[
            SourceCandidate(
                title=candidate.title,
                source_type="paper",
                authors=candidate.authors,
                year=candidate.year,
                venue=candidate.venue,
                abstract=candidate.abstract,
                citation_count=candidate.citation_count,
                source_url=candidate.source_url,
                download_url=candidate.pdf_url,
                doi=candidate.doi,
                access_status="open" if candidate.pdf_url else "unknown",
            )
            for candidate in candidates
        ],
        client=client,
        model=model,
        search_query=search_query,
    )
    return PaperDiscoveryReport(
        field=source_report.field,
        question=source_report.question,
        search_query=source_report.search_query,
        recommendations=[
            PaperRecommendation(
                title=item.title,
                source_type=item.source_type,
                rationale=item.rationale,
                priority=item.priority,
                relevance_to_question=item.relevance_to_question,
                suggested_use=item.suggested_use,
                doi=item.doi,
                pdf_url=item.download_url,
                source_url=item.source_url,
                access_status=item.access_status,
                acquisition_note=item.acquisition_note,
            )
            for item in source_report.recommendations
        ],
        gaps_or_followups=source_report.gaps_or_followups,
    )


def _candidate_key(candidate: SourceCandidate) -> str:
    first_author = candidate.authors[0].strip().lower() if candidate.authors else ""
    return f"{candidate.title.strip().lower()}::{first_author}"
