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
    SourceRecommendation,
)
from phil_mind_rag.corpus.formatting import format_candidates

if TYPE_CHECKING:
    from openai import OpenAI

    from phil_mind_rag.agents.source_search import SourceSearchProvider

_DISCOVERY_SYSTEM = """# Identity
You are a research librarian helping build a high-value source corpus.

# Task
Given a field, a research question, and candidate sources, select the sources
most worth adding to the corpus.

# Ranking Criteria
- Prioritize direct relevance to the question.
- Prefer seminal, canonical, or especially clarifying sources when available.
- Prefer sets of sources that are complementary rather than redundant.
- Mix papers, books, and high-signal essays only when each materially improves
  coverage of the question.
- Surface important paywalled or copyrighted sources when they matter, but
  explain clearly why they are not directly acquirable.

# Rules
- Use only the supplied candidates. Do not invent sources.
- Do not overstate metadata you do not have.
- If two sources are near-duplicates, recommend the stronger one unless the
  disagreement between them matters.
- Keep `acquisition_note` concrete and operational.

# Output Requirements
- Return strict JSON matching the provided schema.
- Do not include prose outside the JSON.
"""

logger = logging.getLogger(__name__)


def suggest_sources(
    *,
    field: str,
    question: str,
    candidates: list[SourceCandidate],
    client: OpenAI,
    model: str | tuple[str, ...],
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
        "<field>\n"
        f"{field.strip()}\n"
        "</field>\n\n"
        "<research_question>\n"
        f"{question.strip()}\n"
        "</research_question>\n\n"
        "<search_query>\n"
        f"{query}\n"
        "</search_query>\n\n"
        "<candidate_sources>\n"
        f"{format_candidates(candidates)}\n"
        "</candidate_sources>"
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
    model: str | tuple[str, ...],
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

    try:
        report = suggest_sources(
            field=field,
            question=question,
            candidates=merged,
            client=client,
            model=model,
            search_query=query,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Structured source ranking failed; using fallback: %s", exc)
        return _fallback_source_report(
            field=field,
            question=question,
            search_query=query,
            candidates=merged,
            provider_errors=provider_errors,
            reason=f"Model ranking failed: {exc}",
        )

    if report.recommendations:
        if provider_errors:
            report.gaps_or_followups.extend(
                [
                    f"Provider warning: {error}"
                    for error in provider_errors
                    if f"Provider warning: {error}" not in report.gaps_or_followups
                ]
            )
        return report

    logger.warning(
        "Structured source ranking returned no recommendations; using fallback"
    )
    return _fallback_source_report(
        field=field,
        question=question,
        search_query=query,
        candidates=merged,
        provider_errors=provider_errors,
        reason="Model ranking returned no recommendations.",
    )


def suggest_papers(
    *,
    field: str,
    question: str,
    candidates: list[PaperCandidate],
    client: OpenAI,
    model: str | tuple[str, ...],
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


def _fallback_source_report(
    *,
    field: str,
    question: str,
    search_query: str,
    candidates: list[SourceCandidate],
    provider_errors: list[str],
    reason: str,
) -> SourceDiscoveryReport:
    recommendations = [
        _candidate_to_recommendation(candidate, priority=index)
        for index, candidate in enumerate(
            sorted(candidates, key=_candidate_priority_key)[:5], start=1
        )
    ]
    followups = [reason]
    followups.extend(f"Provider warning: {error}" for error in provider_errors)
    return SourceDiscoveryReport(
        field=field,
        question=question,
        search_query=search_query,
        recommendations=recommendations,
        gaps_or_followups=followups,
    )


def _candidate_to_recommendation(
    candidate: SourceCandidate, *, priority: int
) -> SourceRecommendation:
    authors = (
        ", ".join(candidate.authors[:2]) if candidate.authors else "Unknown author"
    )
    rationale_bits = [
        f"Selected from provider results for {candidate.source_type} coverage."
    ]
    if candidate.citation_count:
        rationale_bits.append(f"Reported citations: {candidate.citation_count}.")
    if candidate.access_status == "open":
        rationale_bits.append("Open access makes acquisition straightforward.")
    elif candidate.access_note:
        rationale_bits.append(candidate.access_note)

    return SourceRecommendation(
        title=candidate.title,
        source_type=candidate.source_type,
        rationale=" ".join(rationale_bits),
        priority=min(priority, 5),
        relevance_to_question=(
            candidate.abstract[:180]
            if candidate.abstract
            else "Relevant candidate from search results."
        ),
        suggested_use=(
            f"Review {candidate.source_type} by {authors} "
            "for corpus coverage and grounding."
        ),
        doi=candidate.doi,
        source_url=candidate.source_url,
        download_url=candidate.download_url,
        access_status=candidate.access_status,
        acquisition_note=candidate.access_note,
    )


def _candidate_priority_key(
    candidate: SourceCandidate,
) -> tuple[int, int, int, str]:
    type_rank = {
        "paper": 0,
        "book": 1,
        "article": 2,
        "blog": 3,
        "video": 4,
    }.get(candidate.source_type, 5)
    citation_rank = -(candidate.citation_count or 0)
    access_rank = 0 if candidate.access_status == "open" else 1
    return (type_rank, access_rank, citation_rank, candidate.title.lower())
