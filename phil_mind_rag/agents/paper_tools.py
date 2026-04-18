"""Tool-style helpers for corpus-building source discovery and download."""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from phil_mind_rag.agents._llm import generate_structured
from phil_mind_rag.agents.schema import (
    PaperCandidate,
    PaperDiscoveryReport,
    PaperRecommendation,
    SourceCandidate,
    SourceDiscoveryReport,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from openai import OpenAI

    from phil_mind_rag.agents.source_search import SourceSearchProvider
    from phil_mind_rag.pipeline import RAGPipeline


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
        f"{_format_candidates(candidates)}"
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
    for provider in providers:
        for candidate in provider.search(query, limit=per_provider_limit):
            key = _candidate_key(candidate)
            if key in seen:
                continue
            seen.add(key)
            merged.append(candidate)

    if not merged:
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
                pdf_url=item.download_url,
                source_url=item.source_url,
                access_status=item.access_status,
                acquisition_note=item.acquisition_note,
            )
            for item in source_report.recommendations
        ],
        gaps_or_followups=source_report.gaps_or_followups,
    )


@dataclass(frozen=True)
class DownloadJob:
    """A requested source acquisition attempt."""

    title: str
    source_url: str | None = None
    download_url: str | None = None
    access_status: str = "unknown"
    access_note: str | None = None


@dataclass(frozen=True)
class DownloadResult:
    """Outcome of a source acquisition attempt, with optional ingestion count."""

    title: str
    source_url: str | None
    download_url: str | None
    path: Path | None
    success: bool
    error: str | None = None
    ingested_chunks: int | None = None
    skipped: bool = False
    skip_reason: str | None = None


def download_sources(
    jobs: list[DownloadJob],
    output_dir: Path,
    *,
    max_workers: int = 4,
    pipeline: RAGPipeline | None = None,
    fetcher: Callable[[str], bytes] | None = None,
) -> list[DownloadResult]:
    """Fetch multiple sources in parallel and optionally ingest downloaded PDFs."""
    if not jobs:
        raise ValueError("jobs must not be empty")
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")

    output_dir.mkdir(parents=True, exist_ok=True)
    fetch = fetcher or _download_pdf

    results: list[DownloadResult | None] = [None] * len(jobs)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_download_one, job, output_dir, pipeline, fetch): i
            for i, job in enumerate(jobs)
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()

    return [result for result in results if result is not None]


def download_papers(
    jobs: list[DownloadJob],
    output_dir: Path,
    *,
    max_workers: int = 4,
    pipeline: RAGPipeline | None = None,
    fetcher: Callable[[str], bytes] | None = None,
) -> list[DownloadResult]:
    """Backward-compatible alias for paper-only download flows."""
    return download_sources(
        jobs,
        output_dir,
        max_workers=max_workers,
        pipeline=pipeline,
        fetcher=fetcher,
    )


def _download_one(
    job: DownloadJob,
    output_dir: Path,
    pipeline: RAGPipeline | None,
    fetcher: Callable[[str], bytes],
) -> DownloadResult:
    try:
        if job.download_url is None:
            return DownloadResult(
                title=job.title,
                source_url=job.source_url,
                download_url=job.download_url,
                path=None,
                success=False,
                skipped=True,
                skip_reason=_skip_reason(job),
            )

        content = fetcher(job.download_url)
        destination = output_dir / _safe_pdf_name(job.title, job.download_url)
        destination.write_bytes(content)
        chunk_count = pipeline.ingest(destination) if pipeline is not None else None
        return DownloadResult(
            title=job.title,
            source_url=job.source_url,
            download_url=job.download_url,
            path=destination,
            success=True,
            ingested_chunks=chunk_count,
        )
    except Exception as exc:  # noqa: BLE001
        return DownloadResult(
            title=job.title,
            source_url=job.source_url,
            download_url=job.download_url,
            path=None,
            success=False,
            error=str(exc),
        )


def _download_pdf(url: str) -> bytes:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("pdf_url must use http or https")
    request = Request(  # noqa: S310
        url,
        headers={
            "User-Agent": "phil-mind-rag/0.1 (+https://example.invalid/phil-mind-rag)"
        },
    )
    with urlopen(request, timeout=30) as response:  # noqa: S310
        return response.read()


def _format_candidates(candidates: list[SourceCandidate]) -> str:
    blocks = []
    for i, candidate in enumerate(candidates, start=1):
        authors = ", ".join(candidate.authors) if candidate.authors else "Unknown"
        citation_count = (
            str(candidate.citation_count)
            if candidate.citation_count is not None
            else "unknown"
        )
        blocks.append(
            "\n".join(
                [
                    f"[source_{i}] {candidate.title}",
                    f"Type: {candidate.source_type}",
                    f"Authors: {authors}",
                    "Year: "
                    f"{candidate.year if candidate.year is not None else 'unknown'}",
                    f"Venue: {candidate.venue or 'unknown'}",
                    f"Citations: {citation_count}",
                    f"Source URL: {candidate.source_url or 'none'}",
                    f"Download URL: {candidate.download_url or 'none'}",
                    f"Access: {candidate.access_status}",
                    f"Access note: {candidate.access_note or 'none'}",
                    f"Abstract: {candidate.abstract or 'none'}",
                ]
            )
        )
    return "\n\n---\n\n".join(blocks)


def _safe_pdf_name(title: str, url: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", title).strip("._")
    if not stem:
        parsed = urlparse(url)
        stem = Path(parsed.path).stem or "paper"
    if not stem.lower().endswith(".pdf"):
        stem = f"{stem}.pdf"
    return stem


def _candidate_key(candidate: SourceCandidate) -> str:
    first_author = candidate.authors[0].strip().lower() if candidate.authors else ""
    return f"{candidate.title.strip().lower()}::{first_author}"


def _skip_reason(job: DownloadJob) -> str:
    if job.access_note:
        return job.access_note
    if job.access_status == "paywalled":
        return "Source surfaced but could not be downloaded because it is paywalled."
    if job.access_status == "copyrighted":
        return (
            "Source surfaced but could not be downloaded due to copyright restrictions."
        )
    return "Source surfaced but no downloadable artifact was available."
