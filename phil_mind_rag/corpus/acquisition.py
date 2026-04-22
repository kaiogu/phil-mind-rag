"""Source acquisition jobs and download execution."""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol
from urllib.parse import urlparse
from urllib.request import Request, urlopen

if TYPE_CHECKING:
    from collections.abc import Callable

    from phil_mind_rag.pipeline import RAGPipeline

DownloadStatus = Literal["downloaded", "failed", "already-indexed", "skipped"]


@dataclass(frozen=True)
class DownloadJob:
    """A requested source acquisition attempt."""

    title: str
    source_url: str | None = None
    download_url: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    semantic_scholar_id: str | None = None
    access_status: str = "unknown"
    access_note: str | None = None


class DownloadResolver(Protocol):
    """Resolve a source acquisition job to a downloadable artifact URL."""

    def resolve(self, job: DownloadJob) -> str | None:
        """Return a direct PDF URL if this resolver can find one."""


@dataclass(frozen=True)
class DownloadResult:
    """Outcome of a source acquisition attempt, with optional ingestion count."""

    title: str
    source_url: str | None
    download_url: str | None
    path: Path | None
    success: bool
    status: DownloadStatus
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
    resolvers: list[DownloadResolver] | None = None,
) -> list[DownloadResult]:
    """Fetch multiple sources in parallel and optionally ingest downloaded PDFs."""
    if not jobs:
        raise ValueError("jobs must not be empty")
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")

    from phil_mind_rag.corpus.resolvers import default_download_resolvers

    output_dir.mkdir(parents=True, exist_ok=True)
    fetch = fetcher or download_pdf
    resolver_chain = resolvers or default_download_resolvers()

    results: list[DownloadResult | None] = [None] * len(jobs)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _download_one,
                job,
                output_dir,
                pipeline,
                fetch,
                resolver_chain,
            ): i
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
    resolvers: list[DownloadResolver] | None = None,
) -> list[DownloadResult]:
    """Backward-compatible alias for paper-only download flows."""
    return download_sources(
        jobs,
        output_dir,
        max_workers=max_workers,
        pipeline=pipeline,
        fetcher=fetcher,
        resolvers=resolvers,
    )


def download_pdf(url: str) -> bytes:
    """Download a PDF-like artifact from an HTTP(S) URL."""
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


def _download_one(
    job: DownloadJob,
    output_dir: Path,
    pipeline: RAGPipeline | None,
    fetcher: Callable[[str], bytes],
    resolvers: list[DownloadResolver],
) -> DownloadResult:
    from phil_mind_rag.corpus.resolvers import resolve_download_url

    try:
        download_url = resolve_download_url(job, resolvers)
        if download_url is None:
            return DownloadResult(
                title=job.title,
                source_url=job.source_url,
                download_url=None,
                path=None,
                success=False,
                status="skipped",
                skipped=True,
                skip_reason=_skip_reason(job),
            )

        destination = output_dir / _safe_pdf_name(job.title, download_url)
        if destination.exists():
            return DownloadResult(
                title=job.title,
                source_url=job.source_url,
                download_url=download_url,
                path=destination,
                success=True,
                status="already-indexed",
                skipped=True,
                skip_reason="PDF already exists in the acquisition directory.",
            )

        content = fetcher(download_url)
        destination.write_bytes(content)
        chunk_count = pipeline.ingest(destination) if pipeline is not None else None
        return DownloadResult(
            title=job.title,
            source_url=job.source_url,
            download_url=download_url,
            path=destination,
            success=True,
            status="downloaded",
            ingested_chunks=chunk_count,
        )
    except Exception as exc:  # noqa: BLE001
        return DownloadResult(
            title=job.title,
            source_url=job.source_url,
            download_url=job.download_url,
            path=None,
            success=False,
            status="failed",
            error=str(exc),
        )


def _safe_pdf_name(title: str, url: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", title).strip("._")
    if not stem:
        parsed = urlparse(url)
        stem = Path(parsed.path).stem or "paper"
    if not stem.lower().endswith(".pdf"):
        stem = f"{stem}.pdf"
    return stem


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
