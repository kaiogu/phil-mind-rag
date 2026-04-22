"""Download URL resolvers for source acquisition."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request, urlopen

if TYPE_CHECKING:
    from collections.abc import Callable

    from phil_mind_rag.corpus.acquisition import DownloadJob, DownloadResolver

logger = logging.getLogger(__name__)


def resolve_download_url(
    job: DownloadJob,
    resolvers: list[DownloadResolver] | None = None,
) -> str | None:
    """Resolve a job to a PDF URL using the configured fallback order."""
    for resolver in resolvers or default_download_resolvers():
        try:
            resolved = resolver.resolve(job)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Download resolver %s failed for %s: %s",
                resolver.__class__.__name__,
                job.title,
                exc,
            )
            continue
        if resolved:
            return resolved
    return None


def default_download_resolvers(
    *,
    unpaywall_email: str | None = None,
    semantic_scholar_api_key: str | None = None,
) -> list[DownloadResolver]:
    """Return the KGU-105 PDF resolution order."""
    return [
        DirectDownloadResolver(),
        UnpaywallResolver(email=unpaywall_email),
        ArxivResolver(),
        SemanticScholarPdfResolver(api_key=semantic_scholar_api_key),
    ]


class DirectDownloadResolver:
    """Use an already-known direct PDF URL."""

    def resolve(self, job: DownloadJob) -> str | None:
        if job.download_url and _is_http_url(job.download_url):
            return job.download_url
        return None


class UnpaywallResolver:
    """Resolve a DOI through Unpaywall's open-access location metadata."""

    def __init__(
        self,
        *,
        email: str | None = None,
        fetcher: Callable[[str], dict] | None = None,
    ) -> None:
        self._email = email or "phil-mind-rag@example.invalid"
        self._fetcher = fetcher or fetch_json

    def resolve(self, job: DownloadJob) -> str | None:
        if not job.doi:
            return None
        url = (
            f"https://api.unpaywall.org/v2/{quote(job.doi, safe='')}"
            f"?{urlencode({'email': self._email})}"
        )
        payload = self._fetcher(url)
        location = payload.get("best_oa_location") or {}
        pdf_url = location.get("url_for_pdf")
        if pdf_url and _is_http_url(pdf_url):
            return pdf_url
        return None


class ArxivResolver:
    """Resolve arXiv IDs or arXiv landing URLs to PDF URLs."""

    def resolve(self, job: DownloadJob) -> str | None:
        arxiv_id = job.arxiv_id or _arxiv_id_from_url(job.source_url)
        if not arxiv_id:
            return None
        return f"https://arxiv.org/pdf/{arxiv_id}.pdf"


class SemanticScholarPdfResolver:
    """Resolve open-access PDF metadata from Semantic Scholar."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        fetcher: Callable[[str], dict] | None = None,
    ) -> None:
        self._api_key = api_key
        self._fetcher = fetcher or self._fetch

    def resolve(self, job: DownloadJob) -> str | None:
        paper_id = _semantic_scholar_lookup_id(job)
        if not paper_id:
            return None
        payload = self._fetcher(
            "https://api.semanticscholar.org/graph/v1/paper/"
            f"{quote(paper_id, safe=':')}?fields=openAccessPdf"
        )
        pdf_url = (payload.get("openAccessPdf") or {}).get("url")
        if pdf_url and _is_http_url(pdf_url):
            return pdf_url
        return None

    def _fetch(self, url: str) -> dict:
        headers = {"User-Agent": "phil-mind-rag/0.1"}
        if self._api_key:
            headers["x-api-key"] = self._api_key
        return fetch_json(url, headers=headers)


def fetch_json(url: str, headers: dict[str, str] | None = None) -> dict:
    """Fetch and decode a JSON document."""
    request = Request(  # noqa: S310
        url,
        headers=headers or {"User-Agent": "phil-mind-rag/0.1"},
    )
    with urlopen(request, timeout=30) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def _is_http_url(url: str) -> bool:
    return urlparse(url).scheme in {"http", "https"}


def _arxiv_id_from_url(url: str | None) -> str | None:
    if not url:
        return None
    parsed = urlparse(url)
    if parsed.netloc not in {"arxiv.org", "www.arxiv.org"}:
        return None
    path = parsed.path.strip("/")
    if path.startswith("abs/"):
        return path.removeprefix("abs/")
    if path.startswith("pdf/"):
        return path.removeprefix("pdf/").removesuffix(".pdf")
    return None


def _semantic_scholar_lookup_id(job: DownloadJob) -> str | None:
    if job.semantic_scholar_id:
        return job.semantic_scholar_id
    if job.doi:
        return f"DOI:{job.doi}"
    return None
