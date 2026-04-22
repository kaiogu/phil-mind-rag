"""Compatibility exports for corpus discovery and acquisition helpers."""

from __future__ import annotations

from phil_mind_rag.corpus.acquisition import (
    DownloadJob,
    DownloadResolver,
    DownloadResult,
    DownloadStatus,
    download_papers,
    download_sources,
)
from phil_mind_rag.corpus.discovery import (
    discover_sources,
    suggest_papers,
    suggest_sources,
)
from phil_mind_rag.corpus.resolvers import (
    ArxivResolver,
    DirectDownloadResolver,
    SemanticScholarPdfResolver,
    UnpaywallResolver,
    default_download_resolvers,
    resolve_download_url,
)

__all__ = [
    "ArxivResolver",
    "DirectDownloadResolver",
    "DownloadJob",
    "DownloadResolver",
    "DownloadResult",
    "DownloadStatus",
    "SemanticScholarPdfResolver",
    "UnpaywallResolver",
    "default_download_resolvers",
    "discover_sources",
    "download_papers",
    "download_sources",
    "resolve_download_url",
    "suggest_papers",
    "suggest_sources",
]
