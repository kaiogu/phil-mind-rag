"""External source search providers for corpus discovery."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Protocol
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from openai import OpenAI

from phil_mind_rag.agents.schema import SourceCandidate

if TYPE_CHECKING:
    from collections.abc import Callable

    from phil_mind_rag.config import Settings


class SourceSearchProvider(Protocol):
    """Provider contract for external source discovery."""

    def search(self, query: str, limit: int = 5) -> list[SourceCandidate]:
        """Return candidate sources for the query."""


class OpenAlexSourceProvider:
    """OpenAlex works search for academic books, papers, and related records."""

    def __init__(
        self,
        *,
        email: str | None = None,
        fetcher: Callable[[str], dict] | None = None,
    ) -> None:
        self._email = email
        self._fetcher = fetcher or _fetch_json

    def search(self, query: str, limit: int = 5) -> list[SourceCandidate]:
        params = {
            "search": query,
            "per-page": str(limit),
            "filter": "is_retracted:false",
        }
        if self._email:
            params["mailto"] = self._email
        payload = self._fetcher("https://api.openalex.org/works?" + urlencode(params))
        return [
            _openalex_to_candidate(item)
            for item in payload.get("results", [])
            if item.get("display_name")
        ]


class OpenAIWebSearchProvider:
    """General web search provider backed by the OpenAI web search tool."""

    def __init__(
        self,
        *,
        client: OpenAI,
        model: str,
        responder: Callable[..., Any] | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._responder = responder or client.responses.create

    def search(self, query: str, limit: int = 5) -> list[SourceCandidate]:
        response = self._responder(
            model=self._model,
            input=(
                "Search the web for high-signal sources related to this query: "
                f"{query}\n\n"
                f"Return at most {limit} results as strict JSON with shape "
                '[{"title": str, "source_type": str, "authors": list[str], '
                '"year": int|null, "venue": str|null, "abstract": str, '
                '"citation_count": int|null, "source_url": str|null, '
                '"download_url": str|null, "access_status": str, '
                '"access_note": str|null}]. '
                "Include books, blogs, essays, articles, and papers when useful. "
                "If a source appears paywalled or copyrighted, set access_status "
                'to "paywalled" or "copyrighted" and explain in access_note.'
            ),
            tools=[{"type": "web_search_preview"}],
        )
        return [
            SourceCandidate.model_validate(item)
            for item in json.loads(response.output_text)
            if item.get("title")
        ]


def default_source_providers(settings: Settings) -> list[SourceSearchProvider]:
    """Build the configured discovery providers for this environment."""
    client = OpenAI(api_key=settings.openai_api_key.get_secret_value())
    providers: list[SourceSearchProvider] = [
        OpenAlexSourceProvider(email=settings.openalex_email),
        OpenAIWebSearchProvider(
            client=client,
            model=settings.openai_web_search_model,
        ),
    ]
    return providers


def _openalex_to_candidate(item: dict) -> SourceCandidate:
    primary_location = item.get("primary_location") or {}
    source = primary_location.get("source") or {}
    open_access = item.get("open_access") or {}
    host_venue = source.get("display_name")
    landing_url = primary_location.get("landing_page_url")
    pdf_url = open_access.get("oa_url") or primary_location.get("pdf_url")
    source_type = "paper"
    if item.get("type") == "book":
        source_type = "book"
    elif item.get("type") == "book-chapter":
        source_type = "book"
    return SourceCandidate(
        title=item["display_name"],
        source_type=source_type,
        authors=[
            authorship.get("author", {}).get("display_name", "")
            for authorship in item.get("authorships", [])
            if authorship.get("author", {}).get("display_name")
        ],
        year=item.get("publication_year"),
        venue=host_venue,
        abstract=_openalex_abstract(item),
        citation_count=item.get("cited_by_count"),
        source_url=landing_url,
        download_url=pdf_url,
        access_status="open" if pdf_url else "unknown",
        access_note=None,
    )


def _openalex_abstract(item: dict) -> str:
    inverted = item.get("abstract_inverted_index")
    if not inverted:
        return ""
    words = [
        (position, token)
        for token, positions in inverted.items()
        for position in positions
    ]
    return " ".join(token for _, token in sorted(words))


def _fetch_json(url: str) -> dict:
    request = Request(  # noqa: S310
        url,
        headers={"User-Agent": "phil-mind-rag/0.1"},
    )
    with urlopen(request, timeout=30) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))
