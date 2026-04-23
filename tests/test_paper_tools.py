"""Tests for source discovery and acquisition helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from phil_mind_rag.agents.paper_tools import (
    ArxivResolver,
    DirectDownloadResolver,
    DownloadJob,
    DownloadResolver,
    SemanticScholarPdfResolver,
    UnpaywallResolver,
    discover_sources,
    download_sources,
    resolve_download_url,
    suggest_papers,
    suggest_sources,
)
from phil_mind_rag.agents.schema import (
    PaperCandidate,
    PaperDiscoveryReport,
    PaperRecommendation,
    SourceCandidate,
    SourceDiscoveryReport,
    SourceRecommendation,
)
from phil_mind_rag.agents.source_search import (
    OpenAIWebSearchProvider,
    OpenAlexSourceProvider,
    SemanticScholarSourceProvider,
    default_source_providers,
)

if TYPE_CHECKING:
    from pathlib import Path


def _source_candidate(
    title: str,
    *,
    source_type: str = "paper",
    download_url: str | None = None,
    access_status: str = "open",
    access_note: str | None = None,
) -> SourceCandidate:
    return SourceCandidate(
        title=title,
        source_type=source_type,
        authors=["Author One"],
        year=1995,
        venue="Mind",
        abstract="A canonical source about consciousness.",
        citation_count=500,
        doi="10.1234/example",
        source_url="https://example.com/source",
        download_url=download_url,
        access_status=access_status,
        access_note=access_note,
    )


def _paper_candidate(title: str, pdf_url: str | None = None) -> PaperCandidate:
    return PaperCandidate(
        title=title,
        authors=["Author One"],
        year=1995,
        venue="Mind",
        abstract="A canonical paper about consciousness.",
        citation_count=500,
        doi="10.1234/paper",
        pdf_url=pdf_url,
        source_url="https://example.com/paper",
    )


def _source_report() -> SourceDiscoveryReport:
    return SourceDiscoveryReport(
        field="philosophy of mind",
        question="What sources matter for the hard problem of consciousness?",
        search_query="hard problem consciousness seminal sources",
        recommendations=[
            SourceRecommendation(
                title="Facing Up to the Problem of Consciousness",
                source_type="paper",
                rationale="Frames the hard problem directly.",
                priority=1,
                relevance_to_question="Direct statement of the target problem.",
                suggested_use="Anchor source for the core framing.",
                source_url="https://example.com/chalmers",
                download_url="https://example.com/chalmers.pdf",
                access_status="open",
                acquisition_note=None,
            ),
            SourceRecommendation(
                title="The Conscious Mind",
                source_type="book",
                rationale="Canonical book-length treatment of the view.",
                priority=2,
                relevance_to_question="Extends and deepens the core argument.",
                suggested_use="Use for extended argument structure and objections.",
                source_url="https://example.com/book",
                download_url=None,
                access_status="copyrighted",
                acquisition_note=(
                    "Important source, but no lawful downloadable copy was available."
                ),
            ),
        ],
        gaps_or_followups=["Add a strong physicalist response for balance."],
    )


def test_suggest_sources_returns_structured_report() -> None:
    report = _source_report()
    client = MagicMock()

    with patch(
        "phil_mind_rag.corpus.discovery.generate_structured",
        return_value=report,
    ) as mocked:
        result = suggest_sources(
            field="philosophy of mind",
            question="What sources matter for the hard problem of consciousness?",
            candidates=[
                _source_candidate(
                    "Facing Up to the Problem of Consciousness",
                    download_url="https://example.com/chalmers.pdf",
                ),
                _source_candidate(
                    "The Conscious Mind",
                    source_type="book",
                    access_status="copyrighted",
                    access_note="Publisher-controlled full text.",
                ),
            ],
            client=client,
            model="gpt-5-mini",
            search_query="hard problem consciousness seminal sources",
        )

    assert result == report
    _, kwargs = mocked.call_args
    assert kwargs["schema_cls"] is SourceDiscoveryReport
    assert "Candidate sources" in kwargs["user"]
    assert "Type: book" in kwargs["user"]
    assert "DOI: 10.1234/example" in kwargs["user"]
    assert "Publisher-controlled full text." in kwargs["user"]


@pytest.mark.parametrize(
    ("field", "question", "candidates"),
    [
        ("", "Q?", [_source_candidate("Source")]),
        ("field", "", [_source_candidate("Source")]),
        ("field", "Q?", []),
    ],
)
def test_suggest_sources_validates_inputs(
    field: str, question: str, candidates: list[SourceCandidate]
) -> None:
    with pytest.raises(ValueError):
        suggest_sources(
            field=field,
            question=question,
            candidates=candidates,
            client=MagicMock(),
            model="gpt-5-mini",
        )


def test_suggest_papers_wraps_generic_source_suggestions() -> None:
    with patch(
        "phil_mind_rag.corpus.discovery.suggest_sources",
        return_value=SourceDiscoveryReport(
            field="philosophy of mind",
            question="Q?",
            search_query="Q?",
            recommendations=[
                SourceRecommendation(
                    title="Facing Up",
                    source_type="paper",
                    rationale="Important.",
                    priority=1,
                    relevance_to_question="Direct.",
                    suggested_use="Start here.",
                    source_url="https://example.com/paper",
                    download_url="https://example.com/paper.pdf",
                    access_status="open",
                    acquisition_note=None,
                )
            ],
            gaps_or_followups=[],
        ),
    ):
        result = suggest_papers(
            field="philosophy of mind",
            question="Q?",
            candidates=[_paper_candidate("Facing Up", "https://example.com/paper.pdf")],
            client=MagicMock(),
            model="gpt-5-mini",
        )

    assert isinstance(result, PaperDiscoveryReport)
    assert isinstance(result.recommendations[0], PaperRecommendation)
    assert result.recommendations[0].pdf_url == "https://example.com/paper.pdf"


def test_discover_sources_merges_and_deduplicates_provider_results() -> None:
    class _Provider:
        def __init__(self, results: list[SourceCandidate]) -> None:
            self._results = results

        def search(self, query: str, limit: int = 5) -> list[SourceCandidate]:
            assert query == "hard problem"
            assert limit == 2
            return self._results

    shared = _source_candidate(
        "Shared Source", download_url="https://example.com/s.pdf"
    )
    web_only = _source_candidate("Web Only", source_type="blog")

    with patch(
        "phil_mind_rag.corpus.discovery.suggest_sources",
        return_value=_source_report(),
    ) as mocked:
        result = discover_sources(
            field="philosophy of mind",
            question="hard problem",
            providers=[_Provider([shared]), _Provider([shared, web_only])],
            client=MagicMock(),
            model="gpt-5-mini",
            per_provider_limit=2,
        )

    assert result == _source_report()
    merged_candidates = mocked.call_args.kwargs["candidates"]
    assert [candidate.title for candidate in merged_candidates] == [
        "Shared Source",
        "Web Only",
    ]


def test_discover_sources_rejects_empty_provider_set() -> None:
    with pytest.raises(ValueError):
        discover_sources(
            field="philosophy of mind",
            question="Q?",
            providers=[],
            client=MagicMock(),
            model="gpt-5-mini",
        )


def test_discover_sources_skips_failed_provider_when_others_return_results() -> None:
    class _BrokenProvider:
        def search(self, query: str, limit: int = 5) -> list[SourceCandidate]:
            raise RuntimeError("rate limited")

    class _WorkingProvider:
        def search(self, query: str, limit: int = 5) -> list[SourceCandidate]:
            return [_source_candidate("Working Source")]

    with patch(
        "phil_mind_rag.corpus.discovery.suggest_sources",
        return_value=_source_report(),
    ) as mocked:
        result = discover_sources(
            field="philosophy of mind",
            question="hard problem",
            providers=[_BrokenProvider(), _WorkingProvider()],
            client=MagicMock(),
            model="gpt-5-mini",
        )

    assert result == _source_report()
    assert [c.title for c in mocked.call_args.kwargs["candidates"]] == [
        "Working Source"
    ]


def test_discover_sources_reports_provider_errors_when_all_fail() -> None:
    class _BrokenProvider:
        def search(self, query: str, limit: int = 5) -> list[SourceCandidate]:
            raise RuntimeError("rate limited")

    with pytest.raises(ValueError, match="provider errors"):
        discover_sources(
            field="philosophy of mind",
            question="hard problem",
            providers=[_BrokenProvider()],
            client=MagicMock(),
            model="gpt-5-mini",
        )


def test_openalex_provider_maps_results() -> None:
    provider = OpenAlexSourceProvider(
        email="me@example.com",
        fetcher=lambda _: {
            "results": [
                {
                    "display_name": "Facing Up",
                    "type": "journal-article",
                    "publication_year": 1995,
                    "cited_by_count": 1234,
                    "authorships": [{"author": {"display_name": "David Chalmers"}}],
                    "primary_location": {
                        "landing_page_url": "https://example.com/facing-up",
                        "source": {"display_name": "Journal of Consciousness Studies"},
                    },
                    "open_access": {"oa_url": "https://example.com/facing-up.pdf"},
                    "doi": "https://doi.org/10.1234/facing-up",
                    "abstract_inverted_index": {
                        "hard": [0],
                        "problem": [1],
                        "consciousness": [3],
                        "of": [2],
                    },
                }
            ]
        },
    )

    results = provider.search("hard problem", limit=1)

    assert results[0].title == "Facing Up"
    assert results[0].source_type == "paper"
    assert results[0].authors == ["David Chalmers"]
    assert results[0].download_url == "https://example.com/facing-up.pdf"
    assert results[0].doi == "10.1234/facing-up"
    assert results[0].abstract == "hard problem of consciousness"


def test_semantic_scholar_provider_maps_results() -> None:
    provider = SemanticScholarSourceProvider(
        fetcher=lambda _: {
            "data": [
                {
                    "title": "Facing Up",
                    "authors": [{"name": "David Chalmers"}],
                    "year": 1995,
                    "venue": "Journal of Consciousness Studies",
                    "abstract": "A paper about the hard problem.",
                    "citationCount": 4321,
                    "url": "https://semanticscholar.org/paper/123",
                    "externalIds": {"DOI": "10.1234/facing-up"},
                    "openAccessPdf": {"url": "https://example.com/facing-up.pdf"},
                    "isOpenAccess": True,
                }
            ]
        },
    )

    results = provider.search("hard problem", limit=1)

    assert results[0].title == "Facing Up"
    assert results[0].source_type == "paper"
    assert results[0].authors == ["David Chalmers"]
    assert results[0].doi == "10.1234/facing-up"
    assert results[0].citation_count == 4321
    assert results[0].download_url == "https://example.com/facing-up.pdf"
    assert results[0].access_status == "open"


def test_openai_web_search_provider_maps_results() -> None:
    provider = OpenAIWebSearchProvider(
        client=MagicMock(),
        model="gpt-5-mini",
        responder=lambda **_: MagicMock(
            output_text=(
                '[{"title":"A Helpful Blog Post","source_type":"blog","authors":[],'
                '"year":null,"venue":"Example Substack",'
                '"abstract":"A strong summary of the debate.",'
                '"citation_count":null,'
                '"source_url":"https://example.substack.com/p/consciousness",'
                '"download_url":null,"access_status":"open","access_note":null},'
                '{"title":"Book Listing","source_type":"book","authors":[],'
                '"year":null,"venue":"Google Books",'
                '"abstract":"Canonical book entry.","citation_count":null,'
                '"source_url":"https://books.google.com/example",'
                '"download_url":null,"access_status":"copyrighted",'
                '"access_note":"No downloadable full text."}]'
            )
        ),
    )

    results = provider.search("hard problem", limit=2)

    assert results[0].source_type == "blog"
    assert results[0].source_url == "https://example.substack.com/p/consciousness"
    assert results[1].source_type == "book"


def test_default_source_providers_includes_openai_web_search() -> None:
    settings = MagicMock()
    settings.openalex_email = "me@example.com"
    settings.openai_api_key.get_secret_value.return_value = "test-api-key"
    settings.openai_web_search_model = "gpt-5-mini"

    providers = default_source_providers(settings)
    names = {provider.__class__.__name__ for provider in providers}

    assert "OpenAlexSourceProvider" in names
    assert "SemanticScholarSourceProvider" in names
    assert "OpenAIWebSearchProvider" in names


def test_download_sources_downloads_open_sources_in_order(tmp_path: Path) -> None:
    jobs = [
        DownloadJob(
            title="Paper One",
            source_url="https://example.com/one",
            download_url="https://example.com/one.pdf",
        ),
        DownloadJob(
            title="Paper Two",
            source_url="https://example.com/two",
            download_url="https://example.com/two.pdf",
        ),
    ]

    def fetcher(url: str) -> bytes:
        return f"%PDF {url}".encode()

    results = download_sources(jobs, tmp_path, max_workers=2, fetcher=fetcher)

    assert [result.title for result in results] == ["Paper One", "Paper Two"]
    assert all(result.success for result in results)
    assert [result.status for result in results] == ["downloaded", "downloaded"]
    assert [result.state for result in results] == ["downloaded", "downloaded"]
    assert results[0].path == tmp_path / "Paper_One.pdf"
    assert results[0].path is not None and results[0].path.read_bytes().startswith(
        b"%PDF"
    )


def test_download_sources_optionally_ingests(tmp_path: Path) -> None:
    pipeline = MagicMock()
    pipeline.ingest.return_value = 12

    results = download_sources(
        [
            DownloadJob(
                title="Important Paper",
                source_url="https://example.com/p",
                download_url="https://example.com/p.pdf",
            )
        ],
        tmp_path,
        pipeline=pipeline,
        fetcher=lambda _: b"%PDF-1.4 fake",
    )

    assert results[0].success is True
    assert results[0].status == "downloaded"
    assert results[0].state == "indexed"
    assert results[0].ingested_chunks == 12
    pipeline.ingest.assert_called_once()


def test_download_sources_marks_paywalled_or_copyrighted_sources(
    tmp_path: Path,
) -> None:
    results = download_sources(
        [
            DownloadJob(
                title="Important Book",
                source_url="https://example.com/book",
                download_url=None,
                access_status="copyrighted",
                access_note="No lawful downloadable copy was available.",
            ),
            DownloadJob(
                title="Journal Article",
                source_url="https://example.com/article",
                download_url=None,
                access_status="paywalled",
            ),
        ],
        tmp_path,
    )

    assert results[0].skipped is True
    assert results[0].status == "skipped"
    assert results[0].state == "skipped_paywalled"
    assert results[0].skip_reason == "No lawful downloadable copy was available."
    assert results[1].skipped is True
    assert results[1].state == "skipped_paywalled"
    assert "paywalled" in (results[1].skip_reason or "")


def test_download_sources_captures_fetch_errors(tmp_path: Path) -> None:
    results = download_sources(
        [
            DownloadJob(
                title="Broken",
                source_url="https://example.com/broken",
                download_url="https://example.com/broken.pdf",
            )
        ],
        tmp_path,
        fetcher=lambda _: (_ for _ in ()).throw(RuntimeError("network failed")),
    )

    assert results[0].success is False
    assert results[0].status == "failed"
    assert results[0].state == "failed"
    assert results[0].error == "network failed"
    assert results[0].path is None


def test_download_sources_validates_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        download_sources([], tmp_path)

    with pytest.raises(ValueError):
        download_sources(
            [
                DownloadJob(
                    title="Paper",
                    source_url="https://example.com/p",
                    download_url="https://example.com/p.pdf",
                )
            ],
            tmp_path,
            max_workers=0,
        )


def test_download_sources_skips_existing_pdf_without_refetching(tmp_path: Path) -> None:
    destination = tmp_path / "Existing.pdf"
    destination.write_bytes(b"%PDF already here")

    results = download_sources(
        [
            DownloadJob(
                title="Existing",
                download_url="https://example.com/existing.pdf",
            )
        ],
        tmp_path,
        fetcher=lambda _: (_ for _ in ()).throw(RuntimeError("should not fetch")),
    )

    assert results[0].success is True
    assert results[0].status == "already-indexed"
    assert results[0].state == "duplicate"
    assert results[0].skipped is True
    assert results[0].path == destination


def test_download_job_defaults_to_selected_state() -> None:
    assert DownloadJob(title="Candidate").state == "selected"


def test_resolve_download_url_uses_expected_fallback_order() -> None:
    job = DownloadJob(
        title="Facing Up",
        doi="10.1234/facing-up",
        source_url="https://arxiv.org/abs/2501.12345",
    )
    resolvers: list[DownloadResolver] = [
        DirectDownloadResolver(),
        UnpaywallResolver(
            fetcher=lambda _: {"best_oa_location": {"url_for_pdf": None}}
        ),
        ArxivResolver(),
        SemanticScholarPdfResolver(
            fetcher=lambda _: {
                "openAccessPdf": {"url": "https://example.com/semantic.pdf"}
            }
        ),
    ]

    assert (
        resolve_download_url(job, resolvers) == "https://arxiv.org/pdf/2501.12345.pdf"
    )


def test_resolve_download_url_uses_unpaywall_before_arxiv() -> None:
    job = DownloadJob(
        title="Facing Up",
        doi="10.1234/facing-up",
        source_url="https://arxiv.org/abs/2501.12345",
    )

    result = resolve_download_url(
        job,
        [
            DirectDownloadResolver(),
            UnpaywallResolver(
                fetcher=lambda _: {
                    "best_oa_location": {
                        "url_for_pdf": "https://example.com/unpaywall.pdf"
                    }
                }
            ),
            ArxivResolver(),
        ],
    )

    assert result == "https://example.com/unpaywall.pdf"


def test_resolve_download_url_uses_semantic_scholar_after_other_sources() -> None:
    job = DownloadJob(title="Facing Up", doi="10.1234/facing-up")

    result = resolve_download_url(
        job,
        [
            DirectDownloadResolver(),
            UnpaywallResolver(fetcher=lambda _: {"best_oa_location": None}),
            ArxivResolver(),
            SemanticScholarPdfResolver(
                fetcher=lambda _: {
                    "openAccessPdf": {"url": "https://example.com/semantic.pdf"}
                }
            ),
        ],
    )

    assert result == "https://example.com/semantic.pdf"
