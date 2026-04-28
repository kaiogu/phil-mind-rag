"""Tests for Gradio-facing formatting and analysis wiring."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from phil_mind_rag.agents.graph import AnalysisResult
from phil_mind_rag.agents.schema import (
    ArgumentMap,
    ArgumentMapClaim,
    ArgumentMapStance,
    AtomicClaim,
    Claim,
    EvidenceClaim,
    SourceDiscoveryReport,
    SourceRecommendation,
    StanceMemo,
    SynthesisReport,
    VerifiedClaim,
)
from phil_mind_rag.app.callbacks import (
    handle_acquire_sources,
    handle_analysis,
    handle_discover_sources,
    handle_extract_metadata,
    handle_refresh,
    handle_upload,
)
from phil_mind_rag.app.ui import create_app
from phil_mind_rag.ingestion.chunker import Chunk
from phil_mind_rag.ingestion.metadata_extractor import ExtractedMetadata
from phil_mind_rag.ingestion.registry import DocumentRecord
from phil_mind_rag.retrieval.store import RetrievalResult

if TYPE_CHECKING:
    from pathlib import Path


def _source_report() -> SourceDiscoveryReport:
    return SourceDiscoveryReport(
        field="philosophy of mind",
        question="What are the most important sources on consciousness?",
        search_query="consciousness seminal papers books blogs",
        recommendations=[
            SourceRecommendation(
                title="Facing Up to the Problem of Consciousness",
                source_type="paper",
                rationale="Canonical framing of the hard problem.",
                priority=1,
                relevance_to_question="Directly addresses the question.",
                suggested_use="Anchor text for the central distinction.",
                source_url="https://example.com/facing-up",
                download_url="https://example.com/facing-up.pdf",
                access_status="open",
                acquisition_note=None,
            ),
            SourceRecommendation(
                title="The Conscious Mind",
                source_type="book",
                rationale="Extended treatment of the argument.",
                priority=2,
                relevance_to_question="Deepens the framing and objections.",
                suggested_use="Use for book-length argument structure.",
                source_url="https://example.com/the-conscious-mind",
                download_url=None,
                access_status="copyrighted",
                acquisition_note="No lawful downloadable copy was available.",
            ),
        ],
        gaps_or_followups=["Add a strong physicalist reply."],
    )


def _memo(stance: str) -> StanceMemo:
    return StanceMemo(
        stance=stance,
        thesis=f"{stance} thesis",
        supporting_claims=[
            EvidenceClaim(text=f"{stance} support", citations=["chunk_0"])
        ],
        rival_critiques=[
            EvidenceClaim(text=f"{stance} critique", citations=["chunk_1"])
        ],
        confidence=0.7,
        uncertainty_notes="Some uncertainty remains.",
    )


def _report() -> SynthesisReport:
    return SynthesisReport(
        question="What is consciousness?",
        areas_of_disagreement=["reduction", "qualia"],
        strongest_arguments={"materialist": "Neural evidence is strong."},
        supported_claims=[
            Claim(
                text="Materialism has empirical support.",
                stance="materialist",
                supported=True,
                citations=["chunk_0"],
                source_chunk_id="chunk_0",
                note="Directly grounded.",
            )
        ],
        unsupported_claims=[
            Claim(
                text="Idealism explains everything.",
                stance="idealist",
                supported=False,
                citations=["chunk_9"],
                source_chunk_id=None,
                note="Unknown chunk citation.",
            )
        ],
        synthesis="The dispute remains open.",
        decisive_chunks=["chunk_0"],
        source_chunks_used=["chunk_0", "chunk_1"],
    )


def test_handle_analysis_returns_baseline_and_audit_sections() -> None:
    result = AnalysisResult(
        baseline_answer="Baseline answer",
        report=_report(),
        chunks=[
            RetrievalResult(
                text="ctx 0",
                score=0.9,
                metadata={"source": "a", "section": "s1"},
            ),
            RetrievalResult(
                text="ctx 1",
                score=0.8,
                metadata={"source": "b", "section": "s2"},
            ),
        ],
        materialist_memo=_memo("materialist"),
        idealist_memo=_memo("idealist"),
        dualist_memo=_memo("dualist"),
        verified_claims=[
            VerifiedClaim(
                claim=AtomicClaim(
                    text="Materialism has empirical support.",
                    source="synthesis_supported",
                    stance="materialist",
                    citations=["chunk_0"],
                ),
                label="ambiguous",
                citations=["chunk_0"],
                note="Citations are structurally valid.",
            )
        ],
        argument_map=ArgumentMap(
            question="What is consciousness?",
            disagreement_axes=["reduction", "qualia"],
            stances=[
                ArgumentMapStance(
                    stance="materialist",
                    thesis="materialist thesis",
                    strongest_argument="Neural evidence is strong.",
                    supporting_claims=[
                        ArgumentMapClaim(
                            text="Materialism has empirical support.",
                            citations=["chunk_0"],
                            supported=True,
                            note="Directly grounded.",
                        )
                    ],
                    objections=[],
                )
            ],
            decisive_chunks=["chunk_0"],
            synthesis="The dispute remains open.",
        ),
    )

    with (
        patch("phil_mind_rag.app.callbacks.get_pipeline"),
        patch("phil_mind_rag.app.callbacks.get_settings"),
        patch("phil_mind_rag.app.callbacks.run_analysis", return_value=result),
    ):
        baseline, materialist, idealist, dualist, grounding, synthesis, sources = (
            handle_analysis("What is consciousness?")
        )

    assert baseline == "Baseline answer"
    assert "materialist support" in materialist
    assert "idealist support" in idealist
    assert "dualist support" in dualist
    assert "Supported Claims" in grounding
    assert "Flagged Claims" in grounding
    assert "Claim Verification Audit" in grounding
    assert "Materialism has empirical support." in grounding
    assert "Decisive Evidence" in synthesis
    assert "Argument Map" in synthesis
    assert "Materialist Position" in synthesis
    assert "chunk_0" in sources


def test_handle_analysis_rejects_blank_question() -> None:
    result = handle_analysis("   ")
    assert result == (
        "Please enter a question.",
        "Please enter a question.",
        "Please enter a question.",
        "Please enter a question.",
        "Please enter a question.",
        "Please enter a question.",
        "",
    )


def test_handle_analysis_returns_input_error() -> None:
    with (
        patch("phil_mind_rag.app.callbacks.get_pipeline"),
        patch("phil_mind_rag.app.callbacks.get_settings"),
        patch(
            "phil_mind_rag.app.callbacks.run_analysis",
            side_effect=ValueError("bad question"),
        ),
    ):
        result = handle_analysis("Q?")

    assert result == (
        "Input error: bad question",
        "Input error: bad question",
        "Input error: bad question",
        "Input error: bad question",
        "Input error: bad question",
        "Input error: bad question",
        "",
    )


def test_handle_extract_metadata_returns_empty_for_none() -> None:
    assert handle_extract_metadata(None) == ("", "")


def test_handle_extract_metadata_returns_title_and_author() -> None:
    pipeline = MagicMock()
    pipeline.extract_metadata = lambda _: ExtractedMetadata(
        title="The Conscious Mind",
        author="David Chalmers",
        method="pdf_metadata",
    )

    with patch("phil_mind_rag.app.callbacks.get_pipeline", return_value=pipeline):
        result = handle_extract_metadata("paper.pdf")

    assert result == ("The Conscious Mind", "David Chalmers")


def test_handle_extract_metadata_swallows_errors() -> None:
    pipeline = MagicMock()

    def _raise(_: object) -> ExtractedMetadata:
        raise RuntimeError("boom")

    pipeline.extract_metadata = _raise

    with patch("phil_mind_rag.app.callbacks.get_pipeline", return_value=pipeline):
        assert handle_extract_metadata("paper.pdf") == ("", "")


def test_handle_refresh_maps_registry_records() -> None:
    record = DocumentRecord(
        source="paper.pdf",
        title="The Conscious Mind",
        author="David Chalmers",
        chunk_count=12,
        chunk_size=512,
        chunk_overlap=64,
        chunker="phil_mind_rag.ingestion.chunker.SectionAwareChunker",
        embedding_model="text-embedding-3-small",
        ingested_at="2026-04-18T12:34:56+00:00",
    )
    pipeline = MagicMock()
    pipeline.list_documents = lambda: [record]

    with patch("phil_mind_rag.app.callbacks.get_pipeline", return_value=pipeline):
        result = handle_refresh()

    assert result == [
        [
            "The Conscious Mind",
            "David Chalmers",
            "paper.pdf",
            12,
            512,
            64,
            "SectionAwareChunker",
            "text-embedding-3-small",
            "2026-04-18 12:34:56",
        ]
    ]


def test_handle_upload_returns_message_when_no_file() -> None:
    assert list(handle_upload(None, None, None)) == ["No file uploaded."]


def test_handle_upload_yields_progress_and_success(tmp_path) -> None:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF")

    pipeline = MagicMock()
    pipeline.parse = lambda path: {"path": path}
    pipeline.chunk = lambda document: [
        Chunk(text="chunk one", metadata={}),
        Chunk(text="chunk two", metadata={}),
    ]
    pipeline.embed = lambda chunks: [[0.1], [0.2]]
    stored: dict[str, object] = {}

    def _store(chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        stored["chunks"] = chunks
        stored["embeddings"] = embeddings

    def _register(
        path: object,
        count: int,
        title: str | None,
        author: str | None,
    ) -> None:
        stored["register"] = (path, count, title, author)

    pipeline.store = _store
    pipeline.register = _register

    with patch("phil_mind_rag.app.callbacks.get_pipeline", return_value=pipeline):
        updates = list(handle_upload(str(pdf), "Custom Title", "Custom Author"))

    assert updates == [
        "⏳ **paper.pdf** — Parsing PDF...",
        "⏳ **paper.pdf** — Chunking sections...",
        "⏳ **paper.pdf** — Embedding 2 chunks...",
        "⏳ **paper.pdf** — Storing in vector database...",
        "✅ Ingested **paper.pdf** — 2 chunks indexed.",
    ]
    assert stored["register"] == (pdf, 2, "Custom Title", "Custom Author")


def test_handle_upload_stops_when_no_chunks(tmp_path) -> None:
    pdf = tmp_path / "empty.pdf"
    pdf.write_bytes(b"%PDF")

    pipeline = MagicMock()
    pipeline.parse = lambda path: {"path": path}
    pipeline.chunk = lambda document: []

    with patch("phil_mind_rag.app.callbacks.get_pipeline", return_value=pipeline):
        updates = list(handle_upload(str(pdf), None, None))

    assert updates == [
        "⏳ **empty.pdf** — Parsing PDF...",
        "⏳ **empty.pdf** — Chunking sections...",
        "⚠️ **empty.pdf** — No chunks produced.",
    ]


def test_handle_upload_returns_validation_error(tmp_path) -> None:
    pdf = tmp_path / "bad.pdf"
    pdf.write_bytes(b"%PDF")

    pipeline = MagicMock()

    def _raise(path: object) -> None:
        raise ValueError("bad file")

    pipeline.parse = _raise

    with patch("phil_mind_rag.app.callbacks.get_pipeline", return_value=pipeline):
        updates = list(handle_upload(str(pdf), None, None))

    assert updates == [
        "⏳ **bad.pdf** — Parsing PDF...",
        "Validation error: bad file",
    ]


def test_handle_discover_sources_requires_field_and_question() -> None:
    result = handle_discover_sources("", " ", "")
    assert result == ("Please enter both a field and a discovery question.", [])


def test_handle_discover_sources_formats_report_and_caches_state() -> None:
    class _OpenAIWebSearchProvider:
        pass

    settings = MagicMock()
    settings.llm_provider = "openai"
    settings.openai_chat_model = "gpt-5-mini"
    settings.openai_web_search_model = "gpt-5-mini"

    with (
        patch("phil_mind_rag.app.callbacks.get_settings", return_value=settings),
        patch(
            "phil_mind_rag.app.callbacks.default_source_providers",
            return_value=[_OpenAIWebSearchProvider()],
        ),
        patch("phil_mind_rag.app.callbacks.generation_client"),
        patch(
            "phil_mind_rag.app.callbacks.discover_sources",
            return_value=_source_report(),
        ),
    ):
        markdown, state = handle_discover_sources(
            "philosophy of mind",
            "What are the most important sources on consciousness?",
            "",
        )

    assert "Facing Up to the Problem of Consciousness" in markdown
    assert "No lawful downloadable copy was available." in markdown
    assert len(state) == 2
    assert state[1]["title"] == "The Conscious Mind"


def test_handle_discover_sources_shows_web_search_note_when_unconfigured() -> None:
    settings = MagicMock()
    settings.llm_provider = "openai"
    settings.openai_chat_model = "gpt-5-mini"
    settings.openai_web_search_model = "gpt-5-mini"

    with (
        patch("phil_mind_rag.app.callbacks.get_settings", return_value=settings),
        patch(
            "phil_mind_rag.app.callbacks.default_source_providers",
            return_value=[object()],
        ),
        patch("phil_mind_rag.app.callbacks.generation_client"),
        patch(
            "phil_mind_rag.app.callbacks.discover_sources",
            return_value=_source_report(),
        ),
    ):
        markdown, _ = handle_discover_sources(
            "philosophy of mind",
            "What are the most important sources on consciousness?",
            "",
        )

    assert "OpenAI web-search provider" in markdown


def test_handle_acquire_sources_requires_discovery_results() -> None:
    assert handle_acquire_sources([]) == "No discovered sources are available yet."


def test_handle_acquire_sources_formats_results(tmp_path: Path) -> None:
    settings = MagicMock()
    settings.source_download_dir = tmp_path / "discovered"
    pipeline = MagicMock()
    recommendations = [item.model_dump() for item in _source_report().recommendations]

    download_results = [
        MagicMock(
            success=True,
            title="Facing Up to the Problem of Consciousness",
            ingested_chunks=12,
            skipped=False,
            skip_reason=None,
            error=None,
        ),
        MagicMock(
            success=False,
            title="The Conscious Mind",
            ingested_chunks=None,
            skipped=True,
            skip_reason="No lawful downloadable copy was available.",
            error=None,
        ),
    ]

    with (
        patch("phil_mind_rag.app.callbacks.get_settings", return_value=settings),
        patch("phil_mind_rag.app.callbacks.get_pipeline", return_value=pipeline),
        patch(
            "phil_mind_rag.app.callbacks.download_sources",
            return_value=download_results,
        ),
    ):
        markdown = handle_acquire_sources(recommendations)

    assert "Downloaded **Facing Up to the Problem of Consciousness**" in markdown
    assert "Skipped **The Conscious Mind**" in markdown


def test_create_app_builds_expected_tabs() -> None:
    app = create_app()

    assert app.title == "Philosophy of Mind — Multi-Agent RAG"
    assert app.enable_queue is True
    config = app.config
    labels = [
        component.get("props", {}).get("label") for component in config["components"]
    ]
    values = [
        component.get("props", {}).get("value") for component in config["components"]
    ]

    assert "Baseline answer" in labels
    assert "Grounding" in labels
    assert "Synthesis" in labels
    assert "Discovery Results" in labels
    assert "Acquisition Status" in labels
    assert "Upload Status" in labels
    assert "Field" in labels
    assert "Discovery Question" in labels
    assert "Search Query Override (optional)" in labels
    assert any(
        isinstance(value, str) and "Single-Agent Baseline" in value for value in values
    )
    assert any(
        isinstance(value, str) and "Manual PDF Upload" in value for value in values
    )
    assert any(
        isinstance(value, str) and "leave the search query override blank" in value
        for value in values
    )


def test_create_app_enables_queue_in_config() -> None:
    app = create_app()

    assert app.config["enable_queue"] is True
