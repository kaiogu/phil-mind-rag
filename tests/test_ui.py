"""Tests for Gradio-facing formatting and analysis wiring."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from phil_mind_rag.agents.graph import AnalysisResult
from phil_mind_rag.agents.schema import (
    Claim,
    EvidenceClaim,
    StanceMemo,
    SynthesisReport,
)
from phil_mind_rag.app.ui import (
    create_app,
    handle_analysis,
    handle_extract_metadata,
    handle_refresh,
    handle_upload,
)
from phil_mind_rag.ingestion.chunker import Chunk
from phil_mind_rag.ingestion.metadata_extractor import ExtractedMetadata
from phil_mind_rag.ingestion.registry import DocumentRecord
from phil_mind_rag.retrieval.store import RetrievalResult


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
    )

    with (
        patch("phil_mind_rag.app.ui._get_pipeline"),
        patch("phil_mind_rag.app.ui._get_settings"),
        patch("phil_mind_rag.app.ui.run_analysis", return_value=result),
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
    assert "Decisive Evidence" in synthesis
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
        patch("phil_mind_rag.app.ui._get_pipeline"),
        patch("phil_mind_rag.app.ui._get_settings"),
        patch(
            "phil_mind_rag.app.ui.run_analysis",
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

    with patch("phil_mind_rag.app.ui._get_pipeline", return_value=pipeline):
        result = handle_extract_metadata("paper.pdf")

    assert result == ("The Conscious Mind", "David Chalmers")


def test_handle_extract_metadata_swallows_errors() -> None:
    pipeline = MagicMock()

    def _raise(_: object) -> ExtractedMetadata:
        raise RuntimeError("boom")

    pipeline.extract_metadata = _raise

    with patch("phil_mind_rag.app.ui._get_pipeline", return_value=pipeline):
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

    with patch("phil_mind_rag.app.ui._get_pipeline", return_value=pipeline):
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

    with patch("phil_mind_rag.app.ui._get_pipeline", return_value=pipeline):
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

    with patch("phil_mind_rag.app.ui._get_pipeline", return_value=pipeline):
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

    with patch("phil_mind_rag.app.ui._get_pipeline", return_value=pipeline):
        updates = list(handle_upload(str(pdf), None, None))

    assert updates == [
        "⏳ **bad.pdf** — Parsing PDF...",
        "Validation error: bad file",
    ]


def test_create_app_builds_expected_tabs() -> None:
    app = create_app()

    assert app.title == "Philosophy of Mind — Multi-Agent RAG"
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
    assert any(
        isinstance(value, str) and "Single-Agent Baseline" in value for value in values
    )
