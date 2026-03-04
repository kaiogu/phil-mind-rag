"""Shared fixtures for the test suite."""

import pytest

from phil_mind_rag.ingestion.chunker import Chunk
from phil_mind_rag.ingestion.parser import ParsedDocument, Section
from phil_mind_rag.retrieval.store import RetrievalResult


@pytest.fixture
def sample_section() -> Section:
    return Section(
        title="Introduction",
        text="Consciousness is what makes the mind-body problem really intractable.",
        metadata={},
    )


@pytest.fixture
def sample_document(sample_section: Section) -> ParsedDocument:
    return ParsedDocument(source="test.pdf", sections=[sample_section])


@pytest.fixture
def sample_chunk() -> Chunk:
    return Chunk(
        text="Some philosophical text.",
        metadata={"source": "test.pdf", "section": "Introduction"},
    )


@pytest.fixture
def sample_retrieval_result() -> RetrievalResult:
    return RetrievalResult(
        text="Consciousness is a hard problem.",
        score=0.95,
        metadata={"source": "nagel.pdf", "section": "Introduction"},
    )
