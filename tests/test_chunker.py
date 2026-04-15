"""Tests for SectionAwareChunker."""


from phil_mind_rag.ingestion.chunker import SectionAwareChunker
from phil_mind_rag.ingestion.parser import ParsedDocument, Section


def _doc(*sections: tuple[str, str]) -> ParsedDocument:
    """Helper: build a ParsedDocument from (title, text) pairs."""
    return ParsedDocument(
        source="test.pdf",
        sections=[Section(title=t, text=tx, metadata={}) for t, tx in sections],
    )


class TestSectionAwareChunker:
    def test_short_section_produces_single_chunk(self) -> None:
        chunker = SectionAwareChunker(chunk_size=512, chunk_overlap=64)
        doc = _doc(("Intro", "word " * 100))  # 100 words << 512
        assert len(chunker.chunk(doc)) == 1

    def test_long_section_is_split_into_multiple_chunks(self) -> None:
        chunker = SectionAwareChunker(chunk_size=10, chunk_overlap=2)
        doc = _doc(("Body", "word " * 30))  # 30 words >> 10
        assert len(chunker.chunk(doc)) > 1

    def test_empty_section_produces_no_chunks(self) -> None:
        chunker = SectionAwareChunker()
        doc = _doc(("Empty", ""))
        assert chunker.chunk(doc) == []

    def test_empty_document_produces_no_chunks(self) -> None:
        chunker = SectionAwareChunker()
        doc = ParsedDocument(source="test.pdf", sections=[])
        assert chunker.chunk(doc) == []

    def test_chunk_metadata_carries_source_and_section(self) -> None:
        chunker = SectionAwareChunker()
        doc = _doc(("Introduction", "Some meaningful text here."))
        chunk = chunker.chunk(doc)[0]
        assert chunk.metadata["source"] == "test.pdf"
        assert chunk.metadata["section"] == "Introduction"

    def test_multiple_sections_all_appear_in_output(self) -> None:
        chunker = SectionAwareChunker()
        doc = _doc(("Alpha", "Text A."), ("Beta", "Text B."))
        sections_seen = {c.metadata["section"] for c in chunker.chunk(doc)}
        assert {"Alpha", "Beta"} == sections_seen

    def test_overlap_creates_shared_words_between_consecutive_chunks(self) -> None:
        chunker = SectionAwareChunker(chunk_size=5, chunk_overlap=2)
        # 12 words → first chunk [0:5], second [3:8], third [6:11]
        words = list("abcdefghijkl")
        doc = _doc(("Sec", " ".join(words)))
        chunks = chunker.chunk(doc)
        assert len(chunks) >= 2
        tail = set(chunks[0].text.split()[-2:])
        head = set(chunks[1].text.split()[:2])
        assert tail & head  # overlapping words exist

    def test_chunk_text_is_subset_of_original_words(self) -> None:
        chunker = SectionAwareChunker(chunk_size=10, chunk_overlap=2)
        original = "word " * 25
        doc = _doc(("Sec", original))
        original_words = set(original.split())
        for chunk in chunker.chunk(doc):
            assert set(chunk.text.split()).issubset(original_words)

    def test_whitespace_only_section_produces_no_chunks(self) -> None:
        chunker = SectionAwareChunker()
        doc = _doc(("Blank", "   \n  \t  "))
        assert chunker.chunk(doc) == []
