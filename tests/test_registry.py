"""Tests for DocumentRegistry."""

from pathlib import Path

import pytest

from phil_mind_rag.ingestion.registry import DocumentRecord, DocumentRegistry


def _record(source: str = "paper.pdf", title: str = "Test Paper", **kwargs: object) -> DocumentRecord:
    defaults = dict(
        source=source,
        title=title,
        author="Test Author",
        chunk_count=10,
        chunk_size=512,
        chunk_overlap=64,
        chunker="SectionAwareChunker",
        embedding_model="text-embedding-3-small",
        ingested_at="2026-03-04T12:00:00+00:00",
    )
    defaults.update(kwargs)
    return DocumentRecord(**defaults)  # type: ignore[arg-type]


class TestDocumentRegistry:
    def test_empty_registry_returns_empty_list(self, tmp_path: Path) -> None:
        assert DocumentRegistry(tmp_path / "reg.json").list_all() == []

    def test_add_then_list_returns_record(self, tmp_path: Path) -> None:
        reg = DocumentRegistry(tmp_path / "reg.json")
        reg.add(_record())
        results = reg.list_all()
        assert len(results) == 1
        assert results[0].source == "paper.pdf"
        assert results[0].title == "Test Paper"

    def test_upsert_by_source_replaces_old_entry(self, tmp_path: Path) -> None:
        reg = DocumentRegistry(tmp_path / "reg.json")
        reg.add(_record(title="Old"))
        reg.add(_record(title="New"))
        results = reg.list_all()
        assert len(results) == 1
        assert results[0].title == "New"

    def test_different_sources_are_stored_separately(self, tmp_path: Path) -> None:
        reg = DocumentRegistry(tmp_path / "reg.json")
        reg.add(_record(source="a.pdf"))
        reg.add(_record(source="b.pdf"))
        assert len(reg.list_all()) == 2

    def test_get_returns_matching_record(self, tmp_path: Path) -> None:
        reg = DocumentRegistry(tmp_path / "reg.json")
        reg.add(_record(source="nagel.pdf", title="Nagel"))
        result = reg.get("nagel.pdf")
        assert result is not None
        assert result.title == "Nagel"

    def test_get_returns_none_for_missing_source(self, tmp_path: Path) -> None:
        reg = DocumentRegistry(tmp_path / "reg.json")
        assert reg.get("nothere.pdf") is None

    def test_data_persists_across_instances(self, tmp_path: Path) -> None:
        path = tmp_path / "reg.json"
        DocumentRegistry(path).add(_record())
        assert len(DocumentRegistry(path).list_all()) == 1

    def test_creates_parent_directory_automatically(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "dir" / "reg.json"
        reg = DocumentRegistry(path)
        reg.add(_record())
        assert path.exists()

    def test_all_fields_round_trip_correctly(self, tmp_path: Path) -> None:
        reg = DocumentRegistry(tmp_path / "reg.json")
        original = _record(chunk_count=42, chunk_size=256, chunk_overlap=32)
        reg.add(original)
        loaded = reg.list_all()[0]
        assert loaded.chunk_count == 42
        assert loaded.chunk_size == 256
        assert loaded.chunk_overlap == 32
