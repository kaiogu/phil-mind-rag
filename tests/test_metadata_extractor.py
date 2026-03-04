"""Tests for MetadataExtractor."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from phil_mind_rag.ingestion.metadata_extractor import ExtractedMetadata, MetadataExtractor


def _mock_pdf_reader(title: str | None = None, author: str | None = None) -> MagicMock:
    info = MagicMock()
    info.title = title
    info.author = author
    reader = MagicMock()
    reader.metadata = info
    reader.pages = [MagicMock()]
    reader.pages[0].extract_text.return_value = "First page text about consciousness."
    return reader


class TestMetadataExtractor:
    def test_returns_pdf_metadata_when_both_fields_present(self, tmp_path: Path) -> None:
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF")
        reader = _mock_pdf_reader(title="Being and Time", author="Heidegger")

        with patch("pypdf.PdfReader", return_value=reader):
            result = MetadataExtractor().extract(pdf)

        assert result.title == "Being and Time"
        assert result.author == "Heidegger"
        assert result.method == "pdf_metadata"

    def test_falls_back_to_llm_when_pdf_metadata_missing(self, tmp_path: Path) -> None:
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF")
        reader = _mock_pdf_reader(title=None, author=None)

        llm = MagicMock()
        llm.generate.return_value = '{"title": "LLM Title", "author": "LLM Author"}'

        with patch("pypdf.PdfReader", return_value=reader):
            result = MetadataExtractor(llm=llm).extract(pdf)

        assert result.title == "LLM Title"
        assert result.author == "LLM Author"
        assert result.method == "llm"

    def test_partial_pdf_metadata_supplemented_by_llm(self, tmp_path: Path) -> None:
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF")
        reader = _mock_pdf_reader(title="Known Title", author=None)

        llm = MagicMock()
        llm.generate.return_value = '{"title": null, "author": "LLM Author"}'

        with patch("pypdf.PdfReader", return_value=reader):
            result = MetadataExtractor(llm=llm).extract(pdf)

        assert result.title == "Known Title"  # kept from PDF metadata
        assert result.author == "LLM Author"  # filled in by LLM

    def test_no_llm_returns_partial_metadata(self, tmp_path: Path) -> None:
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF")
        reader = _mock_pdf_reader(title="Some Title", author=None)

        with patch("pypdf.PdfReader", return_value=reader):
            result = MetadataExtractor(llm=None).extract(pdf)

        assert result.title == "Some Title"
        assert result.author is None

    def test_llm_failure_is_swallowed_and_returns_pdf_metadata(self, tmp_path: Path) -> None:
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF")
        reader = _mock_pdf_reader(title=None, author=None)

        llm = MagicMock()
        llm.generate.side_effect = RuntimeError("LLM exploded")

        with patch("pypdf.PdfReader", return_value=reader):
            result = MetadataExtractor(llm=llm).extract(pdf)  # should not raise

        assert result.method == "none"

    def test_llm_strips_markdown_code_fences(self, tmp_path: Path) -> None:
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF")
        reader = _mock_pdf_reader(title=None, author=None)

        llm = MagicMock()
        llm.generate.return_value = '```json\n{"title": "Clean", "author": "Tidy"}\n```'

        with patch("pypdf.PdfReader", return_value=reader):
            result = MetadataExtractor(llm=llm).extract(pdf)

        assert result.title == "Clean"
        assert result.author == "Tidy"

    def test_whitespace_in_metadata_is_stripped(self, tmp_path: Path) -> None:
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF")
        reader = _mock_pdf_reader(title="  Spaced Title  ", author="  Author  ")

        with patch("pypdf.PdfReader", return_value=reader):
            result = MetadataExtractor().extract(pdf)

        assert result.title == "Spaced Title"
        assert result.author == "Author"
