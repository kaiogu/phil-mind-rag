from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from phil_mind_rag.generation.llm import BaseLLM

logger = logging.getLogger(__name__)


@dataclass
class ExtractedMetadata:
    title: str | None
    author: str | None
    method: str  # "pdf_metadata" | "llm" | "none"


class MetadataExtractor:
    """Extract title/author from a PDF without touching the Gradio layer.

    Extraction order:
    1. PDF embedded metadata (pypdf — free, instant)
    2. LLM over first-page text (if llm is provided and fields still missing)
    """

    def __init__(self, llm: BaseLLM | None = None) -> None:
        self._llm = llm

    def extract(self, pdf_path: Path) -> ExtractedMetadata:
        meta = self._try_pdf_metadata(pdf_path)

        if meta.title and meta.author:
            return meta  # both fields found cheaply

        if self._llm and (not meta.title or not meta.author):
            try:
                llm_meta = self._try_llm(
                    pdf_path,
                    need_title=not meta.title,
                    need_author=not meta.author,
                )
                return ExtractedMetadata(
                    title=meta.title or llm_meta.title,
                    author=meta.author or llm_meta.author,
                    method=(
                        "llm" if (llm_meta.title or llm_meta.author) else meta.method
                    ),
                )
            except Exception:
                logger.warning(
                    "LLM metadata extraction failed for %s",
                    pdf_path.name,
                    exc_info=True,
                )

        return meta

    # ------------------------------------------------------------------

    def _try_pdf_metadata(self, pdf_path: Path) -> ExtractedMetadata:
        try:
            from pypdf import PdfReader

            info = PdfReader(str(pdf_path)).metadata
            if not info:
                return ExtractedMetadata(title=None, author=None, method="none")
            title = (info.title or "").strip() or None
            author = (info.author or "").strip() or None
            method = "pdf_metadata" if (title or author) else "none"
            return ExtractedMetadata(title=title, author=author, method=method)
        except Exception:
            logger.warning(
                "pypdf metadata read failed for %s", pdf_path.name, exc_info=True
            )
            return ExtractedMetadata(title=None, author=None, method="none")

    def _try_llm(
        self,
        pdf_path: Path,
        need_title: bool,
        need_author: bool,
    ) -> ExtractedMetadata:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        # Grab text from first page only — enough to identify title/author
        first_page_text = reader.pages[0].extract_text() or ""
        if not first_page_text.strip():
            return ExtractedMetadata(title=None, author=None, method="none")

        wants = []
        if need_title:
            wants.append("title")
        if need_author:
            wants.append("author")

        assert self._llm is not None  # narrowing: _try_llm is only called when self._llm is truthy
        prompt = (
            "You are a metadata extractor for academic philosophy papers.\n"
            f"Extract the following fields from the text: {', '.join(wants)}.\n"
            "Respond with ONLY a JSON object — no prose, no code fences:\n"
            '{"title": "...", "author": "..."}\n'
            "Use null for any field you cannot determine with confidence.\n\n"
            f"TEXT (first page):\n{first_page_text[:2500]}"
        )

        raw = self._llm.generate(prompt)
        # Strip markdown code fences if present
        raw = (
            raw.strip()
            .removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        data = json.loads(raw)
        title = (data.get("title") or "").strip() or None
        author = (data.get("author") or "").strip() or None
        return ExtractedMetadata(title=title, author=author, method="llm")
