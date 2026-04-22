"""Chunking strategies — split parsed sections into retrieval-sized pieces.

No framework imports here; this is a pure domain layer.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phil_mind_rag.ingestion.parser import ParsedDocument, Section

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A retrieval-sized text fragment with provenance metadata."""

    text: str
    metadata: dict[str, str] = field(default_factory=dict)


class BaseChunker(ABC):
    """Interface every chunking strategy must implement."""

    @abstractmethod
    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        """Split a parsed document into chunks."""


class SectionAwareChunker(BaseChunker):
    """Chunk within section boundaries, never across them.

    Long sections are split with a sliding window; short sections
    are kept intact.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []

        for section in document.sections:
            section_chunks = self._split_section(section, document.source)
            chunks.extend(section_chunks)

        logger.info("Chunked '%s' into %d chunks", document.source, len(chunks))
        return chunks

    def _split_section(self, section: Section, source: str) -> list[Chunk]:
        """Split a single section using a word-level sliding window."""
        words = section.text.split()
        if not words:
            return []

        base_meta = {
            "source": source,
            "section": section.title,
            **section.metadata,
        }

        # Section fits in one chunk — keep it whole
        if len(words) <= self.chunk_size:
            return [
                Chunk(
                    text=section.text,
                    metadata=_with_chunk_identity(base_meta, source, section.title, 0),
                )
            ]

        chunks: list[Chunk] = []
        start = 0
        chunk_index = 0
        while start < len(words):
            end = start + self.chunk_size
            chunk_text = " ".join(words[start:end])
            chunks.append(
                Chunk(
                    text=chunk_text,
                    metadata=_with_chunk_identity(
                        base_meta,
                        source,
                        section.title,
                        chunk_index,
                    ),
                )
            )
            start += self.chunk_size - self.chunk_overlap
            chunk_index += 1

        return chunks


def _with_chunk_identity(
    metadata: dict[str, str],
    source: str,
    section: str,
    chunk_index: int,
) -> dict[str, str]:
    return {
        **metadata,
        "chunk_index": str(chunk_index),
        "source_chunk_id": _stable_chunk_id(source, section, chunk_index),
    }


def _stable_chunk_id(source: str, section: str, chunk_index: int) -> str:
    return f"{_slug(source)}:{_slug(section)}:chunk_{chunk_index}"


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "unknown"
