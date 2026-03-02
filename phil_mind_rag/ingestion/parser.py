"""Document parsing — turn PDFs into structured sections.

No framework imports here; this is a pure domain layer.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Section:
    """A logical section extracted from a document."""

    title: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class ParsedDocument:
    """The result of parsing a single document."""

    source: str
    sections: list[Section]
    metadata: dict[str, str] = field(default_factory=dict)


class BaseParser(ABC):
    """Interface every document parser must implement."""

    @abstractmethod
    def parse(self, path: Path) -> ParsedDocument:
        """Parse a document and return structured sections."""


class UnstructuredPDFParser(BaseParser):
    """Parse academic PDFs using the `unstructured` library."""

    def parse(self, path: Path) -> ParsedDocument:
        from unstructured.partition.pdf import partition_pdf

        logger.info("Parsing %s with Unstructured", path.name)
        elements = partition_pdf(str(path))

        sections: list[Section] = []
        current_title = "Untitled"
        current_texts: list[str] = []

        for el in elements:
            category = el.category  # type: ignore[attr-defined]

            if category == "Title":
                # Flush previous section
                if current_texts:
                    sections.append(
                        Section(
                            title=current_title,
                            text="\n".join(current_texts),
                            metadata={"source": path.name},
                        )
                    )
                    current_texts = []
                current_title = str(el)
            else:
                current_texts.append(str(el))

        # Flush final section
        if current_texts:
            sections.append(
                Section(
                    title=current_title,
                    text="\n".join(current_texts),
                    metadata={"source": path.name},
                )
            )

        logger.info("Extracted %d sections from %s", len(sections), path.name)
        return ParsedDocument(
            source=path.name,
            sections=sections,
            metadata={"path": str(path)},
        )
