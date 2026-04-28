"""Prompt templates — kept as plain dataclasses, no framework dependency."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

# TYPE_CHECKING guard breaks a circular import:
# retrieval/store.py → ingestion/chunker.py → generation/__init__.py
#                    → generation/prompts.py → retrieval/store.py  ← cycle
# Since from __future__ import annotations makes all annotations lazy strings,
# RetrievalResult is never needed at runtime here — only for static analysis.
if TYPE_CHECKING:
    from phil_mind_rag.retrieval.store import RetrievalResult

_DEFAULT_SYSTEM = """# Identity
You are a knowledgeable assistant specialising in Philosophy of Mind.

# Task
Answer the user's question using only the provided context passages.

# Rules
- Do not use outside knowledge.
- If the context is insufficient, say so explicitly instead of guessing.
- Prefer a concise answer that distinguishes well-supported claims from open
  uncertainty.
- When making a supported claim, cite the relevant source as `(source — section)`.
"""


@dataclass
class RAGPrompt:
    """Build a grounded prompt from retrieved context + user query."""

    system: str = _DEFAULT_SYSTEM

    def build(self, query: str, contexts: list[RetrievalResult]) -> str:
        context_block = "\n\n---\n\n".join(
            f'<source id="{c.metadata.get("source", "?")}" '
            f'section="{c.metadata.get("section", "?")}">\n'
            f"{c.text}"
            "\n</source>"
            for c in contexts
        )
        return (
            f"{self.system}\n\n"
            "<context>\n"
            f"{context_block}\n"
            "</context>\n\n"
            "<question>\n"
            f"{query}\n"
            "</question>"
        )
