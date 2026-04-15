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

_DEFAULT_SYSTEM = (
    "You are a knowledgeable assistant specialising in Philosophy of Mind. "
    "Answer the user's question based ONLY on the provided context passages. "
    "If the context does not contain enough information, say so explicitly. "
    "Cite the source section when possible."
)


@dataclass
class RAGPrompt:
    """Build a grounded prompt from retrieved context + user query."""

    system: str = _DEFAULT_SYSTEM

    def build(self, query: str, contexts: list[RetrievalResult]) -> str:
        context_block = "\n\n---\n\n".join(
            f"[{c.metadata.get('source', '?')} — {c.metadata.get('section', '?')}]\n"
            f"{c.text}"
            for c in contexts
        )

        return f"{self.system}\n\n### Context\n{context_block}\n\n### Question\n{query}"
