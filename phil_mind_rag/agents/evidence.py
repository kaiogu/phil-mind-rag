"""Stance-specific evidence packing for agent prompts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from phil_mind_rag.retrieval.store import RetrievalResult

if TYPE_CHECKING:
    from collections.abc import Iterable

StanceName = Literal["materialist", "idealist", "dualist"]

STANCE_QUERY_TERMS: dict[StanceName, tuple[str, ...]] = {
    "materialist": (
        "physical",
        "physicalism",
        "brain",
        "neural",
        "neuroscience",
        "functional",
        "reduction",
        "behavior",
    ),
    "idealist": (
        "consciousness",
        "subjective",
        "experience",
        "phenomenal",
        "first-person",
        "irreducible",
        "reducible",
        "reduced",
        "qualia",
        "mind",
    ),
    "dualist": (
        "dualism",
        "property",
        "substance",
        "mental",
        "physical",
        "explanatory gap",
        "irreducible",
        "mind-body",
    ),
}


@dataclass(frozen=True)
class EvidencePack:
    """A stance-specific ordering of retrieved evidence."""

    stance: StanceName
    query_terms: tuple[str, ...]
    chunks: list[RetrievalResult]
    chunk_ids: list[str]


def build_evidence_pack(
    stance: StanceName,
    chunks: list[RetrievalResult],
) -> EvidencePack:
    """Return stance-reranked chunks while preserving original chunk IDs."""
    terms = STANCE_QUERY_TERMS[stance]
    indexed_chunks = [
        (f"chunk_{index}", chunk, _term_hits(chunk.text, terms))
        for index, chunk in enumerate(chunks)
    ]
    ranked = sorted(indexed_chunks, key=lambda item: (-item[2], item[0]))
    evidence_chunks = [
        _with_prompt_chunk_id(chunk, chunk_id) for chunk_id, chunk, _ in ranked
    ]
    return EvidencePack(
        stance=stance,
        query_terms=terms,
        chunks=evidence_chunks,
        chunk_ids=[chunk_id for chunk_id, _, _ in ranked],
    )


def _term_hits(text: str, terms: Iterable[str]) -> int:
    normalized = text.lower()
    return sum(1 for term in terms if term in normalized)


def _with_prompt_chunk_id(
    chunk: RetrievalResult,
    chunk_id: str,
) -> RetrievalResult:
    metadata = {**chunk.metadata, "chunk_id": chunk_id}
    return RetrievalResult(text=chunk.text, score=chunk.score, metadata=metadata)
