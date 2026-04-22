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


@dataclass(frozen=True)
class StanceEvidenceSets:
    """Global grounding chunks plus stance-specific prompt subsets."""

    chunks: list[RetrievalResult]
    by_stance: dict[StanceName, list[RetrievalResult]]


def build_stance_query(question: str, stance: StanceName) -> str:
    """Expand a user question with stance-specific retrieval terms."""
    terms = ", ".join(STANCE_QUERY_TERMS[stance])
    return f"{question}\n\nStance retrieval focus: {terms}"


def build_stance_evidence_sets(
    *,
    base_chunks: list[RetrievalResult],
    stance_chunks: dict[StanceName, list[RetrievalResult]],
) -> StanceEvidenceSets:
    """Assign one global chunk-ID space across all retrieval calls."""
    all_chunks: list[RetrievalResult] = []
    global_ids: dict[tuple[str, ...], str] = {}

    def add(chunk: RetrievalResult) -> RetrievalResult:
        key = _dedupe_key(chunk)
        if key not in global_ids:
            global_ids[key] = f"chunk_{len(all_chunks)}"
            all_chunks.append(_with_prompt_chunk_id(chunk, global_ids[key]))
        return _with_prompt_chunk_id(chunk, global_ids[key])

    for chunk in base_chunks:
        add(chunk)

    def stance_subset(chunks: list[RetrievalResult]) -> list[RetrievalResult]:
        seen_ids: set[str] = set()
        subset: list[RetrievalResult] = []
        for chunk in [*base_chunks, *chunks]:
            mapped = add(chunk)
            chunk_id = mapped.metadata["chunk_id"]
            if chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)
            subset.append(mapped)
        return subset

    by_stance = {
        stance: stance_subset(chunks) for stance, chunks in stance_chunks.items()
    }
    return StanceEvidenceSets(chunks=all_chunks, by_stance=by_stance)


def build_evidence_pack(
    stance: StanceName,
    chunks: list[RetrievalResult],
) -> EvidencePack:
    """Return stance-reranked chunks while preserving original chunk IDs."""
    terms = STANCE_QUERY_TERMS[stance]
    indexed_chunks = [
        (_chunk_id(index, chunk), chunk, _term_hits(chunk.text, terms))
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


def _chunk_id(index: int, chunk: RetrievalResult) -> str:
    return chunk.metadata.get("chunk_id", f"chunk_{index}")


def _dedupe_key(chunk: RetrievalResult) -> tuple[str, ...]:
    for key in ("source_chunk_id", "id"):
        if value := chunk.metadata.get(key):
            return (key, value)
    if (
        (source := chunk.metadata.get("source"))
        and (section := chunk.metadata.get("section"))
        and (chunk_index := chunk.metadata.get("chunk_index"))
    ):
        return ("derived", source, section, chunk_index)
    return ("text", chunk.text)
