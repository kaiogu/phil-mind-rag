"""Claim verification and conservative citation repair."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal, Protocol

from phil_mind_rag.agents.schema import AtomicClaim, VerifiedClaim

if TYPE_CHECKING:
    from phil_mind_rag.retrieval.store import RetrievalResult

ClaimSupportLabel = Literal["supported", "unsupported", "ambiguous"]


class ClaimSupportJudge(Protocol):
    """Callable contract for optional semantic support checks."""

    def __call__(
        self,
        *,
        claim_text: str,
        cited_texts: list[str],
    ) -> ClaimSupportLabel:
        """Return whether cited texts semantically support the claim."""


def verify_claims(
    claims: list[AtomicClaim],
    chunks: list[RetrievalResult],
    *,
    semantic_judge: ClaimSupportJudge | None = None,
) -> list[VerifiedClaim]:
    """Verify a list of claims against available chunks."""
    chunk_text_by_id = _chunk_text_by_id(chunks)
    return [
        verify_claim(
            claim,
            chunk_text_by_id=chunk_text_by_id,
            semantic_judge=semantic_judge,
        )
        for claim in claims
    ]


def verify_claim(
    claim: AtomicClaim,
    *,
    chunk_text_by_id: dict[str, str],
    semantic_judge: ClaimSupportJudge | None = None,
) -> VerifiedClaim:
    """Verify one claim structurally and optionally semantically."""
    valid_citations = [
        citation for citation in claim.citations if citation in chunk_text_by_id
    ]
    invalid_citations = [
        citation for citation in claim.citations if citation not in chunk_text_by_id
    ]

    if invalid_citations:
        repaired = _repair_citations(claim.text, chunk_text_by_id)
        if repaired:
            return _verified(
                claim,
                label="ambiguous",
                citations=repaired,
                repaired_citations=repaired,
                note=(
                    "Invalid citations were replaced with conservative lexical "
                    "repair candidates; semantic support was not established."
                ),
            )
        return _verified(
            claim,
            label="unsupported",
            citations=[],
            note=f"Claim cites unknown chunk IDs: {', '.join(invalid_citations)}.",
        )

    if not valid_citations:
        repaired = _repair_citations(claim.text, chunk_text_by_id)
        if repaired:
            return _verified(
                claim,
                label="ambiguous",
                citations=repaired,
                repaired_citations=repaired,
                note=(
                    "Claim had no citations; conservative lexical repair found "
                    "candidate chunks but semantic support was not established."
                ),
            )
        return _verified(
            claim,
            label="unsupported",
            citations=[],
            note="Claim has no citations and no repair candidate was found.",
        )

    if semantic_judge is None:
        return _verified(
            claim,
            label="ambiguous",
            citations=valid_citations,
            note=(
                "Citations are structurally valid, but no semantic judge was "
                "configured."
            ),
        )

    label = semantic_judge(
        claim_text=claim.text,
        cited_texts=[chunk_text_by_id[citation] for citation in valid_citations],
    )
    return _verified(
        claim,
        label=label,
        citations=valid_citations,
        note=f"Semantic judge labeled this claim as {label}.",
    )


def _chunk_text_by_id(chunks: list[RetrievalResult]) -> dict[str, str]:
    chunk_texts: dict[str, str] = {}
    for index, chunk in enumerate(chunks):
        ids = [f"chunk_{index}"]
        ids.extend(
            value
            for key in ("chunk_id", "source_chunk_id", "id")
            if (value := chunk.metadata.get(key))
        )
        if (
            (source := chunk.metadata.get("source"))
            and (section := chunk.metadata.get("section"))
            and (chunk_index := chunk.metadata.get("chunk_index"))
        ):
            ids.append(f"{source}:{section}:chunk_{chunk_index}")
        for chunk_id in ids:
            chunk_texts[chunk_id] = chunk.text
    return chunk_texts


def _repair_citations(
    claim_text: str,
    chunk_text_by_id: dict[str, str],
) -> list[str]:
    claim_terms = _terms(claim_text)
    if len(claim_terms) < 2:
        return []

    scored: list[tuple[int, str]] = []
    for chunk_id, chunk_text in chunk_text_by_id.items():
        overlap = len(claim_terms & _terms(chunk_text))
        if overlap >= 2:
            scored.append((overlap, chunk_id))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [chunk_id for _, chunk_id in scored[:2]]


def _terms(text: str) -> set[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "be",
        "but",
        "by",
        "for",
        "in",
        "is",
        "it",
        "of",
        "or",
        "that",
        "the",
        "to",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2 and token not in stopwords
    }


def _verified(
    claim: AtomicClaim,
    *,
    label: ClaimSupportLabel,
    citations: list[str],
    note: str,
    repaired_citations: list[str] | None = None,
) -> VerifiedClaim:
    return VerifiedClaim(
        claim=claim,
        label=label,
        citations=citations,
        repaired_citations=repaired_citations or [],
        note=note,
    )
