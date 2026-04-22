"""Deterministic claim extraction from structured agent outputs."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from phil_mind_rag.agents.schema import AtomicClaim

if TYPE_CHECKING:
    from phil_mind_rag.agents.schema import Claim, StanceMemo, SynthesisReport


def claims_from_stance_memo(memo: StanceMemo) -> list[AtomicClaim]:
    """Extract atomic-ish claims from a stance memo's structured fields."""
    claims: list[AtomicClaim] = []
    claims.extend(
        AtomicClaim(
            text=claim.text,
            source="stance_support",
            stance=memo.stance,
            citations=claim.citations,
        )
        for claim in memo.supporting_claims
    )
    claims.extend(
        AtomicClaim(
            text=claim.text,
            source="rival_critique",
            stance=memo.stance,
            citations=claim.citations,
        )
        for claim in memo.rival_critiques
    )
    return claims


def claims_from_stance_memos(memos: list[StanceMemo]) -> list[AtomicClaim]:
    """Extract claims from multiple stance memos."""
    return [claim for memo in memos for claim in claims_from_stance_memo(memo)]


def claims_from_synthesis_report(report: SynthesisReport) -> list[AtomicClaim]:
    """Extract audited claims from a grounding synthesis report."""
    return [
        _from_report_claim(claim, source="synthesis_supported")
        for claim in report.supported_claims
    ] + [
        _from_report_claim(claim, source="synthesis_unsupported")
        for claim in report.unsupported_claims
    ]


def claims_from_answer(
    answer: str,
    *,
    citations: list[str] | None = None,
) -> list[AtomicClaim]:
    """Split a baseline answer into sentence-level claim candidates."""
    return [
        AtomicClaim(
            text=sentence,
            source="baseline_answer",
            citations=citations or [],
        )
        for sentence in _sentences(answer)
    ]


def extract_analysis_claims(
    *,
    memos: list[StanceMemo],
    report: SynthesisReport,
    baseline_answer: str | None = None,
) -> list[AtomicClaim]:
    """Extract claims from the complete multi-agent analysis output."""
    claims = [
        *claims_from_stance_memos(memos),
        *claims_from_synthesis_report(report),
    ]
    if baseline_answer:
        claims.extend(claims_from_answer(baseline_answer))
    return _dedupe_claims(claims)


def _from_report_claim(claim: Claim, *, source: str) -> AtomicClaim:
    return AtomicClaim(
        text=claim.text,
        source=source,
        stance=claim.stance,
        supported=claim.supported,
        citations=claim.citations,
        source_chunk_id=claim.source_chunk_id,
    )


def _sentences(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text.strip())
        if sentence.strip()
    ]


def _dedupe_claims(claims: list[AtomicClaim]) -> list[AtomicClaim]:
    seen: set[tuple[str, str, str | None, tuple[str, ...]]] = set()
    deduped: list[AtomicClaim] = []
    for claim in claims:
        key = (claim.text, claim.source, claim.stance, tuple(claim.citations))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(claim)
    return deduped
