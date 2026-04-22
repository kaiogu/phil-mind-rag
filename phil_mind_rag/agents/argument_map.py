"""Deterministic argument-map construction from agent outputs."""

from __future__ import annotations

from phil_mind_rag.agents.schema import (
    ArgumentMap,
    ArgumentMapClaim,
    ArgumentMapStance,
    EvidenceClaim,
    StanceMemo,
    SynthesisReport,
)


def build_argument_map(
    *,
    report: SynthesisReport,
    memos: list[StanceMemo],
) -> ArgumentMap:
    """Build a readable disagreement map without another model call."""
    claim_audit = _claim_audit(report)
    stances = [
        ArgumentMapStance(
            stance=memo.stance,
            thesis=memo.thesis,
            strongest_argument=report.strongest_arguments.get(memo.stance),
            supporting_claims=[
                _map_claim(claim, memo.stance, claim_audit)
                for claim in memo.supporting_claims
            ],
            objections=[
                _map_claim(claim, memo.stance, claim_audit)
                for claim in memo.rival_critiques
            ],
        )
        for memo in memos
    ]
    return ArgumentMap(
        question=report.question,
        disagreement_axes=report.areas_of_disagreement,
        stances=stances,
        decisive_chunks=report.decisive_chunks,
        synthesis=report.synthesis,
    )


def _claim_audit(
    report: SynthesisReport,
) -> dict[tuple[str, str], tuple[bool, str]]:
    audited = {}
    for claim in [*report.supported_claims, *report.unsupported_claims]:
        audited[(claim.stance, claim.text)] = (claim.supported, claim.note)
    return audited


def _map_claim(
    claim: EvidenceClaim,
    stance: str,
    claim_audit: dict[tuple[str, str], tuple[bool, str]],
) -> ArgumentMapClaim:
    supported, note = claim_audit.get((stance, claim.text), (None, None))
    return ArgumentMapClaim(
        text=claim.text,
        citations=claim.citations,
        supported=supported,
        note=note,
    )
