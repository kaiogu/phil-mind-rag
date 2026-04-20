"""Deterministic eval scaffolding for the multi-agent philosophy pipeline.

These checks do not decide philosophical truth. They measure whether the agent
outputs satisfy the KGU-93 contracts: cited claims, stance coverage, concrete
disagreements, and an adjudicator that actually audits memo claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phil_mind_rag.agents.schema import (
        EvidenceClaim,
        StanceMemo,
        SynthesisReport,
    )

DEFAULT_STANCES = ("materialist", "idealist", "dualist")


@dataclass(frozen=True)
class DimensionScore:
    """A normalized eval score with short diagnostic notes."""

    score: float
    notes: list[str]


@dataclass(frozen=True)
class MultiAgentEvalResult:
    """Deterministic scores for KGU-93 multi-agent output quality."""

    grounding_fidelity: DimensionScore
    position_fidelity: DimensionScore
    disagreement_quality: DimensionScore
    adjudication_quality: DimensionScore

    @property
    def overall(self) -> float:
        scores = [
            self.grounding_fidelity.score,
            self.position_fidelity.score,
            self.disagreement_quality.score,
            self.adjudication_quality.score,
        ]
        return sum(scores) / len(scores)

    def summary(self) -> str:
        return (
            f"Grounding fidelity:   {self.grounding_fidelity.score:.3f}\n"
            f"Position fidelity:    {self.position_fidelity.score:.3f}\n"
            f"Disagreement quality: {self.disagreement_quality.score:.3f}\n"
            f"Adjudication quality: {self.adjudication_quality.score:.3f}\n"
            f"Overall:              {self.overall:.3f}"
        )


def evaluate_multi_agent_output(
    *,
    memos: list[StanceMemo],
    report: SynthesisReport,
    chunk_count: int,
    expected_stances: tuple[str, ...] = DEFAULT_STANCES,
) -> MultiAgentEvalResult:
    """Score one multi-agent run against deterministic KGU-93 contracts."""
    if chunk_count < 0:
        raise ValueError("chunk_count must not be negative")
    valid_chunk_ids = {f"chunk_{i}" for i in range(chunk_count)}
    return MultiAgentEvalResult(
        grounding_fidelity=_score_grounding_fidelity(
            memos=memos,
            report=report,
            valid_chunk_ids=valid_chunk_ids,
        ),
        position_fidelity=_score_position_fidelity(
            memos=memos,
            expected_stances=expected_stances,
            valid_chunk_ids=valid_chunk_ids,
        ),
        disagreement_quality=_score_disagreement_quality(
            report=report,
            expected_stances=expected_stances,
        ),
        adjudication_quality=_score_adjudication_quality(
            memos=memos,
            report=report,
        ),
    )


def _score_grounding_fidelity(
    *,
    memos: list[StanceMemo],
    report: SynthesisReport,
    valid_chunk_ids: set[str],
) -> DimensionScore:
    notes: list[str] = []
    memo_claims = _memo_claims(memos)
    claims_with_valid_citations = sum(
        1
        for claim in memo_claims
        if claim.citations
        and all(citation in valid_chunk_ids for citation in claim.citations)
    )
    memo_claim_score = _ratio(claims_with_valid_citations, len(memo_claims))
    if memo_claim_score < 1.0:
        notes.append("Some stance memo claims have missing or invalid chunk citations.")

    report_claims = [*report.supported_claims, *report.unsupported_claims]
    report_claims_valid = sum(
        1
        for claim in report_claims
        if all(citation in valid_chunk_ids for citation in claim.citations)
        and (claim.source_chunk_id is None or claim.source_chunk_id in valid_chunk_ids)
    )
    report_claim_score = _ratio(report_claims_valid, len(report_claims))
    if report_claim_score < 1.0:
        notes.append("Some grounding report claims cite unknown chunk IDs.")

    used_chunks = [*report.source_chunks_used, *report.decisive_chunks]
    valid_used_chunks = sum(
        1 for chunk_id in used_chunks if chunk_id in valid_chunk_ids
    )
    chunk_reference_score = _ratio(valid_used_chunks, len(used_chunks))
    if chunk_reference_score < 1.0:
        notes.append("Report references chunks outside the retrieval result set.")

    return DimensionScore(
        score=_mean([memo_claim_score, report_claim_score, chunk_reference_score]),
        notes=notes or ["All cited chunk IDs are structurally valid."],
    )


def _score_position_fidelity(
    *,
    memos: list[StanceMemo],
    expected_stances: tuple[str, ...],
    valid_chunk_ids: set[str],
) -> DimensionScore:
    notes: list[str] = []
    memos_by_stance = {memo.stance: memo for memo in memos}
    present = sum(1 for stance in expected_stances if stance in memos_by_stance)
    coverage_score = _ratio(present, len(expected_stances))
    missing = [stance for stance in expected_stances if stance not in memos_by_stance]
    if missing:
        notes.append(f"Missing stance memos: {', '.join(missing)}.")

    complete = 0
    cited = 0
    for memo in memos_by_stance.values():
        claims = _claims_for_memo(memo)
        if memo.thesis.strip() and memo.supporting_claims and memo.rival_critiques:
            complete += 1
        if claims and all(
            claim.citations
            and all(citation in valid_chunk_ids for citation in claim.citations)
            for claim in claims
        ):
            cited += 1

    completeness_score = _ratio(complete, len(expected_stances))
    citation_score = _ratio(cited, len(expected_stances))
    if completeness_score < 1.0:
        notes.append("Some stance memos lack thesis, support, or rival critique.")
    if citation_score < 1.0:
        notes.append("Some stance memos are not fully citation-grounded.")

    return DimensionScore(
        score=_mean([coverage_score, completeness_score, citation_score]),
        notes=notes
        or ["All expected stance memos are present and structurally complete."],
    )


def _score_disagreement_quality(
    *,
    report: SynthesisReport,
    expected_stances: tuple[str, ...],
) -> DimensionScore:
    notes: list[str] = []
    areas_score = min(
        len([a for a in report.areas_of_disagreement if a.strip()]) / 2, 1.0
    )
    if areas_score < 1.0:
        notes.append("Report should surface at least two concrete disagreement areas.")

    strongest_stances = {
        stance
        for stance, argument in report.strongest_arguments.items()
        if argument.strip()
    }
    strongest_score = _ratio(
        sum(1 for stance in expected_stances if stance in strongest_stances),
        len(expected_stances),
    )
    if strongest_score < 1.0:
        notes.append("Strongest-argument map does not cover every expected stance.")

    synthesis_score = 1.0 if report.synthesis.strip() else 0.0
    if synthesis_score == 0.0:
        notes.append("Synthesis is empty.")

    return DimensionScore(
        score=_mean([areas_score, strongest_score, synthesis_score]),
        notes=notes or ["Report surfaces disagreements and stance-specific arguments."],
    )


def _score_adjudication_quality(
    *,
    memos: list[StanceMemo],
    report: SynthesisReport,
) -> DimensionScore:
    notes: list[str] = []
    memo_claim_texts = {claim.text for claim in _memo_claims(memos)}
    adjudicated_texts = {
        claim.text for claim in [*report.supported_claims, *report.unsupported_claims]
    }
    coverage_score = _ratio(
        len(memo_claim_texts & adjudicated_texts),
        len(memo_claim_texts),
    )
    if coverage_score < 1.0:
        notes.append("Grounding report does not adjudicate every stance memo claim.")

    has_supported = bool(report.supported_claims)
    has_unsupported = bool(report.unsupported_claims)
    both_buckets_score = 1.0 if has_supported and has_unsupported else 0.5
    if not has_supported:
        notes.append("Grounding report has no supported claims.")
    if not has_unsupported:
        notes.append("Grounding report has no unsupported or flagged claims.")

    annotated_claims = sum(
        1
        for claim in [*report.supported_claims, *report.unsupported_claims]
        if claim.note.strip()
    )
    annotation_score = _ratio(
        annotated_claims,
        len([*report.supported_claims, *report.unsupported_claims]),
    )
    if annotation_score < 1.0:
        notes.append("Some adjudicated claims lack grounding notes.")

    return DimensionScore(
        score=_mean([coverage_score, both_buckets_score, annotation_score]),
        notes=notes or ["Grounding report adjudicates memo claims with annotations."],
    )


def _memo_claims(memos: list[StanceMemo]) -> list[EvidenceClaim]:
    return [claim for memo in memos for claim in _claims_for_memo(memo)]


def _claims_for_memo(memo: StanceMemo) -> list[EvidenceClaim]:
    return [*memo.supporting_claims, *memo.rival_critiques]


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 1.0
    return numerator / denominator


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)
