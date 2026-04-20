"""Tests for deterministic multi-agent eval scaffolding."""

from __future__ import annotations

import pytest

from phil_mind_rag.agents.schema import (
    Claim,
    EvidenceClaim,
    StanceMemo,
    SynthesisReport,
)
from phil_mind_rag.eval.agent_evaluator import evaluate_multi_agent_output


def _memo(stance: str, chunk_id: str = "chunk_0") -> StanceMemo:
    return StanceMemo(
        stance=stance,
        thesis=f"{stance} thesis.",
        supporting_claims=[
            EvidenceClaim(
                text=f"{stance} support",
                citations=[chunk_id],
            )
        ],
        rival_critiques=[
            EvidenceClaim(
                text=f"{stance} critique",
                citations=[chunk_id],
            )
        ],
        confidence=0.8,
        uncertainty_notes="Some uncertainty.",
    )


def _claim(
    text: str,
    stance: str,
    *,
    supported: bool,
    chunk_id: str | None = "chunk_0",
) -> Claim:
    return Claim(
        text=text,
        stance=stance,
        supported=supported,
        citations=[chunk_id] if chunk_id else [],
        source_chunk_id=chunk_id if supported else None,
        note="Audited against retrieved evidence.",
    )


def _report() -> SynthesisReport:
    return SynthesisReport(
        question="What is consciousness?",
        areas_of_disagreement=["Reduction", "Explanatory gap"],
        strongest_arguments={
            "materialist": "Materialist support",
            "idealist": "Idealist support",
            "dualist": "Dualist support",
        },
        supported_claims=[
            _claim("materialist support", "materialist", supported=True),
            _claim("idealist support", "idealist", supported=True),
            _claim("dualist support", "dualist", supported=True),
        ],
        unsupported_claims=[
            _claim("materialist critique", "materialist", supported=False),
            _claim("idealist critique", "idealist", supported=False),
            _claim("dualist critique", "dualist", supported=False),
        ],
        synthesis="The dispute remains live.",
        decisive_chunks=["chunk_0"],
        source_chunks_used=["chunk_0"],
    )


def test_evaluate_multi_agent_output_scores_complete_report_high() -> None:
    result = evaluate_multi_agent_output(
        memos=[_memo("materialist"), _memo("idealist"), _memo("dualist")],
        report=_report(),
        chunk_count=1,
    )

    assert result.overall == pytest.approx(1.0)
    assert "Overall" in result.summary()


def test_evaluate_multi_agent_output_penalizes_missing_stance_and_bad_citation() -> (
    None
):
    report = _report().model_copy(
        update={
            "strongest_arguments": {"materialist": "Materialist support"},
            "source_chunks_used": ["chunk_99"],
        }
    )

    result = evaluate_multi_agent_output(
        memos=[_memo("materialist", "chunk_99")],
        report=report,
        chunk_count=1,
    )

    assert result.grounding_fidelity.score < 1.0
    assert result.position_fidelity.score < 1.0
    assert result.disagreement_quality.score < 1.0
    assert result.overall < 1.0


def test_evaluate_multi_agent_output_rejects_negative_chunk_count() -> None:
    with pytest.raises(ValueError, match="chunk_count"):
        evaluate_multi_agent_output(
            memos=[],
            report=_report(),
            chunk_count=-1,
        )
