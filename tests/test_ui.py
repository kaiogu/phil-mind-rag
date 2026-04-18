"""Tests for Gradio-facing formatting and analysis wiring."""

from __future__ import annotations

from unittest.mock import patch

from phil_mind_rag.agents.graph import AnalysisResult
from phil_mind_rag.agents.schema import (
    Claim,
    EvidenceClaim,
    StanceMemo,
    SynthesisReport,
)
from phil_mind_rag.app.ui import handle_analysis
from phil_mind_rag.retrieval.store import RetrievalResult


def _memo(stance: str) -> StanceMemo:
    return StanceMemo(
        stance=stance,
        thesis=f"{stance} thesis",
        supporting_claims=[
            EvidenceClaim(text=f"{stance} support", citations=["chunk_0"])
        ],
        rival_critiques=[
            EvidenceClaim(text=f"{stance} critique", citations=["chunk_1"])
        ],
        confidence=0.7,
        uncertainty_notes="Some uncertainty remains.",
    )


def _report() -> SynthesisReport:
    return SynthesisReport(
        question="What is consciousness?",
        areas_of_disagreement=["reduction", "qualia"],
        strongest_arguments={"materialist": "Neural evidence is strong."},
        supported_claims=[
            Claim(
                text="Materialism has empirical support.",
                stance="materialist",
                supported=True,
                citations=["chunk_0"],
                source_chunk_id="chunk_0",
                note="Directly grounded.",
            )
        ],
        unsupported_claims=[
            Claim(
                text="Idealism explains everything.",
                stance="idealist",
                supported=False,
                citations=["chunk_9"],
                source_chunk_id=None,
                note="Unknown chunk citation.",
            )
        ],
        synthesis="The dispute remains open.",
        decisive_chunks=["chunk_0"],
        source_chunks_used=["chunk_0", "chunk_1"],
    )


def test_handle_analysis_returns_baseline_and_audit_sections() -> None:
    result = AnalysisResult(
        baseline_answer="Baseline answer",
        report=_report(),
        chunks=[
            RetrievalResult(
                text="ctx 0",
                score=0.9,
                metadata={"source": "a", "section": "s1"},
            ),
            RetrievalResult(
                text="ctx 1",
                score=0.8,
                metadata={"source": "b", "section": "s2"},
            ),
        ],
        materialist_memo=_memo("materialist"),
        idealist_memo=_memo("idealist"),
        dualist_memo=_memo("dualist"),
    )

    with (
        patch("phil_mind_rag.app.ui._get_pipeline"),
        patch("phil_mind_rag.app.ui._get_settings"),
        patch("phil_mind_rag.app.ui.run_analysis", return_value=result),
    ):
        baseline, materialist, idealist, dualist, grounding, synthesis, sources = (
            handle_analysis("What is consciousness?")
        )

    assert baseline == "Baseline answer"
    assert "materialist support" in materialist
    assert "idealist support" in idealist
    assert "dualist support" in dualist
    assert "Supported Claims" in grounding
    assert "Flagged Claims" in grounding
    assert "Decisive Evidence" in synthesis
    assert "chunk_0" in sources
