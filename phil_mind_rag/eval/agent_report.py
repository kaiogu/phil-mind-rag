"""Offline multi-agent eval report builder."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from phil_mind_rag.agents.schema import (
    Claim,
    EvidenceClaim,
    StanceMemo,
    SynthesisReport,
)
from phil_mind_rag.eval.agent_evaluator import (
    SemanticSupportJudge,
    evaluate_multi_agent_output,
)
from phil_mind_rag.eval.reporting import timestamped_eval_dir, write_json_report
from phil_mind_rag.retrieval.store import RetrievalResult

if TYPE_CHECKING:
    from pathlib import Path


def build_smoke_agent_eval_report(
    semantic_judge: SemanticSupportJudge | None = None,
) -> dict[str, Any]:
    """Build a deterministic multi-agent eval report with no paid calls."""
    chunks = [
        RetrievalResult(
            text=(
                "Nagel argues that conscious experience has a subjective "
                "character that objective physical descriptions do not capture."
            ),
            score=0.95,
            metadata={"source": "nagel_bat", "section": "Subjective character"},
        )
    ]
    memos = [
        _memo(
            "materialist",
            support=(
                "Physicalists can accept Nagel's gap while seeking a future reduction."
            ),
            critique="Idealism may overstate what follows from subjective character.",
        ),
        _memo(
            "idealist",
            support=(
                "Nagel's subjectivity argument pressures purely objective accounts."
            ),
            critique=(
                "Materialism has not explained why physical facts yield experience."
            ),
        ),
        _memo(
            "dualist",
            support=(
                "Nagel's gap supports treating phenomenal character as irreducible."
            ),
            critique=(
                "Both materialism and idealism need more evidence to settle ontology."
            ),
        ),
    ]
    report = SynthesisReport(
        question="What does Nagel's bat argument show about consciousness?",
        areas_of_disagreement=[
            "Whether subjective character can be reduced to objective facts.",
            (
                "Whether the explanatory gap supports dualism or only exposes "
                "a theory gap."
            ),
        ],
        strongest_arguments={
            "materialist": (
                "Materialism can treat the gap as a current explanatory limit."
            ),
            "idealist": (
                "Idealism foregrounds the irreducibility of first-person experience."
            ),
            "dualist": (
                "Dualism captures the contrast between objective and subjective facts."
            ),
        },
        supported_claims=[
            _claim("materialist", "Physicalists can accept Nagel's gap.", True),
            _claim("idealist", "Nagel pressures purely objective accounts.", True),
            _claim("dualist", "Nagel supports an irreducibility reading.", True),
        ],
        unsupported_claims=[
            _claim(
                "materialist",
                "Materialism has already solved consciousness.",
                False,
            ),
            _claim("idealist", "Nagel conclusively proves idealism.", False),
            _claim("dualist", "Nagel conclusively proves substance dualism.", False),
        ],
        synthesis=(
            "The fixture checks whether the evaluator rewards cited, complete "
            "stance memos and a synthesis that adjudicates each stance."
        ),
        decisive_chunks=["chunk_0"],
        source_chunks_used=["chunk_0"],
    )
    result = evaluate_multi_agent_output(
        memos=memos,
        report=report,
        chunk_count=len(chunks),
        chunks=chunks,
        semantic_judge=semantic_judge,
    )
    return {
        "config": {
            "mode": "offline_smoke",
            "semantic_support_judge": (
                "llm" if semantic_judge is not None else "not_configured"
            ),
            "expected_stances": ["materialist", "idealist", "dualist"],
        },
        "fixture": {
            "question": report.question,
            "chunk_count": len(chunks),
            "chunk_sources": [chunk.metadata.get("source", "") for chunk in chunks],
            "memo_stances": [memo.stance for memo in memos],
        },
        "result": result,
    }


def write_smoke_agent_eval_report(
    base_dir: Path,
    semantic_judge: SemanticSupportJudge | None = None,
) -> Path:
    """Write the offline agent eval report and return the JSON path."""
    run_dir = timestamped_eval_dir(base_dir)
    report_path = run_dir / "agent_eval.json"
    write_json_report(report_path, build_smoke_agent_eval_report(semantic_judge))
    return report_path


def _memo(stance: str, *, support: str, critique: str) -> StanceMemo:
    return StanceMemo(
        stance=stance,
        thesis=f"{stance} thesis over the retrieved Nagel passage.",
        supporting_claims=[EvidenceClaim(text=support, citations=["chunk_0"])],
        rival_critiques=[EvidenceClaim(text=critique, citations=["chunk_0"])],
        confidence=0.75,
        uncertainty_notes="This is an offline smoke fixture, not a live model run.",
    )


def _claim(stance: str, text: str, supported: bool) -> Claim:
    return Claim(
        text=text,
        stance=stance,
        supported=supported,
        citations=["chunk_0"],
        source_chunk_id="chunk_0" if supported else None,
        note="Fixture claim audited against chunk_0.",
    )
