"""Tests for deterministic multi-agent eval scaffolding."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from phil_mind_rag.agents.schema import (
    Claim,
    EvidenceClaim,
    StanceMemo,
    SynthesisReport,
)
from phil_mind_rag.eval.agent_evaluator import (
    LLMSemanticJudge,
    SemanticSupportLabel,
    _parse_support_label,
    evaluate_multi_agent_output,
)
from phil_mind_rag.retrieval.store import RetrievalResult


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
    assert "Semantic support" in result.summary()


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


def test_evaluate_multi_agent_output_scores_semantic_support_separately() -> None:
    def judge(*, claim_text: str, cited_texts: list[str]) -> SemanticSupportLabel:
        assert cited_texts == ["The brain produces conscious states."]
        if "unsupported" in claim_text:
            return "unsupported"
        if "ambiguous" in claim_text:
            return "ambiguous"
        return "supported"

    memo = StanceMemo(
        stance="materialist",
        thesis="Consciousness is physically grounded.",
        supporting_claims=[
            EvidenceClaim(text="supported memo claim", citations=["chunk_0"]),
            EvidenceClaim(text="ambiguous memo claim", citations=["chunk_0"]),
        ],
        rival_critiques=[
            EvidenceClaim(text="unsupported memo claim", citations=["chunk_0"])
        ],
        confidence=0.7,
        uncertainty_notes="The evidence is limited.",
    )
    report = _report().model_copy(
        update={
            "supported_claims": [
                _claim("supported report claim", "materialist", supported=True)
            ],
            "unsupported_claims": [
                _claim("unsupported report claim", "materialist", supported=False)
            ],
        }
    )

    result = evaluate_multi_agent_output(
        memos=[memo],
        report=report,
        chunk_count=1,
        expected_stances=("materialist",),
        chunks=[
            RetrievalResult(
                text="The brain produces conscious states.",
                score=0.9,
                metadata={"source": "paper.pdf"},
            )
        ],
        semantic_judge=judge,
    )

    assert result.grounding_fidelity.score == pytest.approx(1.0)
    assert result.semantic_support.score == pytest.approx(0.5)
    assert {finding.label for finding in result.semantic_findings} == {
        "supported",
        "ambiguous",
        "unsupported",
    }


def test_evaluate_multi_agent_output_skips_semantic_support_without_judge() -> None:
    result = evaluate_multi_agent_output(
        memos=[_memo("materialist"), _memo("idealist"), _memo("dualist")],
        report=_report(),
        chunk_count=1,
    )

    assert result.semantic_support.score == pytest.approx(1.0)
    assert "skipped" in result.semantic_support.notes[0]
    assert result.semantic_findings == []


class TestParseSupportLabel:
    def test_parses_supported(self) -> None:
        assert _parse_support_label("supported") == "supported"

    def test_parses_unsupported(self) -> None:
        assert _parse_support_label("unsupported") == "unsupported"

    def test_parses_ambiguous(self) -> None:
        assert _parse_support_label("ambiguous") == "ambiguous"

    def test_case_insensitive(self) -> None:
        assert _parse_support_label("SUPPORTED") == "supported"

    def test_extracts_from_sentence(self) -> None:
        result = _parse_support_label("The claim is supported by the text.")
        assert result == "supported"

    def test_defaults_to_ambiguous_on_garbage(self) -> None:
        assert _parse_support_label("I don't know, maybe?") == "ambiguous"

    def test_empty_string_defaults_to_ambiguous(self) -> None:
        assert _parse_support_label("") == "ambiguous"


def _stub_openai_client(response_text: str | None) -> MagicMock:
    """Build a minimal OpenAI client stub that returns *response_text*."""
    message = SimpleNamespace(content=response_text)
    choice = SimpleNamespace(message=message)
    completion = SimpleNamespace(choices=[choice])
    client = MagicMock()
    client.chat.completions.create.return_value = completion
    return client


class TestLLMSemanticJudge:
    def test_returns_supported_label(self) -> None:
        client = _stub_openai_client("supported")
        judge = LLMSemanticJudge(client=client, model="gpt-4o-mini")
        label = judge(
            claim_text="Brain produces consciousness.", cited_texts=["Evidence."]
        )
        assert label == "supported"

    def test_returns_unsupported_label(self) -> None:
        client = _stub_openai_client("unsupported")
        judge = LLMSemanticJudge(client=client, model="gpt-4o-mini")
        label = judge(
            claim_text="Qualia are reducible.", cited_texts=["Counter-evidence."]
        )
        assert label == "unsupported"

    def test_defaults_ambiguous_on_unparseable_response(self) -> None:
        client = _stub_openai_client("I cannot determine this.")
        judge = LLMSemanticJudge(client=client, model="gpt-4o-mini")
        label = judge(claim_text="Claim.", cited_texts=["Text."])
        assert label == "ambiguous"

    def test_passes_model_to_client(self) -> None:
        client = _stub_openai_client("supported")
        judge = LLMSemanticJudge(client=client, model="claude-3-5-sonnet")
        judge(claim_text="Claim.", cited_texts=["Text."])
        call_kwargs = client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "claude-3-5-sonnet"

    def test_truncates_long_evidence_to_500_chars(self) -> None:
        long_text = "x" * 1000
        client = _stub_openai_client("supported")
        judge = LLMSemanticJudge(client=client, model="gpt-4o-mini")
        judge(claim_text="Claim.", cited_texts=[long_text])
        call_args = client.chat.completions.create.call_args
        prompt = call_args.kwargs["messages"][0]["content"]
        assert "x" * 501 not in prompt

    def test_handles_none_response_content(self) -> None:
        client = _stub_openai_client(None)
        judge = LLMSemanticJudge(client=client, model="gpt-4o-mini")
        label = judge(claim_text="Claim.", cited_texts=["Text."])
        assert label == "ambiguous"
