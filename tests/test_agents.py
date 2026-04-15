"""Unit tests for the multi-agent system.

All OpenAI and ChromaDB calls are mocked — no network access required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from phil_mind_rag.agents.graph import AnalysisResult, run_analysis
from phil_mind_rag.agents.grounding import run_grounding
from phil_mind_rag.agents.prompts import grounding_prompt, stance_prompt
from phil_mind_rag.agents.schema import Claim, StanceMemo, SynthesisReport
from phil_mind_rag.agents.stance import run_dualist, run_idealist, run_materialist
from phil_mind_rag.agents.state import AgentState
from phil_mind_rag.retrieval.store import RetrievalResult

# --- Fixtures -----------------------------------------------------------


@pytest.fixture
def sample_chunks() -> list[RetrievalResult]:
    return [
        RetrievalResult(
            text="The brain produces consciousness through neural correlates.",
            score=0.9,
            metadata={"source": "nagel.pdf", "section": "Introduction"},
        ),
        RetrievalResult(
            text="Consciousness cannot be reduced to physical processes.",
            score=0.85,
            metadata={"source": "chalmers.pdf", "section": "Hard Problem"},
        ),
    ]


@pytest.fixture
def stub_memo() -> StanceMemo:
    return StanceMemo(
        stance="materialist",
        thesis="Consciousness is a physical process.",
        supporting_arguments=["Neural correlates of consciousness exist."],
        attack_on_rivals=["Idealism lacks empirical support."],
        confidence=0.8,
        uncertainty_notes="The hard problem remains.",
        citations=["chunk_0"],
    )


@pytest.fixture
def stub_report() -> SynthesisReport:
    return SynthesisReport(
        question="What is consciousness?",
        areas_of_disagreement=["The hard problem", "Qualia"],
        strongest_arguments={
            "materialist": "Neural correlates are well-documented.",
            "idealist": "Qualia cannot be reduced.",
            "dualist": "Both physical and mental properties exist.",
        },
        unsupported_claims=[
            Claim(
                text="Consciousness is purely physical.",
                stance="materialist",
                supported=False,
                source_chunk_id=None,
                note="No chunk supports this without qualification.",
            )
        ],
        synthesis="The hard problem remains genuinely open.",
        source_chunks_used=["chunk_0", "chunk_1"],
    )


# --- Schema tests -------------------------------------------------------


class TestSchema:
    def test_stance_memo_round_trip(self, stub_memo: StanceMemo) -> None:
        json_str = stub_memo.model_dump_json()
        restored = StanceMemo.model_validate_json(json_str)
        assert restored == stub_memo

    def test_claim_round_trip(self) -> None:
        claim = Claim(
            text="X causes Y",
            stance="materialist",
            supported=True,
            source_chunk_id="chunk_0",
            note="Directly cited.",
        )
        assert Claim.model_validate_json(claim.model_dump_json()) == claim

    def test_synthesis_report_round_trip(self, stub_report: SynthesisReport) -> None:
        restored = SynthesisReport.model_validate_json(stub_report.model_dump_json())
        assert restored == stub_report

    def test_claim_allows_none_source(self) -> None:
        claim = Claim(
            text="Unsupported claim", stance="idealist",
            supported=False, source_chunk_id=None, note="No evidence.",
        )
        assert claim.source_chunk_id is None


# --- Prompt tests -------------------------------------------------------


class TestPrompts:
    def test_stance_prompt_contains_question(
        self, sample_chunks: list[RetrievalResult]
    ) -> None:
        system, user = stance_prompt(
            "materialist", "Foreground physicalism.", "What is qualia?", sample_chunks
        )
        assert "What is qualia?" in user
        assert "materialist" in system.lower()

    def test_stance_prompt_contains_chunk_ids(
        self, sample_chunks: list[RetrievalResult]
    ) -> None:
        _, user = stance_prompt("idealist", "Foreground idealism.", "Q?", sample_chunks)
        assert "chunk_0" in user
        assert "chunk_1" in user

    def test_stance_prompt_empty_chunks(self) -> None:
        system, user = stance_prompt("dualist", "Focus on dualism.", "Q?", [])
        assert isinstance(system, str)
        assert isinstance(user, str)

    def test_grounding_prompt_contains_memos(
        self,
        sample_chunks: list[RetrievalResult],
        stub_memo: StanceMemo,
    ) -> None:
        _, user = grounding_prompt("Q?", sample_chunks, [stub_memo])
        assert "MATERIALIST" in user
        assert stub_memo.thesis in user

    def test_grounding_prompt_contains_chunks(
        self,
        sample_chunks: list[RetrievalResult],
        stub_memo: StanceMemo,
    ) -> None:
        _, user = grounding_prompt("Q?", sample_chunks, [stub_memo])
        assert "chunk_0" in user
        assert "chunk_1" in user


# --- Stance node tests --------------------------------------------------


class TestStanceNodes:
    def _state(self, sample_chunks: list[RetrievalResult]) -> AgentState:
        return AgentState(
            question="What is consciousness?",
            chunks=sample_chunks,
            materialist_memo=None,
            idealist_memo=None,
            dualist_memo=None,
            report=None,
        )

    def test_run_materialist_returns_memo(
        self, sample_chunks: list[RetrievalResult], stub_memo: StanceMemo
    ) -> None:
        client = MagicMock()
        stub_memo_mat = stub_memo.model_copy(update={"stance": "materialist"})
        _patch = patch(
            "phil_mind_rag.agents.stance.generate_structured",
            return_value=stub_memo_mat,
        )
        with _patch:
            result = run_materialist(self._state(sample_chunks), client, "gpt-4o-mini")
        assert "materialist_memo" in result
        assert result["materialist_memo"].stance == "materialist"

    def test_run_idealist_returns_memo(
        self, sample_chunks: list[RetrievalResult], stub_memo: StanceMemo
    ) -> None:
        client = MagicMock()
        stub_ide = stub_memo.model_copy(update={"stance": "idealist"})
        _patch = patch(
            "phil_mind_rag.agents.stance.generate_structured", return_value=stub_ide
        )
        with _patch:
            result = run_idealist(self._state(sample_chunks), client, "gpt-4o-mini")
        assert result["idealist_memo"].stance == "idealist"

    def test_run_dualist_returns_memo(
        self, sample_chunks: list[RetrievalResult], stub_memo: StanceMemo
    ) -> None:
        client = MagicMock()
        stub_dua = stub_memo.model_copy(update={"stance": "dualist"})
        _patch = patch(
            "phil_mind_rag.agents.stance.generate_structured", return_value=stub_dua
        )
        with _patch:
            result = run_dualist(self._state(sample_chunks), client, "gpt-4o-mini")
        assert result["dualist_memo"].stance == "dualist"


# --- Grounding node tests -----------------------------------------------


class TestGroundingNode:
    def test_run_grounding_returns_report(
        self,
        sample_chunks: list[RetrievalResult],
        stub_memo: StanceMemo,
        stub_report: SynthesisReport,
    ) -> None:
        state = AgentState(
            question="What is consciousness?",
            chunks=sample_chunks,
            materialist_memo=stub_memo.model_copy(update={"stance": "materialist"}),
            idealist_memo=stub_memo.model_copy(update={"stance": "idealist"}),
            dualist_memo=stub_memo.model_copy(update={"stance": "dualist"}),
            report=None,
        )
        client = MagicMock()
        _patch = patch(
            "phil_mind_rag.agents.grounding.generate_structured",
            return_value=stub_report,
        )
        with _patch:
            result = run_grounding(state, client, "gpt-4o-mini")
        assert "report" in result
        assert isinstance(result["report"], SynthesisReport)

    def test_run_grounding_skips_none_memos(
        self,
        sample_chunks: list[RetrievalResult],
        stub_memo: StanceMemo,
        stub_report: SynthesisReport,
    ) -> None:
        """Grounding should handle partial memo set gracefully."""
        state = AgentState(
            question="Q?",
            chunks=sample_chunks,
            materialist_memo=stub_memo,
            idealist_memo=None,
            dualist_memo=None,
            report=None,
        )
        client = MagicMock()
        _patch = patch(
            "phil_mind_rag.agents.grounding.generate_structured",
            return_value=stub_report,
        )
        with _patch:
            result = run_grounding(state, client, "gpt-4o-mini")
        assert result["report"] is not None


# --- End-to-end graph test ---------------------------------------------


class TestRunAnalysis:
    def test_returns_analysis_result(
        self,
        sample_chunks: list[RetrievalResult],
        stub_memo: StanceMemo,
        stub_report: SynthesisReport,
    ) -> None:
        mock_pipeline = MagicMock()
        mock_pipeline.retrieve.return_value = sample_chunks

        mock_settings = MagicMock()
        mock_settings.openai_api_key.get_secret_value.return_value = "sk-test"
        mock_settings.openai_chat_model = "gpt-4o-mini"

        mat = stub_memo.model_copy(update={"stance": "materialist"})
        ide = stub_memo.model_copy(update={"stance": "idealist"})
        dua = stub_memo.model_copy(update={"stance": "dualist"})

        def fake_generate_structured(client, model, system, user, schema_cls):
            if schema_cls is StanceMemo:
                if "materialist" in system:
                    return mat
                if "idealist" in system:
                    return ide
                return dua
            return stub_report

        stance_patch = patch(
            "phil_mind_rag.agents.stance.generate_structured",
            side_effect=fake_generate_structured,
        )
        ground_patch = patch(
            "phil_mind_rag.agents.grounding.generate_structured",
            return_value=stub_report,
        )
        with patch("phil_mind_rag.agents.graph.OpenAI"), stance_patch, ground_patch:
            result = run_analysis(
                "What is consciousness?", mock_pipeline, mock_settings
            )

        assert isinstance(result, AnalysisResult)
        assert isinstance(result.report, SynthesisReport)
        assert isinstance(result.materialist_memo, StanceMemo)
        assert isinstance(result.idealist_memo, StanceMemo)
        assert isinstance(result.dualist_memo, StanceMemo)
        assert len(result.chunks) == 2

    def test_retrieval_called_with_question(
        self,
        sample_chunks: list[RetrievalResult],
        stub_memo: StanceMemo,
        stub_report: SynthesisReport,
    ) -> None:
        mock_pipeline = MagicMock()
        mock_pipeline.retrieve.return_value = sample_chunks

        mock_settings = MagicMock()
        mock_settings.openai_api_key.get_secret_value.return_value = "sk-test"
        mock_settings.openai_chat_model = "gpt-4o-mini"

        mat = stub_memo.model_copy(update={"stance": "materialist"})
        ide = stub_memo.model_copy(update={"stance": "idealist"})
        dua = stub_memo.model_copy(update={"stance": "dualist"})

        stance_patch = patch(
            "phil_mind_rag.agents.stance.generate_structured",
            side_effect=[mat, ide, dua],
        )
        ground_patch = patch(
            "phil_mind_rag.agents.grounding.generate_structured",
            return_value=stub_report,
        )
        with patch("phil_mind_rag.agents.graph.OpenAI"), stance_patch, ground_patch:
            run_analysis("Hard problem of consciousness", mock_pipeline, mock_settings)

        mock_pipeline.retrieve.assert_called_once_with("Hard problem of consciousness")


# --- Import / startup regression tests -----------------------------------


class TestStartup:
    """Guard against the circular import that broke startup in the first run.

    The cycle was:
        retrieval/store.py → ingestion/chunker.py
                           → ingestion/metadata_extractor.py
                           → generation/__init__.py
                           → generation/prompts.py
                           → retrieval/store.py  ← cycle

    Fix: generation/prompts.py imports RetrievalResult under TYPE_CHECKING only,
    since from __future__ import annotations makes all annotations lazy strings.

    If the fix is reverted, build_graph() raises:
        NameError: name 'RetrievalResult' is not defined
    because LangGraph calls get_type_hints(AgentState) at StateGraph construction
    time and can't resolve the lazy annotation string.
    """

    def test_build_graph_constructs_without_error(self) -> None:
        """StateGraph(AgentState) must not raise NameError on RetrievalResult."""
        mock_pipeline = MagicMock()
        mock_settings = MagicMock()
        mock_settings.openai_api_key.get_secret_value.return_value = "sk-test"
        mock_settings.openai_chat_model = "gpt-4o-mini"

        with patch("phil_mind_rag.agents.graph.OpenAI"):
            # build_graph() calls StateGraph(AgentState) which calls
            # get_type_hints(AgentState) — this is the call that failed.
            from phil_mind_rag.agents.graph import build_graph
            graph = build_graph(mock_pipeline, mock_settings)

        assert graph is not None
