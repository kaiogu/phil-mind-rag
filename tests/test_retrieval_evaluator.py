"""Tests for deterministic retrieval eval metrics."""

from dataclasses import replace

from phil_mind_rag.eval.evaluator import EvalQuestion
from phil_mind_rag.eval.retrieval_evaluator import (
    evaluate_retrieval,
    evaluate_retrieval_question,
)
from phil_mind_rag.retrieval.store import RetrievalResult


def _question(
    *,
    expected_sources: list[str] | None = None,
    expected_chunk_ids: list[str] | None = None,
) -> EvalQuestion:
    return EvalQuestion(
        id="q1",
        question="What does Nagel argue?",
        ground_truth="Nagel argues consciousness has subjective character.",
        question_type="factual_retrieval",
        difficulty="easy",
        expected_sources=["nagel_bat"]
        if expected_sources is None
        else expected_sources,
        expected_chunk_ids=["nagel_bat:intro:chunk_0"]
        if expected_chunk_ids is None
        else expected_chunk_ids,
    )


def _result(source: str, chunk_id: str | None = None) -> RetrievalResult:
    metadata = {"source": source}
    if chunk_id is not None:
        metadata["chunk_id"] = chunk_id
    return RetrievalResult(text="passage", score=0.9, metadata=metadata)


def test_evaluate_retrieval_question_scores_source_and_chunk_matches() -> None:
    result = evaluate_retrieval_question(
        question=_question(),
        retrieved=[
            _result("nagel_bat", "nagel_bat:intro:chunk_0"),
            _result("chalmers_hard_problem", "chalmers:intro:chunk_0"),
        ],
    )

    assert result.source_precision_at_k == 0.5
    assert result.source_recall_at_k == 1.0
    assert result.chunk_precision_at_k == 0.5
    assert result.chunk_recall_at_k == 1.0
    assert result.retrieved_sources == ["nagel_bat", "chalmers_hard_problem"]


def test_evaluate_retrieval_question_reports_misses() -> None:
    result = evaluate_retrieval_question(
        question=_question(),
        retrieved=[_result("chalmers_hard_problem", "chalmers:intro:chunk_0")],
    )

    assert result.source_precision_at_k == 0.0
    assert result.source_recall_at_k == 0.0
    assert result.chunk_precision_at_k == 0.0
    assert result.chunk_recall_at_k == 0.0
    assert any("Missed expected sources" in note for note in result.notes)
    assert any("Missed expected chunk IDs" in note for note in result.notes)


def test_evaluate_retrieval_question_skips_unconfigured_expectations() -> None:
    result = evaluate_retrieval_question(
        question=_question(expected_sources=[], expected_chunk_ids=[]),
        retrieved=[_result("nagel_bat")],
    )

    assert result.source_precision_at_k is None
    assert result.source_recall_at_k is None
    assert result.chunk_precision_at_k is None
    assert result.chunk_recall_at_k is None
    assert any("No expected sources configured" in note for note in result.notes)
    assert any("No expected chunk IDs configured" in note for note in result.notes)


def test_evaluate_retrieval_aggregates_configured_metrics() -> None:
    q1 = _question()
    q2 = _question(
        expected_sources=["chalmers_hard_problem"],
        expected_chunk_ids=["chalmers:intro:chunk_0"],
    )
    q2 = replace(q2, id="q2")

    result = evaluate_retrieval(
        questions=[q1, q2],
        retrieved_by_eval_id={
            "q1": [_result("nagel_bat", "nagel_bat:intro:chunk_0")],
            "q2": [_result("nagel_bat", "nagel_bat:intro:chunk_0")],
        },
    )

    assert result.source_precision_at_k == 0.5
    assert result.source_recall_at_k == 0.5
    assert result.chunk_precision_at_k == 0.5
    assert result.chunk_recall_at_k == 0.5
    assert "Source Precision@k: 0.500" in result.summary()


def test_chunk_id_can_be_derived_from_source_section_and_index() -> None:
    result = evaluate_retrieval_question(
        question=_question(expected_chunk_ids=["nagel_bat:intro:chunk_0"]),
        retrieved=[
            RetrievalResult(
                text="passage",
                score=0.9,
                metadata={
                    "source": "nagel_bat",
                    "section": "intro",
                    "chunk_index": "0",
                },
            )
        ],
    )

    assert result.chunk_precision_at_k == 1.0
    assert result.chunk_recall_at_k == 1.0


def test_stable_source_chunk_id_is_used_for_eval_matching() -> None:
    result = evaluate_retrieval_question(
        question=_question(expected_chunk_ids=["nagel_bat:intro:chunk_0"]),
        retrieved=[
            RetrievalResult(
                text="passage",
                score=0.9,
                metadata={
                    "source": "nagel_bat",
                    "source_chunk_id": "nagel_bat:intro:chunk_0",
                },
            )
        ],
    )

    assert result.chunk_precision_at_k == 1.0
    assert result.chunk_recall_at_k == 1.0
