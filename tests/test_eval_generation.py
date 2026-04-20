"""Tests for corpus-pinned eval generation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from phil_mind_rag.agents.eval_generation import (
    GeneratedEvalQuestion,
    GeneratedEvalSet,
    generate_eval_set,
    to_ragas_eval_rows,
)
from phil_mind_rag.retrieval.store import RetrievalResult


def _chunks() -> list[RetrievalResult]:
    return [
        RetrievalResult(
            text="Nagel argues that consciousness has a subjective character.",
            score=0.9,
            metadata={"source": "nagel.pdf", "section": "Introduction"},
        )
    ]


def _eval_set(chunk_id: str = "chunk_0") -> GeneratedEvalSet:
    return GeneratedEvalSet(
        field="philosophy of mind",
        source="nagel.pdf",
        questions=[
            GeneratedEvalQuestion(
                question_type="factual_retrieval",
                question="What does Nagel say consciousness has?",
                reference_answer="Nagel says consciousness has a subjective character.",
                source_chunk_id=chunk_id,
                source_excerpt="consciousness has a subjective character",
                grading_note="Answer should mention subjective character.",
            ),
            GeneratedEvalQuestion(
                question_type="stance_divergence",
                question="How would positions diverge on Nagel's claim?",
                reference_answer=(
                    "Materialists, idealists, and dualists should disagree about "
                    "whether subjective character is reducible."
                ),
                source_chunk_id=chunk_id,
                source_excerpt="consciousness has a subjective character",
                expected_stances=["materialist", "idealist", "dualist"],
                grading_note="Question should elicit position-sensitive answers.",
            ),
            GeneratedEvalQuestion(
                question_type="grounding_fidelity",
                question="Which passage supports the claim about subjectivity?",
                reference_answer=(
                    "The cited chunk states that consciousness has a subjective "
                    "character."
                ),
                source_chunk_id=chunk_id,
                source_excerpt="consciousness has a subjective character",
                grading_note="Answer must trace back to the cited chunk.",
            ),
        ],
        human_review_notes=["Spot-check philosophical framing before committing."],
    )


def test_generate_eval_set_returns_valid_structured_output() -> None:
    expected = _eval_set()

    with patch(
        "phil_mind_rag.agents.eval_generation.generate_structured",
        return_value=expected,
    ) as mocked:
        result = generate_eval_set(
            field="philosophy of mind",
            source="nagel.pdf",
            chunks=_chunks(),
            client=MagicMock(),
            model="gpt-5-mini",
        )

    assert result == expected
    assert mocked.call_args.kwargs["schema_cls"] is GeneratedEvalSet
    assert "chunk_0" in mocked.call_args.kwargs["user"]
    assert "grounding_fidelity" in mocked.call_args.kwargs["user"]


def test_generate_eval_set_rejects_invalid_chunk_ids() -> None:
    with (
        patch(
            "phil_mind_rag.agents.eval_generation.generate_structured",
            return_value=_eval_set("chunk_99"),
        ),
        pytest.raises(ValueError, match="unknown chunk IDs"),
    ):
        generate_eval_set(
            field="philosophy of mind",
            source="nagel.pdf",
            chunks=_chunks(),
            client=MagicMock(),
            model="gpt-5-mini",
        )


@pytest.mark.parametrize(
    ("field", "source", "chunks", "questions_per_chunk"),
    [
        ("", "source", _chunks(), 3),
        ("field", "", _chunks(), 3),
        ("field", "source", [], 3),
        ("field", "source", _chunks(), 0),
    ],
)
def test_generate_eval_set_validates_inputs(
    field: str,
    source: str,
    chunks: list[RetrievalResult],
    questions_per_chunk: int,
) -> None:
    with pytest.raises(ValueError):
        generate_eval_set(
            field=field,
            source=source,
            chunks=chunks,
            client=MagicMock(),
            model="gpt-5-mini",
            questions_per_chunk=questions_per_chunk,
        )


def test_to_ragas_eval_rows_matches_existing_eval_shape() -> None:
    rows = to_ragas_eval_rows(_eval_set())

    assert rows[0] == {
        "question": "What does Nagel say consciousness has?",
        "ground_truth": "Nagel says consciousness has a subjective character.",
    }
