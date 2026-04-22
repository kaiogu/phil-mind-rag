"""Tests for the committed eval set."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from phil_mind_rag.eval.evaluator import load_eval_questions

EVAL_SET = Path("data/eval_set.json")
CORPUS_METADATA = Path("data/corpus_sources.json")

REQUIRED_TYPES = {
    "factual_retrieval",
    "cross_paper_comparison",
    "source_attribution",
    "stance_divergence",
    "grounding_fidelity",
    "unanswerable",
}


def _eval_rows() -> list[dict[str, Any]]:
    return json.loads(EVAL_SET.read_text(encoding="utf-8"))


def _corpus_source_ids() -> set[str]:
    metadata = json.loads(CORPUS_METADATA.read_text(encoding="utf-8"))
    return {source["id"] for source in metadata["sources"]}


def test_eval_set_loads_with_chunk_pinned_schema() -> None:
    questions = load_eval_questions(EVAL_SET)

    assert len(questions) >= 20
    assert all(question.id for question in questions)
    assert all(isinstance(question.expected_sources, list) for question in questions)
    assert all(isinstance(question.expected_chunk_ids, list) for question in questions)


def test_eval_set_covers_representative_question_types() -> None:
    question_types = {
        question.question_type for question in load_eval_questions(EVAL_SET)
    }

    assert REQUIRED_TYPES <= question_types


def test_eval_expected_sources_exist_in_curated_corpus() -> None:
    source_ids = _corpus_source_ids()

    for question in load_eval_questions(EVAL_SET):
        assert set(question.expected_sources) <= source_ids


def test_new_curated_rows_record_human_review_notes() -> None:
    curated_rows = [row for row in _eval_rows() if row["id"].startswith("corpus_")]

    assert curated_rows
    assert all(
        row.get("review_notes", "").startswith("Manually authored")
        for row in curated_rows
    )


def test_unanswerable_rows_expect_missing_evidence_acknowledgment() -> None:
    unanswerable = [
        question
        for question in load_eval_questions(EVAL_SET)
        if question.question_type == "unanswerable"
    ]

    assert len(unanswerable) >= 2
    assert all(not question.expected_sources for question in unanswerable)
    assert all(
        "should" in question.ground_truth.lower()
        and (
            "acknowledge" in question.ground_truth.lower()
            or "not include" in question.ground_truth.lower()
        )
        for question in unanswerable
    )
