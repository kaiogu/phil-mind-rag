"""RAG evaluation using RAGAS.

This module wraps RAGAS so the rest of the codebase stays framework-agnostic.
Swap out by implementing a different evaluator with the same interface.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

logger = logging.getLogger(__name__)

EvalQuestionType = Literal[
    "factual_retrieval",
    "cross_paper_comparison",
    "source_attribution",
    "stance_divergence",
    "grounding_fidelity",
    "unanswerable",
]

EvalDifficulty = Literal["easy", "medium", "hard"]


@dataclass
class EvalSample:
    """A single evaluation sample."""

    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    question_type: EvalQuestionType | None = None


@dataclass(frozen=True)
class EvalQuestion:
    """A chunk-pinned eval question with RAGAS compatibility."""

    id: str
    question: str
    ground_truth: str
    question_type: EvalQuestionType
    difficulty: EvalDifficulty
    expected_sources: list[str]
    expected_chunk_ids: list[str]

    @classmethod
    def from_mapping(cls, row: dict[str, Any], index: int) -> EvalQuestion:
        """Parse both the new eval schema and legacy RAGAS smoke rows."""
        question = _required_str(row, "question")
        ground_truth = _answer_text(row)
        return cls(
            id=_optional_str(row, "id") or f"eval_{index:03d}",
            question=question,
            ground_truth=ground_truth,
            question_type=_question_type(row),
            difficulty=_difficulty(row),
            expected_sources=_str_list(row, "expected_sources"),
            expected_chunk_ids=_str_list(row, "expected_chunk_ids"),
        )

    def to_ragas_row(self) -> dict[str, str]:
        """Return the legacy row shape expected by the RAGAS wrapper."""
        return {"question": self.question, "ground_truth": self.ground_truth}

    def to_sample(self, *, answer: str, contexts: list[str]) -> EvalSample:
        """Build an EvalSample after the pipeline has generated an answer."""
        return EvalSample(
            question=self.question,
            answer=answer,
            contexts=contexts,
            ground_truth=self.ground_truth,
            question_type=self.question_type,
        )


@dataclass
class EvalResult:
    """Aggregated evaluation scores."""

    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float

    def summary(self) -> str:
        return (
            f"Faithfulness:       {self.faithfulness:.3f}\n"
            f"Answer Relevancy:   {self.answer_relevancy:.3f}\n"
            f"Context Precision:  {self.context_precision:.3f}\n"
            f"Context Recall:     {self.context_recall:.3f}"
        )


def load_eval_questions(path: str | Path) -> list[EvalQuestion]:
    """Load eval questions from JSON, accepting legacy or chunk-pinned rows."""
    eval_path = Path(path)
    rows = json.loads(eval_path.read_text())
    if not isinstance(rows, list):
        raise ValueError("eval set must be a JSON list")
    questions: list[EvalQuestion] = []
    for index, row in enumerate(rows, start=1):
        questions.append(EvalQuestion.from_mapping(_ensure_mapping(row, index), index))
    return questions


def to_ragas_eval_rows(questions: list[EvalQuestion]) -> list[dict[str, str]]:
    """Convert chunk-pinned eval questions to the existing RAGAS row shape."""
    return [question.to_ragas_row() for question in questions]


class RAGEvaluator:
    """Evaluate a RAG pipeline using RAGAS metrics."""

    def evaluate(self, samples: list[EvalSample]) -> EvalResult:
        from ragas import EvaluationDataset
        from ragas import evaluate as ragas_evaluate
        from ragas.dataset_schema import SingleTurnSample
        from ragas.metrics.collections import (
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )

        ragas_samples = [
            SingleTurnSample(
                user_input=s.question,
                response=s.answer,
                retrieved_contexts=s.contexts,
                reference=s.ground_truth,
            )
            for s in samples
        ]

        dataset = EvaluationDataset(samples=ragas_samples)  # type: ignore

        metrics = [
            Faithfulness(),  # type: ignore
            AnswerRelevancy(),  # type: ignore
            ContextPrecision(),  # type: ignore
            ContextRecall(),  # type: ignore
        ]

        logger.info("Running RAGAS evaluation on %d samples", len(samples))
        result = ragas_evaluate(dataset=dataset, metrics=metrics)  # type: ignore

        def _mean(key: str) -> float:
            values = [v for v in result[key] if v is not None]  # type: ignore
            return sum(values) / len(values) if values else float("nan")

        return EvalResult(
            faithfulness=_mean("faithfulness"),
            answer_relevancy=_mean("answer_relevancy"),
            context_precision=_mean("context_precision"),
            context_recall=_mean("context_recall"),
        )


def _ensure_mapping(row: object, index: int) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise ValueError(f"eval row {index} must be an object")
    return cast("dict[str, Any]", row)


def _required_str(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"eval row missing non-empty string field: {key}")
    return value.strip()


def _optional_str(row: dict[str, Any], key: str) -> str | None:
    value = row.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"eval field must be a string when present: {key}")
    value = value.strip()
    return value or None


def _answer_text(row: dict[str, Any]) -> str:
    for key in ("ground_truth", "answer", "reference_answer"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(
        "eval row must include one of: ground_truth, answer, reference_answer"
    )


def _str_list(row: dict[str, Any], key: str) -> list[str]:
    value = row.get(key, [])
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"eval field must be a list of strings: {key}")
    return [item.strip() for item in value if item.strip()]


def _question_type(row: dict[str, Any]) -> EvalQuestionType:
    value = row.get("type", row.get("question_type", "factual_retrieval"))
    allowed = set(EvalQuestionType.__args__)
    if value not in allowed:
        raise ValueError(f"unknown eval question type: {value}")
    return value


def _difficulty(row: dict[str, Any]) -> EvalDifficulty:
    value = row.get("difficulty", "medium")
    allowed = set(EvalDifficulty.__args__)
    if value not in allowed:
        raise ValueError(f"unknown eval difficulty: {value}")
    return value
