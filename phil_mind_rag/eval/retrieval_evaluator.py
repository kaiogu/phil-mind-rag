"""Deterministic retrieval metrics for chunk-pinned eval questions."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from phil_mind_rag.eval.evaluator import EvalQuestion
    from phil_mind_rag.retrieval.store import RetrievalResult


@dataclass(frozen=True)
class RetrievalQuestionResult:
    """Retrieval precision/recall for one eval question."""

    eval_id: str
    question: str
    expected_sources: list[str]
    retrieved_sources: list[str]
    expected_chunk_ids: list[str]
    retrieved_chunk_ids: list[str]
    source_precision_at_k: float | None
    source_recall_at_k: float | None
    chunk_precision_at_k: float | None
    chunk_recall_at_k: float | None
    notes: list[str]


@dataclass(frozen=True)
class RetrievalEvalResult:
    """Aggregated retrieval eval result with per-question diagnostics."""

    question_results: list[RetrievalQuestionResult]

    @property
    def source_precision_at_k(self) -> float | None:
        return _mean_optional(
            result.source_precision_at_k for result in self.question_results
        )

    @property
    def source_recall_at_k(self) -> float | None:
        return _mean_optional(
            result.source_recall_at_k for result in self.question_results
        )

    @property
    def chunk_precision_at_k(self) -> float | None:
        return _mean_optional(
            result.chunk_precision_at_k for result in self.question_results
        )

    @property
    def chunk_recall_at_k(self) -> float | None:
        return _mean_optional(
            result.chunk_recall_at_k for result in self.question_results
        )

    def summary(self) -> str:
        return (
            f"Source Precision@k: {_format_score(self.source_precision_at_k)}\n"
            f"Source Recall@k:    {_format_score(self.source_recall_at_k)}\n"
            f"Chunk Precision@k:  {_format_score(self.chunk_precision_at_k)}\n"
            f"Chunk Recall@k:     {_format_score(self.chunk_recall_at_k)}"
        )


def evaluate_retrieval(
    retrieved_by_eval_id: dict[str, list[RetrievalResult]],
    questions: list[EvalQuestion],
) -> RetrievalEvalResult:
    """Evaluate retrieval outputs against expected sources and chunk IDs."""
    return RetrievalEvalResult(
        question_results=[
            evaluate_retrieval_question(
                question=question,
                retrieved=retrieved_by_eval_id.get(question.id, []),
            )
            for question in questions
        ]
    )


def evaluate_retrieval_question(
    *,
    question: EvalQuestion,
    retrieved: list[RetrievalResult],
) -> RetrievalQuestionResult:
    """Evaluate retrieval for one question."""
    retrieved_sources = _dedupe(
        result.metadata.get("source", "").strip()
        for result in retrieved
        if result.metadata.get("source", "").strip()
    )
    retrieved_chunk_ids = _dedupe(
        chunk_id for result in retrieved if (chunk_id := _chunk_id(result)) is not None
    )

    notes: list[str] = []
    source_precision, source_recall = _precision_recall(
        expected=question.expected_sources,
        retrieved=retrieved_sources,
        label="source",
        notes=notes,
    )
    chunk_precision, chunk_recall = _precision_recall(
        expected=question.expected_chunk_ids,
        retrieved=retrieved_chunk_ids,
        label="chunk ID",
        notes=notes,
    )

    return RetrievalQuestionResult(
        eval_id=question.id,
        question=question.question,
        expected_sources=question.expected_sources,
        retrieved_sources=retrieved_sources,
        expected_chunk_ids=question.expected_chunk_ids,
        retrieved_chunk_ids=retrieved_chunk_ids,
        source_precision_at_k=source_precision,
        source_recall_at_k=source_recall,
        chunk_precision_at_k=chunk_precision,
        chunk_recall_at_k=chunk_recall,
        notes=notes or ["Retrieval expectations were evaluated."],
    )


def _chunk_id(result: RetrievalResult) -> str | None:
    for key in ("chunk_id", "source_chunk_id", "id"):
        value = result.metadata.get(key)
        if value:
            return value

    source = result.metadata.get("source")
    section = result.metadata.get("section")
    chunk_index = result.metadata.get("chunk_index")
    if source and section and chunk_index:
        return f"{source}:{section}:chunk_{chunk_index}"
    return None


def _precision_recall(
    *,
    expected: list[str],
    retrieved: list[str],
    label: str,
    notes: list[str],
) -> tuple[float | None, float | None]:
    if not expected:
        notes.append(f"No expected {label}s configured; skipped {label} metrics.")
        return None, None
    if not retrieved:
        notes.append(f"No retrieved {label}s found.")
        return 0.0, 0.0

    expected_set = set(expected)
    retrieved_set = set(retrieved)
    hits = expected_set & retrieved_set
    precision = len(hits) / len(retrieved_set)
    recall = len(hits) / len(expected_set)

    missed = sorted(expected_set - retrieved_set)
    if missed:
        notes.append(f"Missed expected {label}s: {', '.join(missed)}.")
    return precision, recall


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _mean_optional(values: Iterable[float | None]) -> float | None:
    concrete = [value for value in values if isinstance(value, float)]
    return mean(concrete) if concrete else None


def _format_score(score: float | None) -> str:
    return "n/a" if score is None else f"{score:.3f}"
