"""RAG evaluation using RAGAS.

This module wraps RAGAS so the rest of the codebase stays framework-agnostic.
Swap out by implementing a different evaluator with the same interface.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EvalSample:
    """A single evaluation sample."""

    question: str
    answer: str
    contexts: list[str]
    ground_truth: str


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


class RAGEvaluator:
    """Evaluate a RAG pipeline using RAGAS metrics."""

    def evaluate(self, samples: list[EvalSample]) -> EvalResult:
        from ragas import EvaluationDataset, evaluate as ragas_evaluate
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
