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
        from datasets import Dataset
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        data = {
            "question": [s.question for s in samples],
            "answer": [s.answer for s in samples],
            "contexts": [s.contexts for s in samples],
            "ground_truth": [s.ground_truth for s in samples],
        }

        dataset = Dataset.from_dict(data)

        logger.info("Running RAGAS evaluation on %d samples", len(samples))
        result = ragas_evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
        )

        return EvalResult(
            faithfulness=result["faithfulness"],
            answer_relevancy=result["answer_relevancy"],
            context_precision=result["context_precision"],
            context_recall=result["context_recall"],
        )
