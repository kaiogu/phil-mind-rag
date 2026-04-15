"""Tests for EvalSample, EvalResult, and RAGEvaluator."""

import pytest

from phil_mind_rag.eval.evaluator import EvalResult, EvalSample, RAGEvaluator


class TestEvalResult:
    def test_summary_contains_all_metric_names(self) -> None:
        result = EvalResult(
            faithfulness=0.8,
            answer_relevancy=0.7,
            context_precision=0.9,
            context_recall=0.6,
        )
        summary = result.summary()
        assert "Faithfulness" in summary
        assert "Answer Relevancy" in summary
        assert "Context Precision" in summary
        assert "Context Recall" in summary

    def test_summary_formats_scores_to_three_decimal_places(self) -> None:
        result = EvalResult(
            faithfulness=0.85,
            answer_relevancy=0.725,
            context_precision=0.9,
            context_recall=0.6,
        )
        summary = result.summary()
        assert "0.850" in summary
        assert "0.725" in summary

    def test_summary_is_multiline_string(self) -> None:
        result = EvalResult(0.8, 0.7, 0.9, 0.6)
        assert "\n" in result.summary()


class TestEvalSample:
    def test_can_be_constructed(self) -> None:
        sample = EvalSample(
            question="What is consciousness?",
            answer="It is the hard problem.",
            contexts=["Context passage."],
            ground_truth="Consciousness involves subjective experience.",
        )
        assert sample.question == "What is consciousness?"
        assert len(sample.contexts) == 1


class TestRAGEvaluatorMocked:
    """Test RAGEvaluator by mocking RAGAS internals."""

    def _make_samples(self, n: int = 2) -> list[EvalSample]:
        return [
            EvalSample(
                question=f"Q{i}?",
                answer=f"A{i}.",
                contexts=[f"Ctx{i}."],
                ground_truth=f"GT{i}.",
            )
            for i in range(n)
        ]

    def test_evaluate_returns_eval_result(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_scores = {
            "faithfulness": [0.8, 0.9],
            "answer_relevancy": [0.7, 0.8],
            "context_precision": [0.85, 0.95],
            "context_recall": [0.6, 0.7],
        }

        # Patch the lazy imports inside the method
        import sys
        import types

        fake_ragas = types.ModuleType("ragas")
        fake_ragas.EvaluationDataset = lambda samples: None  # type: ignore
        fake_ragas.evaluate = lambda **_: fake_scores  # type: ignore

        fake_schema = types.ModuleType("ragas.dataset_schema")
        fake_schema.SingleTurnSample = lambda **_: None  # type: ignore

        fake_collections = types.ModuleType("ragas.metrics.collections")
        for cls in (
            "Faithfulness",
            "AnswerRelevancy",
            "ContextPrecision",
            "ContextRecall",
        ):  # noqa: E501
            setattr(fake_collections, cls, MagicMockClass)

        monkeypatch.setitem(sys.modules, "ragas", fake_ragas)
        monkeypatch.setitem(sys.modules, "ragas.dataset_schema", fake_schema)
        monkeypatch.setitem(sys.modules, "ragas.metrics.collections", fake_collections)

        result = RAGEvaluator().evaluate(self._make_samples())

        assert isinstance(result, EvalResult)
        assert result.faithfulness == pytest.approx(0.85)
        assert result.answer_relevancy == pytest.approx(0.75)
        assert result.context_precision == pytest.approx(0.90)
        assert result.context_recall == pytest.approx(0.65)

    def test_mean_skips_none_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Scores with None entries (failed rows) should be averaged over valid
        values only."""
        fake_scores = {
            "faithfulness": [0.8, None, 1.0],
            "answer_relevancy": [0.6, 0.8, None],
            "context_precision": [None, None, None],
            "context_recall": [1.0],
        }

        import sys
        import types

        fake_ragas = types.ModuleType("ragas")
        fake_ragas.EvaluationDataset = lambda samples: None  # type: ignore
        fake_ragas.evaluate = lambda **_: fake_scores  # type: ignore
        fake_schema = types.ModuleType("ragas.dataset_schema")
        fake_schema.SingleTurnSample = lambda **_: None  # type: ignore
        fake_collections = types.ModuleType("ragas.metrics.collections")
        for cls in (
            "Faithfulness",
            "AnswerRelevancy",
            "ContextPrecision",
            "ContextRecall",
        ):  # noqa: E501
            setattr(fake_collections, cls, MagicMockClass)

        monkeypatch.setitem(sys.modules, "ragas", fake_ragas)
        monkeypatch.setitem(sys.modules, "ragas.dataset_schema", fake_schema)
        monkeypatch.setitem(sys.modules, "ragas.metrics.collections", fake_collections)

        result = RAGEvaluator().evaluate(self._make_samples(3))

        assert result.faithfulness == pytest.approx(0.9)  # mean(0.8, 1.0)
        assert result.answer_relevancy == pytest.approx(0.7)  # mean(0.6, 0.8)
        import math

        assert math.isnan(result.context_precision)  # all None → nan
        assert result.context_recall == pytest.approx(1.0)


class MagicMockClass:
    """Minimal stand-in for a RAGAS metric class (needs to be instantiable)."""

    def __init__(self) -> None:
        pass
