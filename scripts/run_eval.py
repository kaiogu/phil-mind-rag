"""CLI evaluation script.

Loads the RAG pipeline, runs it over data/eval_set.json, and prints RAGAS scores.

Usage:
    python scripts/run_eval.py [--eval-set data/eval_set.json] [--top-k 5]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make the project importable when run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phil_mind_rag.config import get_settings
from phil_mind_rag.eval.evaluator import EvalSample, RAGEvaluator, load_eval_questions
from phil_mind_rag.pipeline import RAGPipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation over the pipeline."
    )
    parser.add_argument(
        "--eval-set",
        type=Path,
        default=Path("data/eval_set.json"),
        help="Path to the JSON eval set.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve per question.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    logger = logging.getLogger("run_eval")

    if not args.eval_set.exists():
        logger.error("Eval set not found: %s", args.eval_set)
        sys.exit(1)

    questions = load_eval_questions(args.eval_set)
    logger.info("Loaded %d eval questions from %s", len(questions), args.eval_set)

    settings = get_settings()
    pipeline = RAGPipeline(settings)

    if pipeline.document_count == 0:
        logger.error(
            "Vector store is empty. Ingest documents first "
            "(e.g. via the Gradio UI or pipeline.ingest())."
        )
        sys.exit(1)

    logger.info(
        "Generating answers for %d questions (top_k=%d)…",
        len(questions),
        args.top_k,
    )
    samples: list[EvalSample] = []
    for i, eval_question in enumerate(questions, 1):
        contexts = pipeline.retrieve(eval_question.question, top_k=args.top_k)
        answer = pipeline.query(eval_question.question, top_k=args.top_k)

        samples.append(
            eval_question.to_sample(
                answer=answer,
                contexts=[r.text for r in contexts],
            )
        )
        logger.info(
            "[%d/%d] answered %s: %s…",
            i,
            len(questions),
            eval_question.id,
            eval_question.question[:60],
        )

    logger.info("Running RAGAS evaluation…")
    evaluator = RAGEvaluator()
    result = evaluator.evaluate(samples)

    print("\n=== RAGAS Evaluation Results ===")
    print(result.summary())
    print("================================\n")


if __name__ == "__main__":
    main()
