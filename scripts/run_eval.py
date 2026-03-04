"""CLI evaluation script.

Loads the RAG pipeline, runs it over data/eval_set.json, and prints RAGAS scores.

Usage:
    python scripts/run_eval.py [--eval-set data/eval_set.json] [--top-k 5]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Make the project importable when run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phil_mind_rag.config import get_settings
from phil_mind_rag.eval.evaluator import EvalSample, RAGEvaluator
from phil_mind_rag.pipeline import RAGPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation over the pipeline.")
    parser.add_argument(
        "--eval-set",
        type=Path,
        default=Path("data/eval_set.json"),
        help="Path to the JSON eval set (list of {question, ground_truth} objects).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve per question.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger("run_eval")

    if not args.eval_set.exists():
        logger.error("Eval set not found: %s", args.eval_set)
        sys.exit(1)

    raw = json.loads(args.eval_set.read_text())
    logger.info("Loaded %d eval samples from %s", len(raw), args.eval_set)

    settings = get_settings()
    pipeline = RAGPipeline(settings)

    if pipeline.document_count == 0:
        logger.error(
            "Vector store is empty. Ingest documents first (e.g. via the Gradio UI or pipeline.ingest())."
        )
        sys.exit(1)

    logger.info("Generating answers for %d questions (top_k=%d)…", len(raw), args.top_k)
    samples: list[EvalSample] = []
    for i, item in enumerate(raw, 1):
        question = item["question"]
        ground_truth = item["ground_truth"]

        contexts = pipeline.retrieve(question, top_k=args.top_k)
        answer = pipeline.query(question, top_k=args.top_k)

        samples.append(
            EvalSample(
                question=question,
                answer=answer,
                contexts=[r.text for r in contexts],
                ground_truth=ground_truth,
            )
        )
        logger.info("[%d/%d] answered: %s…", i, len(raw), question[:60])

    logger.info("Running RAGAS evaluation…")
    evaluator = RAGEvaluator()
    result = evaluator.evaluate(samples)

    print("\n=== RAGAS Evaluation Results ===")
    print(result.summary())
    print("================================\n")


if __name__ == "__main__":
    main()
