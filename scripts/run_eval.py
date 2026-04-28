"""CLI evaluation script.

Loads the RAG pipeline, runs it over data/eval_set.json, and prints RAGAS scores.

Usage:
    python scripts/run_eval.py [--eval-set data/eval_set.json] [--top-k 5]
    python scripts/run_eval.py --by-type          # also print per-question-type table
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

# Make the project importable when run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phil_mind_rag.config import get_settings
from phil_mind_rag.eval.evaluator import EvalSample, RAGEvaluator, load_eval_questions
from phil_mind_rag.eval.reporting import timestamped_eval_dir, write_json_report
from phil_mind_rag.eval.retrieval_evaluator import evaluate_retrieval
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
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Optional base directory for timestamped JSON eval reports.",
    )
    parser.add_argument(
        "--by-type",
        action="store_true",
        default=False,
        help="Also run RAGAS grouped by question_type (skips types with < 3 samples).",
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
    retrieved_by_eval_id = {}
    for i, eval_question in enumerate(questions, 1):
        contexts = pipeline.retrieve(eval_question.question, top_k=args.top_k)
        retrieved_by_eval_id[eval_question.id] = contexts
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

    retrieval_result = evaluate_retrieval(
        questions=questions,
        retrieved_by_eval_id=retrieved_by_eval_id,
    )
    print("\n=== Retrieval Evaluation Results ===")
    print(retrieval_result.summary())
    print("====================================\n")

    logger.info("Running RAGAS evaluation…")
    evaluator = RAGEvaluator()
    result = evaluator.evaluate(samples)

    print("\n=== RAGAS Evaluation Results ===")
    print(result.summary())
    print("================================\n")

    per_type_results: dict[str, object] = {}
    if args.by_type:
        by_type: dict[str, list] = defaultdict(list)
        for sample in samples:
            key = sample.question_type or "unknown"
            by_type[key].append(sample)

        print("\n=== RAGAS Results by Question Type ===")
        cols = f"{'Faith':>6}  {'Rel':>6}  {'Prec':>6}  {'Rec':>6}"
        header = f"{'Type':<28} {'n':>3}  {cols}"
        print(header)
        print("-" * len(header))
        for qtype in sorted(by_type):
            group = by_type[qtype]
            if len(group) < 3:
                skip_msg = "(skipped — fewer than 3 samples)"
                print(f"  {qtype:<26} {len(group):>3}  {skip_msg}")
                continue
            logger.info("Running RAGAS for type '%s' (%d samples)…", qtype, len(group))
            type_result = evaluator.evaluate(group)
            per_type_results[qtype] = type_result
            print(
                f"  {qtype:<26} {len(group):>3}"
                f"  {type_result.faithfulness:>6.3f}"
                f"  {type_result.answer_relevancy:>6.3f}"
                f"  {type_result.context_precision:>6.3f}"
                f"  {type_result.context_recall:>6.3f}"
            )
        print("=" * len(header) + "\n")

    if args.report_dir is not None:
        run_dir = timestamped_eval_dir(args.report_dir)
        write_json_report(
            run_dir / "retrieval.json",
            {
                "config": {
                    "eval_set": args.eval_set,
                    "top_k": args.top_k,
                },
                "result": retrieval_result,
            },
        )
        write_json_report(
            run_dir / "ragas.json",
            {
                "config": {
                    "eval_set": args.eval_set,
                    "top_k": args.top_k,
                },
                "result": result,
                "by_type": per_type_results,
            },
        )
        logger.info("Wrote eval reports to %s", run_dir)


if __name__ == "__main__":
    main()
