"""Run deterministic multi-agent evals and persist a JSON report.

Usage:
    uv run python scripts/run_agent_eval.py [--report-dir eval_runs]
    uv run python scripts/run_agent_eval.py --no-llm-judge  # skip LLM judge
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Make the project importable when run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phil_mind_rag.eval.agent_evaluator import LLMSemanticJudge
from phil_mind_rag.eval.agent_report import (
    build_smoke_agent_eval_report,
    write_smoke_agent_eval_report,
)
from phil_mind_rag.eval.reporting import write_json_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run offline deterministic multi-agent evals."
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("eval_runs"),
        help="Base directory for timestamped JSON eval reports.",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Also print the full JSON report payload.",
    )
    parser.add_argument(
        "--no-llm-judge",
        action="store_true",
        help="Skip the LLM semantic support judge (no API call for claim scoring).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    logger = logging.getLogger("run_agent_eval")

    semantic_judge = None
    if not args.no_llm_judge:
        try:
            from phil_mind_rag.config import get_settings
            from phil_mind_rag.providers import generation_client, generation_models

            settings = get_settings()
            client = generation_client(settings)
            model = generation_models(settings)[0]
            semantic_judge = LLMSemanticJudge(client=client, model=model)
            logger.info("LLM semantic judge enabled (model: %s)", model)
        except Exception:
            logger.warning(
                "Could not initialise LLM semantic judge — falling back to skip.",
                exc_info=True,
            )

    report = build_smoke_agent_eval_report(semantic_judge)
    result = report["result"]
    print("\n=== Multi-Agent Evaluation Results ===")
    print(result.summary())
    print("======================================\n")

    report_path = write_smoke_agent_eval_report(args.report_dir, semantic_judge)
    logger.info("Wrote agent eval report to %s", report_path)

    if args.stdout:
        temp_path = args.report_dir / ".stdout-agent-eval.json"
        write_json_report(temp_path, report)
        try:
            print(json.dumps(json.loads(temp_path.read_text()), indent=2))
        finally:
            temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
