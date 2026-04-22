"""Run deterministic multi-agent evals and persist a JSON report.

Usage:
    uv run python scripts/run_agent_eval.py [--report-dir eval_runs]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Make the project importable when run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    logger = logging.getLogger("run_agent_eval")

    report = build_smoke_agent_eval_report()
    result = report["result"]
    print("\n=== Multi-Agent Evaluation Results ===")
    print(result.summary())
    print("======================================\n")

    report_path = write_smoke_agent_eval_report(args.report_dir)
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
