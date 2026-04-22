"""Tests for offline multi-agent eval report artifacts."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from phil_mind_rag.eval.agent_report import (
    build_smoke_agent_eval_report,
    write_smoke_agent_eval_report,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_build_smoke_agent_eval_report_is_offline_and_complete() -> None:
    report = build_smoke_agent_eval_report()
    result = report["result"]

    assert report["config"]["mode"] == "offline_smoke"
    assert report["config"]["semantic_support_judge"] == "not_configured"
    assert report["fixture"]["memo_stances"] == [
        "materialist",
        "idealist",
        "dualist",
    ]
    assert result.overall > 0
    assert "Semantic support" in result.summary()


def test_write_smoke_agent_eval_report_creates_agent_eval_json(
    tmp_path: Path,
) -> None:
    report_path = write_smoke_agent_eval_report(tmp_path)

    assert report_path.name == "agent_eval.json"
    assert report_path.parent.parent == tmp_path

    report = json.loads(report_path.read_text())
    assert report["config"]["mode"] == "offline_smoke"
    assert "overall" not in report["result"]
    assert report["result"]["grounding_fidelity"]["score"] > 0
