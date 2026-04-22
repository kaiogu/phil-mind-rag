"""Tests for eval report artifact helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from phil_mind_rag.eval.reporting import timestamped_eval_dir, write_json_report


@dataclass(frozen=True)
class NestedPayload:
    name: str
    path: Path


def test_timestamped_eval_dir_uses_stable_utc_name(tmp_path: Path) -> None:
    run_dir = timestamped_eval_dir(
        tmp_path,
        now=datetime(2026, 4, 22, 12, 30, 5, tzinfo=UTC),
    )

    assert run_dir == tmp_path / "2026-04-22T12-30-05Z"
    assert run_dir.exists()


def test_write_json_report_serializes_dataclasses_and_paths(tmp_path: Path) -> None:
    report_path = tmp_path / "reports" / "report.json"

    write_json_report(
        report_path,
        {
            "payload": NestedPayload(name="eval", path=Path("data/eval_set.json")),
        },
    )

    report = json.loads(report_path.read_text())
    assert report == {
        "payload": {
            "name": "eval",
            "path": "data/eval_set.json",
        }
    }
