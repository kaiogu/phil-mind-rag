"""Small helpers for writing eval report artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def timestamped_eval_dir(base_dir: Path, *, now: datetime | None = None) -> Path:
    """Create and return a timestamped eval-run directory."""
    current = now or datetime.now(UTC)
    run_dir = base_dir / current.strftime("%Y-%m-%dT%H-%M-%SZ")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_json_report(path: Path, payload: dict[str, Any]) -> None:
    """Write a stable, pretty JSON report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")


def _jsonable(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
