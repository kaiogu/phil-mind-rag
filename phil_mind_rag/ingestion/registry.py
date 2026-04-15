from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DocumentRecord:
    source: str           # PDF filename e.g. "chalmers1995.pdf"
    title: str            # user-supplied or filename stem
    author: str           # user-supplied or ""
    chunk_count: int
    chunk_size: int
    chunk_overlap: int
    chunker: str          # fully-qualified class name
    embedding_model: str
    ingested_at: str      # ISO-8601 UTC string


class DocumentRegistry:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> list[dict[str, Any]]:
        if not self._path.exists():
            return []
        with self._path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def _save(self, records: list[dict[str, Any]]) -> None:
        with self._path.open("w", encoding="utf-8") as fh:
            json.dump(records, fh, indent=2, ensure_ascii=False)

    def add(self, record: DocumentRecord) -> None:
        """Upsert by source filename."""
        records = self._load()
        records = [r for r in records if r["source"] != record.source]
        records.append(asdict(record))
        self._save(records)
        logger.info("Registry: upserted '%s'", record.source)

    def list_all(self) -> list[DocumentRecord]:
        return [DocumentRecord(**r) for r in self._load()]

    def get(self, source: str) -> DocumentRecord | None:
        for raw in self._load():
            if raw["source"] == source:
                return DocumentRecord(**raw)
        return None
