"""Tests for curated corpus metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CORPUS_PATH = Path("data/corpus_sources.json")

REQUIRED_FIELDS = {
    "id",
    "title",
    "authors",
    "year",
    "source_type",
    "venue",
    "citation",
    "doi",
    "source_url",
    "access_status",
    "license_status",
    "local_filename",
    "ingestion_status",
    "stance_tags",
    "school_tags",
    "eval_roles",
    "priority",
    "acquisition_note",
}

ALLOWED_ACCESS_STATUSES = {
    "archive_record",
    "copyrighted_book",
    "local_sample_copyrighted",
    "open_author_html",
    "paywalled",
    "repository_metadata",
    "repository_open",
}

ALLOWED_INGESTION_STATUSES = {"candidate", "local_sample"}


def _metadata() -> dict[str, Any]:
    return json.loads(CORPUS_PATH.read_text(encoding="utf-8"))


def test_corpus_metadata_has_expected_top_level_shape() -> None:
    metadata = _metadata()

    assert metadata["schema_version"] == 1
    assert isinstance(metadata["description"], str)
    assert 8 <= len(metadata["sources"]) <= 12


def test_corpus_sources_have_required_fields_and_unique_ids() -> None:
    sources = _metadata()["sources"]
    ids = [source["id"] for source in sources]

    assert len(ids) == len(set(ids))
    for source in sources:
        assert REQUIRED_FIELDS <= set(source)
        assert source["id"] == source["id"].strip()
        assert source["title"]
        assert source["authors"]
        assert isinstance(source["year"], int)
        assert source["source_url"].startswith("https://")
        assert source["access_status"] in ALLOWED_ACCESS_STATUSES
        assert source["ingestion_status"] in ALLOWED_INGESTION_STATUSES
        assert source["stance_tags"]
        assert source["school_tags"]
        assert source["eval_roles"]
        assert isinstance(source["priority"], int)
        assert source["acquisition_note"]


def test_starter_corpus_contains_expected_anchor_authors() -> None:
    authors = {
        author for source in _metadata()["sources"] for author in source["authors"]
    }

    assert {
        "Thomas Nagel",
        "David J. Chalmers",
        "Frank Jackson",
        "Paul M. Churchland",
        "Daniel C. Dennett",
        "Ned Block",
        "Hilary Putnam",
        "John R. Searle",
        "Michael Tye",
        "Francis Crick",
        "Christof Koch",
    } <= authors


def test_corpus_metadata_is_safe_to_commit() -> None:
    for source in _metadata()["sources"]:
        if source["access_status"] in {
            "paywalled",
            "copyrighted_book",
            "local_sample_copyrighted",
            "repository_metadata",
        }:
            assert source["local_filename"] is None or source["id"] == "nagel_bat"
            assert "copyright" in source["license_status"] or (
                source["license_status"] == "copyrighted_or_unknown"
            )


def test_nagel_is_the_only_local_sample_for_now() -> None:
    local_samples = [
        source
        for source in _metadata()["sources"]
        if source["ingestion_status"] == "local_sample"
    ]

    assert [source["id"] for source in local_samples] == ["nagel_bat"]
    assert local_samples[0]["local_filename"] == "nagel_bat.pdf"
