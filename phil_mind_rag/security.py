"""Security utilities — input validation, sanitisation, document checks."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# --- Query sanitisation -------------------------------------------------

# Patterns commonly used in prompt-injection attempts.
_INJECTION_MARKERS: list[str] = [
    "ignore previous instructions",
    "ignore above instructions",
    "disregard",
    "system prompt",
    "you are now",
    "new instructions",
    "override",
]

MAX_QUERY_LENGTH = 2_000  # characters


def sanitise_query(query: str) -> str:
    """Validate and clean a user query before it hits the LLM.

    Raises ValueError on suspicious or oversized input.
    """
    stripped = query.strip()
    if not stripped:
        raise ValueError("Query must not be empty.")

    if len(stripped) > MAX_QUERY_LENGTH:
        raise ValueError(
            f"Query exceeds maximum length ({MAX_QUERY_LENGTH} characters)."
        )

    lowered = stripped.lower()
    for marker in _INJECTION_MARKERS:
        if marker in lowered:
            logger.warning("Potential prompt-injection attempt blocked: %r", stripped)
            raise ValueError("Query contains disallowed content.")

    return stripped


# --- Document validation ------------------------------------------------


def validate_document(path: Path, *, allowed_exts: set[str], max_mb: int) -> None:
    """Check a document before ingestion.

    Raises ValueError if the file is too large, wrong type, or missing.
    """
    if not path.exists():
        raise ValueError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in allowed_exts:
        raise ValueError(f"Unsupported file type '{suffix}'. Allowed: {allowed_exts}")

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > max_mb:
        raise ValueError(f"File too large ({size_mb:.1f} MB). Maximum: {max_mb} MB.")
