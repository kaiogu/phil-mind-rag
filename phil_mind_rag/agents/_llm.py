"""Structured-output helper — single call site for OpenAI .parse() endpoint."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from openai import OpenAI

logger = logging.getLogger(__name__)


def generate_structured[T: BaseModel](
    client: OpenAI,
    model: str | tuple[str, ...],
    system: str,
    user: str,
    schema_cls: type[T],
) -> T:
    """Call an OpenAI-compatible structured-output endpoint."""
    models = (model,) if isinstance(model, str) else model
    last_exc: Exception | None = None
    for candidate_model in models:
        try:
            response = client.chat.completions.create(
                model=candidate_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_cls.__name__,
                        "strict": True,
                        "schema": schema_cls.model_json_schema(),
                    },
                },
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError(
                    f"Structured output returned None for {schema_cls.__name__}"
                )
            if isinstance(content, str):
                return schema_cls.model_validate(json.loads(content))
            text = "".join(
                part.text
                for part in content
                if getattr(part, "type", None) == "text" and getattr(part, "text", None)
            )
            if not text:
                raise ValueError(
                    "Structured output returned empty content for "
                    f"{schema_cls.__name__}"
                )
            return schema_cls.model_validate(json.loads(text))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Structured generation failed with %s for %s: %s",
                candidate_model,
                schema_cls.__name__,
                exc,
            )
            last_exc = exc

    if last_exc is not None:
        raise RuntimeError(
            f"All structured generation attempts failed for {schema_cls.__name__}"
        ) from last_exc
    raise RuntimeError(
        f"All structured generation attempts returned empty content for "
        f"{schema_cls.__name__}"
    )
