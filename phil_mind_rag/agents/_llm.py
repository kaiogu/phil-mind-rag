"""Structured-output helper — single call site for OpenAI .parse() endpoint."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from openai import OpenAI


def generate_structured[T: BaseModel](
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    schema_cls: type[T],
) -> T:
    """Call an OpenAI-compatible structured-output endpoint."""
    response = client.chat.completions.create(
        model=model,
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
        raise ValueError(f"Structured output returned None for {schema_cls.__name__}")
    if isinstance(content, str):
        return schema_cls.model_validate(json.loads(content))
    text = "".join(
        part.text
        for part in content
        if getattr(part, "type", None) == "text" and getattr(part, "text", None)
    )
    if not text:
        raise ValueError(
            f"Structured output returned empty content for {schema_cls.__name__}"
        )
    return schema_cls.model_validate(json.loads(text))
