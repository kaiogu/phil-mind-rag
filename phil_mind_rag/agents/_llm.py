"""Structured-output helper — single call site for OpenAI .parse() endpoint."""

from __future__ import annotations

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
    """Call the OpenAI structured-outputs endpoint and return a validated model."""
    response = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format=schema_cls,
    )
    result = response.choices[0].message.parsed
    if result is None:
        raise ValueError(f"Structured output returned None for {schema_cls.__name__}")
    return result
