"""LLM judge: a single structured-output call against Claude Haiku.

Uses ``instructor`` to coerce the model's reply into a Pydantic model, so each
judging step returns a validated object instead of free text we have to parse.
One ``Judge`` is shared across a whole run (the Anthropic client is reused).
"""

from __future__ import annotations

from typing import TypeVar

import instructor
from anthropic import Anthropic
from jinja2 import Template
from pydantic import BaseModel

from evaluator.config import settings

T = TypeVar("T", bound=BaseModel)



class Judge:
    """Render a Jinja template and ask Claude for a structured answer."""

    def __init__(self) -> None:
        self._client = instructor.from_anthropic(
            Anthropic(api_key=settings.anthropic_api_key)
        )
        self._model = settings.judge_model

    def run(self, template: str, response_model: type[T], **vars: object) -> T:
        prompt = Template(template).render(**vars)
        return self._client.chat.completions.create(
            model=self._model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
            response_model=response_model,
        )