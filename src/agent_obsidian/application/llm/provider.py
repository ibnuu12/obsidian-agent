
from __future__ import annotations

from typing import Protocol

from agent_obsidian.application.llm.models import (
    LLMRequest,
    LLMResponse,
)


class LLMProvider(Protocol):
    """Provider-independent interface for language models."""

    @property
    def provider_name(self) -> str:
        ...

    async def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        ...
