
from __future__ import annotations

from agent_obsidian.application.llm.models import (
    LLMRequest,
    LLMResponse,
    LLMUsage,
)


class MockLLMProvider:
    """Deterministic provider for development and testing."""

    def __init__(
        self,
        response_content: str = "Mock LLM response.",
        model_name: str = "mock-model",
    ) -> None:
        self._response_content = response_content
        self._model_name = model_name
        self.requests: list[LLMRequest] = []

    @property
    def provider_name(self) -> str:
        return "mock"

    async def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        self.requests.append(request)

        return LLMResponse(
            content=self._response_content,
            provider=self.provider_name,
            model=self._model_name,
            finish_reason="stop",
            usage=LLMUsage(),
        )
