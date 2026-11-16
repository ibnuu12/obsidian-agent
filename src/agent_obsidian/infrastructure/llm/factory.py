
from __future__ import annotations

from typing import TYPE_CHECKING

from agent_obsidian.application.llm.provider import LLMProvider
from agent_obsidian.infrastructure.llm.mock_provider import (
    MockLLMProvider,
)
from agent_obsidian.infrastructure.llm.ollama_provider import (
    OllamaLLMProvider,
)

if TYPE_CHECKING:
    from agent_obsidian.shared.config import Settings


def create_llm_provider(
    settings: Settings,
) -> LLMProvider | None:
    """
    Create an LLM provider based on application settings.

    Returns None when LLM is disabled.
    """

    provider_name = settings.llm_provider.strip().lower()

    if provider_name == "none":
        return None

    if provider_name == "mock":
        return MockLLMProvider(
            model_name=settings.llm_model or "mock-model",
        )

    if provider_name == "ollama":
        return OllamaLLMProvider(
            base_url=settings.llm_base_url,
            model_name=settings.llm_model,
            timeout=settings.llm_timeout_seconds,
        )

    raise ValueError(
        f"Unsupported LLM provider: {provider_name}"
    )
