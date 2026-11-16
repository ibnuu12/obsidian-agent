
from types import SimpleNamespace

import pytest

from agent_obsidian.infrastructure.llm import (
    MockLLMProvider,
    create_llm_provider,
)

def test_factory_returns_none_when_disabled():
    settings = SimpleNamespace(
        llm_provider="none",
        llm_model="",
    )

    provider = create_llm_provider(settings)

    assert provider is None


def test_factory_creates_mock_provider():
    settings = SimpleNamespace(
        llm_provider="mock",
        llm_model="test-model",
    )

    provider = create_llm_provider(settings)

    assert isinstance(provider, MockLLMProvider)
    assert provider.provider_name == "mock"


def test_factory_uses_default_mock_model():
    settings = SimpleNamespace(
        llm_provider="mock",
        llm_model="",
    )

    provider = create_llm_provider(settings)

    assert isinstance(provider, MockLLMProvider)


def test_factory_rejects_unsupported_provider():
    settings = SimpleNamespace(
        llm_provider="unknown",
        llm_model="test-model",
    )

    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        create_llm_provider(settings)


def test_factory_normalizes_provider_name():
    settings = SimpleNamespace(
        llm_provider=" MOCK ",
        llm_model="test-model",
    )

    provider = create_llm_provider(settings)

    assert isinstance(provider, MockLLMProvider)
