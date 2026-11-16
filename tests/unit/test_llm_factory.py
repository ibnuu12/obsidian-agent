
from dataclasses import replace

import pytest

from agent_obsidian.infrastructure.llm import (
    MockLLMProvider,
    OllamaLLMProvider,
    create_llm_provider,
)
from agent_obsidian.shared.config import load_settings


def make_settings(**overrides):
    settings = load_settings()

    defaults = {
        "llm_provider": "none",
        "llm_model": "",
        "llm_base_url": "http://127.0.0.1:11434",
        "llm_timeout_seconds": 30,
    }
    defaults.update(overrides)

    return replace(settings, **defaults)


def test_factory_returns_none_when_provider_disabled():
    provider = create_llm_provider(make_settings())

    assert provider is None


def test_factory_creates_mock_provider():
    provider = create_llm_provider(
        make_settings(
            llm_provider="mock",
            llm_model="mock-model",
        )
    )

    assert isinstance(provider, MockLLMProvider)
    assert provider.provider_name == "mock"


def test_factory_creates_ollama_provider():
    provider = create_llm_provider(
        make_settings(
            llm_provider="ollama",
            llm_model="qwen2.5:3b",
        )
    )

    assert isinstance(provider, OllamaLLMProvider)
    assert provider.provider_name == "ollama"


def test_factory_rejects_unsupported_provider():
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        create_llm_provider(
            make_settings(llm_provider="unsupported")
        )


def test_factory_passes_ollama_configuration():
    provider = create_llm_provider(
        make_settings(
            llm_provider="ollama",
            llm_model="qwen2.5:3b",
            llm_base_url="http://127.0.0.1:11434",
            llm_timeout_seconds=45,
        )
    )

    assert provider._base_url == "http://127.0.0.1:11434"
    assert provider._model_name == "qwen2.5:3b"
    assert provider._timeout == 45
