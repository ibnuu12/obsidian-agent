
import asyncio

import pytest

from agent_obsidian.application.llm import (
    LLMMessage,
    LLMRequest,
)
from agent_obsidian.infrastructure.llm import MockLLMProvider


def test_mock_provider_name():
    provider = MockLLMProvider()

    assert provider.provider_name == "mock"


def test_mock_provider_generates_response():
    provider = MockLLMProvider(
        response_content="Knowledge graph connects related notes.",
        model_name="test-model",
    )

    request = LLMRequest(
        messages=(
            LLMMessage(
                role="user",
                content="Explain knowledge graphs.",
            ),
        )
    )

    response = asyncio.run(provider.generate(request))

    assert response.content == (
        "Knowledge graph connects related notes."
    )
    assert response.provider == "mock"
    assert response.model == "test-model"
    assert response.finish_reason == "stop"


def test_mock_provider_records_request():
    provider = MockLLMProvider()

    request = LLMRequest(
        messages=(
            LLMMessage(
                role="user",
                content="Test request",
            ),
        )
    )

    asyncio.run(provider.generate(request))

    assert len(provider.requests) == 1
    assert provider.requests[0] == request


def test_llm_message_rejects_invalid_role():
    with pytest.raises(ValueError):
        LLMMessage(
            role="tool",
            content="Unexpected role",
        )


def test_llm_message_rejects_empty_content():
    with pytest.raises(ValueError):
        LLMMessage(
            role="user",
            content="   ",
        )
