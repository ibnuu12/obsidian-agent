import asyncio
import json

import httpx2
import pytest

from agent_obsidian.application.llm import (
    LLMMessage,
    LLMRequest,
)
from agent_obsidian.application.llm.exceptions import (
    LLMAuthenticationError,
    LLMInvalidResponseError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from agent_obsidian.infrastructure.llm.ollama_provider import (
    OllamaLLMProvider,
)


def make_request() -> LLMRequest:
    return LLMRequest(
        messages=(
            LLMMessage(
                role="system",
                content="You are a helpful assistant.",
            ),
            LLMMessage(
                role="user",
                content="Explain knowledge graphs.",
            ),
        ),
        temperature=0.5,
        max_output_tokens=512,
    )


def test_ollama_provider_name():
    provider = OllamaLLMProvider()

    assert provider.provider_name == "ollama"


def test_ollama_provider_generates_response():
    async def handler(request: httpx2.Request):
        payload = json.loads(request.content)

        assert request.url.path == "/api/chat"
        assert payload["model"] == "qwen2.5:3b"
        assert payload["stream"] is False
        assert payload["options"]["temperature"] == 0.5
        assert payload["options"]["num_predict"] == 512
        assert len(payload["messages"]) == 2

        return httpx2.Response(
            200,
            json={
                "model": "qwen2.5:3b",
                "message": {
                    "role": "assistant",
                    "content": "A knowledge graph connects related entities.",
                },
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 100,
                "eval_count": 25,
            },
        )

    provider = OllamaLLMProvider(
        transport=httpx2.MockTransport(handler),
    )

    response = asyncio.run(
        provider.generate(make_request())
    )

    assert response.provider == "ollama"
    assert response.model == "qwen2.5:3b"
    assert response.finish_reason == "stop"
    assert response.usage.input_tokens == 100
    assert response.usage.output_tokens == 25
    assert response.usage.total_tokens == 125


def test_ollama_provider_rejects_invalid_json():
    async def handler(request):
        return httpx2.Response(
            200,
            content=b"invalid-json",
        )

    provider = OllamaLLMProvider(
        transport=httpx2.MockTransport(handler),
    )

    with pytest.raises(LLMInvalidResponseError):
        asyncio.run(provider.generate(make_request()))


def test_ollama_provider_rejects_invalid_response_structure():
    async def handler(request):
        return httpx2.Response(
            200,
            json={"unexpected": "response"},
        )

    provider = OllamaLLMProvider(
        transport=httpx2.MockTransport(handler),
    )

    with pytest.raises(LLMInvalidResponseError):
        asyncio.run(provider.generate(make_request()))


@pytest.mark.parametrize(
    ("status_code", "expected_exception"),
    [
        (401, LLMAuthenticationError),
        (403, LLMAuthenticationError),
        (429, LLMRateLimitError),
        (500, LLMProviderError),
    ],
)
def test_ollama_provider_maps_http_errors(
    status_code,
    expected_exception,
):
    async def handler(request):
        return httpx2.Response(status_code)

    provider = OllamaLLMProvider(
        transport=httpx2.MockTransport(handler),
    )

    with pytest.raises(expected_exception):
        asyncio.run(provider.generate(make_request()))


def test_ollama_provider_maps_timeout():
    async def handler(request):
        raise httpx2.ReadTimeout("Request timed out")

    provider = OllamaLLMProvider(
        transport=httpx2.MockTransport(handler),
    )

    with pytest.raises(LLMTimeoutError):
        asyncio.run(provider.generate(make_request()))


def test_ollama_provider_maps_connection_error():
    async def handler(request):
        raise httpx2.ConnectError("Connection refused")

    provider = OllamaLLMProvider(
        transport=httpx2.MockTransport(handler),
    )

    with pytest.raises(LLMProviderError):
        asyncio.run(provider.generate(make_request()))


def test_ollama_provider_rejects_invalid_token_count():
    async def handler(request):
        return httpx2.Response(
            200,
            json={
                "model": "qwen2.5:3b",
                "message": {
                    "content": "Test response",
                },
                "prompt_eval_count": -1,
                "eval_count": 10,
            },
        )

    provider = OllamaLLMProvider(
        transport=httpx2.MockTransport(handler),
    )

    with pytest.raises(LLMInvalidResponseError):
        asyncio.run(provider.generate(make_request()))