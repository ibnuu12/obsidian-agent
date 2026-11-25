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
                "total_duration": 5_000_000_000,
                "load_duration": 500_000_000,
                "prompt_eval_duration": 1_000_000_000,
                "eval_duration": 2_500_000_000,
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

    diagnostics = response.runtime_metrics

    assert diagnostics is not None
    assert diagnostics.total_duration_ms == 5000.0
    assert diagnostics.load_duration_ms == 500.0
    assert diagnostics.prompt_eval_duration_ms == 1000.0
    assert diagnostics.eval_duration_ms == 2500.0
    assert diagnostics.tokens_per_second == 10.0

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

def test_ollama_provider_handles_missing_diagnostics():
    async def handler(request):
        return httpx2.Response(
            200,
            json={
                "model": "qwen2.5:3b",
                "message": {
                    "content": "Test response",
                },
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

    metrics = response.runtime_metrics

    assert metrics is not None
    assert metrics.total_duration_ms is None
    assert metrics.load_duration_ms is None
    assert metrics.prompt_eval_duration_ms is None
    assert metrics.eval_duration_ms is None
    assert metrics.tokens_per_second is None

    assert response.usage.input_tokens == 100
    assert response.usage.output_tokens == 25


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("total_duration", -1),
        ("load_duration", -1),
        ("prompt_eval_duration", -1),
        ("eval_duration", -1),
        ("total_duration", True),
        ("eval_duration", "invalid"),
    ],
)
def test_ollama_provider_rejects_invalid_diagnostics(key, value):
    async def handler(request):
        return httpx2.Response(
            200,
            json={
                "model": "qwen2.5:3b",
                "message": {
                    "content": "Test response",
                },
                "prompt_eval_count": 100,
                "eval_count": 25,
                key: value,
            },
        )

    provider = OllamaLLMProvider(
        transport=httpx2.MockTransport(handler),
    )

    with pytest.raises(LLMInvalidResponseError):
        asyncio.run(provider.generate(make_request()))


def test_ollama_provider_handles_zero_eval_duration():
    async def handler(request):
        return httpx2.Response(
            200,
            json={
                "model": "qwen2.5:3b",
                "message": {
                    "content": "Test response",
                },
                "eval_count": 25,
                "eval_duration": 0,
            },
        )

    provider = OllamaLLMProvider(
        transport=httpx2.MockTransport(handler),
    )

    response = asyncio.run(
        provider.generate(make_request())
    )

    assert response.runtime_metrics is not None
    assert response.runtime_metrics.eval_duration_ms == 0.0
    assert response.runtime_metrics.tokens_per_second is None

def test_ollama_provider_keeps_metrics_per_response():
    responses = [
        {
            "model": "qwen2.5:3b",
            "message": {"content": "First response"},
            "eval_count": 10,
            "eval_duration": 1_000_000_000,
        },
        {
            "model": "qwen2.5:3b",
            "message": {"content": "Second response"},
            "eval_count": 20,
            "eval_duration": 2_000_000_000,
        },
    ]

    async def handler(request):
        return httpx2.Response(
            200,
            json=responses.pop(0),
        )

    provider = OllamaLLMProvider(
        transport=httpx2.MockTransport(handler),
    )

    async def run_requests():
        first = await provider.generate(make_request())
        second = await provider.generate(make_request())
        return first, second

    first, second = asyncio.run(run_requests())

    assert first.runtime_metrics is not None
    assert second.runtime_metrics is not None

    assert first.runtime_metrics.tokens_per_second == 10.0
    assert second.runtime_metrics.tokens_per_second == 10.0

    assert first.usage.output_tokens == 10
    assert second.usage.output_tokens == 20