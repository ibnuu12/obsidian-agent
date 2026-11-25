
import pytest
import math

from agent_obsidian.application.llm import (
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMRuntimeMetrics,
    LLMUsage,
)


def test_llm_message_accepts_valid_role():
    message = LLMMessage(
        role="user",
        content="Explain knowledge graphs.",
    )

    assert message.role == "user"
    assert message.content == "Explain knowledge graphs."


def test_llm_request_requires_messages():
    with pytest.raises(ValueError):
        LLMRequest(messages=())


def test_llm_request_validates_temperature():
    with pytest.raises(ValueError):
        LLMRequest(
            messages=(LLMMessage("user", "Hello"),),
            temperature=3.0,
        )


def test_llm_request_validates_max_output_tokens():
    with pytest.raises(ValueError):
        LLMRequest(
            messages=(LLMMessage("user", "Hello"),),
            max_output_tokens=0,
        )


def test_llm_response_has_default_usage():
    response = LLMResponse(
        content="Hello!",
        provider="mock",
        model="mock-model",
    )

    assert response.content == "Hello!"
    assert response.usage == LLMUsage()
    assert response.finish_reason is None

def test_llm_request_rejects_non_finite_temperature():
    message = LLMMessage("user", "Hello")

    for temperature in (
        math.nan,
        math.inf,
        -math.inf,
    ):
        with pytest.raises(ValueError):
            LLMRequest(
                messages=(message,),
                temperature=temperature,
            )


def test_llm_request_rejects_invalid_message_type():
    with pytest.raises(TypeError):
        LLMRequest(messages=("invalid",))


def test_llm_message_rejects_non_string_content():
    with pytest.raises(TypeError):
        LLMMessage(
            role="user",
            content=None,
        )

def test_llm_usage_rejects_negative_tokens():
    with pytest.raises(ValueError):
        LLMUsage(input_tokens=-1)


def test_llm_usage_rejects_non_integer_tokens():
    with pytest.raises(ValueError):
        LLMUsage(output_tokens=1.5)


def test_llm_response_rejects_invalid_usage():
    with pytest.raises(TypeError):
        LLMResponse(
            content="Hello",
            provider="mock",
            model="mock-model",
            usage=None,
        )


def test_llm_runtime_metrics_accepts_valid_values():
    metrics = LLMRuntimeMetrics(
        total_duration_ms=1200.5,
        load_duration_ms=100.0,
        prompt_eval_duration_ms=300.0,
        eval_duration_ms=800.0,
        tokens_per_second=25.5,
    )

    assert metrics.total_duration_ms == 1200.5
    assert metrics.load_duration_ms == 100.0
    assert metrics.prompt_eval_duration_ms == 300.0
    assert metrics.eval_duration_ms == 800.0
    assert metrics.tokens_per_second == 25.5


def test_llm_runtime_metrics_accepts_missing_values():
    metrics = LLMRuntimeMetrics()

    assert metrics.total_duration_ms is None
    assert metrics.load_duration_ms is None
    assert metrics.prompt_eval_duration_ms is None
    assert metrics.eval_duration_ms is None
    assert metrics.tokens_per_second is None


def test_llm_runtime_metrics_rejects_negative_values():
    for field_name in (
        "total_duration_ms",
        "load_duration_ms",
        "prompt_eval_duration_ms",
        "eval_duration_ms",
        "tokens_per_second",
    ):
        with pytest.raises(ValueError):
            LLMRuntimeMetrics(**{field_name: -1.0})


def test_llm_runtime_metrics_rejects_non_finite_values():
    for field_name in (
        "total_duration_ms",
        "load_duration_ms",
        "prompt_eval_duration_ms",
        "eval_duration_ms",
        "tokens_per_second",
    ):
        for value in (math.nan, math.inf, -math.inf):
            with pytest.raises(ValueError):
                LLMRuntimeMetrics(**{field_name: value})


def test_llm_runtime_metrics_rejects_invalid_types():
    for field_name in (
        "total_duration_ms",
        "load_duration_ms",
        "prompt_eval_duration_ms",
        "eval_duration_ms",
        "tokens_per_second",
    ):
        for value in (True, "100"):
            with pytest.raises(ValueError):
                LLMRuntimeMetrics(**{field_name: value})


def test_llm_response_accepts_runtime_metrics():
    metrics = LLMRuntimeMetrics(
        total_duration_ms=1500.0,
        tokens_per_second=20.0,
    )

    response = LLMResponse(
        content="Hello",
        provider="ollama",
        model="qwen2.5:3b",
        runtime_metrics=metrics,
    )

    assert response.runtime_metrics == metrics


def test_llm_response_rejects_invalid_runtime_metrics():
    with pytest.raises(TypeError):
        LLMResponse(
            content="Hello",
            provider="mock",
            model="mock-model",
            runtime_metrics="invalid",
        )


def test_llm_response_keeps_runtime_metrics_optional():
    response = LLMResponse(
        content="Hello",
        provider="mock",
        model="mock-model",
    )

    assert response.runtime_metrics is None
