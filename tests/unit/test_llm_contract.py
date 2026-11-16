
import pytest
import math

from agent_obsidian.application.llm import (
    LLMMessage,
    LLMRequest,
    LLMResponse,
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
