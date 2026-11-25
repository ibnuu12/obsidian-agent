
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal


MessageRole = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class LLMMessage:
    role: MessageRole
    content: str

    def __post_init__(self) -> None:
        if self.role not in ("system", "user", "assistant"):
            raise ValueError(
                f"Unsupported message role: {self.role}"
            )

        if not isinstance(self.content, str):
            raise TypeError("Message content must be a string")

        if not self.content.strip():
            raise ValueError("Message content cannot be empty")

@dataclass(frozen=True, slots=True)
class LLMRequest:
    messages: tuple[LLMMessage, ...]
    temperature: float = 0.2
    max_output_tokens: int = 1024

    def __post_init__(self) -> None:
        if not isinstance(self.messages, tuple):
            raise TypeError("messages must be a tuple")

        if not self.messages:
            raise ValueError(
                "LLM request must contain at least one message"
            )

        if not all(
            isinstance(message, LLMMessage)
            for message in self.messages
        ):
            raise TypeError(
                "All messages must be LLMMessage instances"
            )

        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not math.isfinite(self.temperature)
        ):
            raise ValueError("temperature must be a finite number")

        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(
                "temperature must be between 0.0 and 2.0"
            )

        if (
            isinstance(self.max_output_tokens, bool)
            or not isinstance(self.max_output_tokens, int)
            or self.max_output_tokens < 1
        ):
            raise ValueError(
                "max_output_tokens must be a positive integer"
            )


@dataclass(frozen=True, slots=True)
class LLMUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
        ):
            value = getattr(self, field_name)

            if value is None:
                continue

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(
                    f"{field_name} must be a non-negative integer"
                )

@dataclass(frozen=True, slots=True)
class LLMRuntimeMetrics:
    total_duration_ms: float | None = None
    load_duration_ms: float | None = None
    prompt_eval_duration_ms: float | None = None
    eval_duration_ms: float | None = None
    tokens_per_second: float | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "total_duration_ms",
            "load_duration_ms",
            "prompt_eval_duration_ms",
            "eval_duration_ms",
            "tokens_per_second",
        ):
            value = getattr(self, field_name)

            if value is None:
                continue

            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(
                    f"{field_name} must be a finite non-negative number"
                )

@dataclass(frozen=True, slots=True)
class LLMResponse:
    content: str
    provider: str
    model: str
    finish_reason: str | None = None
    usage: LLMUsage = field(default_factory=LLMUsage)
    runtime_metrics: LLMRuntimeMetrics | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.content, str):
            raise TypeError("Response content must be a string")

        if not isinstance(self.provider, str) or not self.provider.strip():
            raise ValueError("Provider name cannot be empty")

        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("Model name cannot be empty")

        if (
            self.finish_reason is not None
            and not isinstance(self.finish_reason, str)
        ):
            raise TypeError("finish_reason must be a string or None")

        if not isinstance(self.usage, LLMUsage):
            raise TypeError("usage must be an LLMUsage instance")

        if (
            self.runtime_metrics is not None
            and not isinstance(self.runtime_metrics, LLMRuntimeMetrics)
        ):
            raise TypeError(
                "runtime_metrics must be an LLMRuntimeMetrics instance or None"
            )