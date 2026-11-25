from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    """Immutable performance metrics for one agent execution."""

    retrieval_ms: float = 0.0
    context_build_ms: float = 0.0
    prompt_build_ms: float = 0.0
    llm_inference_ms: float = 0.0
    total_ms: float = 0.0

    retrieved_notes: int = 0
    context_items: int = 0
    estimated_context_tokens: int = 0
    estimated_prompt_tokens: int = 0

    def __post_init__(self) -> None:
        duration_fields = (
            "retrieval_ms",
            "context_build_ms",
            "prompt_build_ms",
            "llm_inference_ms",
            "total_ms",
        )

        count_fields = (
            "retrieved_notes",
            "context_items",
            "estimated_context_tokens",
            "estimated_prompt_tokens",
        )

        for field_name in duration_fields:
            value = getattr(self, field_name)

            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(
                    f"{field_name} must be a finite non-negative number"
                )

        for field_name in count_fields:
            value = getattr(self, field_name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(
                    f"{field_name} must be a non-negative integer"
                )