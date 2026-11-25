
from __future__ import annotations
import math
from contextlib import contextmanager
from time import perf_counter
from typing import Iterator, Literal

from agent_obsidian.observability.models import (
    PerformanceMetrics,
)


StageName = Literal[
    "retrieval",
    "context_build",
    "prompt_build",
    "llm_inference",
    "total",
]

CounterName = Literal[
    "retrieved_notes",
    "context_items",
    "estimated_context_tokens",
    "estimated_prompt_tokens",
]


class PerformanceCollector:
    """Collect elapsed durations and counters for one agent execution."""

    def __init__(self) -> None:
        self._durations_ms: dict[str, float] = {}
        self._counters: dict[str, int] = {}

    @contextmanager
    def measure(self, stage: StageName) -> Iterator[None]:
        """Measure a stage duration, including when it raises an error."""

        start = perf_counter()

        try:
            yield
        finally:
            elapsed_ms = (perf_counter() - start) * 1000.0
            self.record_duration(stage, elapsed_ms)

    def record_duration(
        self,
        stage: StageName,
        duration_ms: float,
    ) -> None:
        """Record or accumulate a stage duration."""

        if stage not in (
            "retrieval",
            "context_build",
            "prompt_build",
            "llm_inference",
            "total",
        ):
            raise ValueError(f"Unsupported performance stage: {stage}")

        if (
            isinstance(duration_ms, bool)
            or not isinstance(duration_ms, (int, float))
            or not math.isfinite(duration_ms)
            or duration_ms < 0
        ):
            raise ValueError(
                "duration_ms must be a finite non-negative number"
            )

        self._durations_ms[stage] = (
            self._durations_ms.get(stage, 0.0)
            + float(duration_ms)
        )

    def record_count(
        self,
        name: CounterName,
        value: int,
    ) -> None:
        """Record a non-negative integer counter."""

        if name not in (
            "retrieved_notes",
            "context_items",
            "estimated_context_tokens",
            "estimated_prompt_tokens",
        ):
            raise ValueError(f"Unsupported performance counter: {name}")

        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
        ):
            raise ValueError(
                f"{name} must be a non-negative integer"
            )

        self._counters[name] = value

    def snapshot(self) -> PerformanceMetrics:
        """Return an immutable snapshot of the collected metrics."""

        return PerformanceMetrics(
            retrieval_ms=self._durations_ms.get("retrieval", 0.0),
            context_build_ms=self._durations_ms.get(
                "context_build",
                0.0,
            ),
            prompt_build_ms=self._durations_ms.get(
                "prompt_build",
                0.0,
            ),
            llm_inference_ms=self._durations_ms.get(
                "llm_inference",
                0.0,
            ),
            total_ms=self._durations_ms.get("total", 0.0),
            retrieved_notes=self._counters.get(
                "retrieved_notes",
                0,
            ),
            context_items=self._counters.get("context_items", 0),
            estimated_context_tokens=self._counters.get(
                "estimated_context_tokens",
                0,
            ),
            estimated_prompt_tokens=self._counters.get(
                "estimated_prompt_tokens",
                0,
            ),
        )