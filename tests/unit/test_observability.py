
import pytest

from agent_obsidian.observability import (
    PerformanceCollector,
    PerformanceMetrics,
)
import agent_obsidian.observability.performance as performance_module


def test_performance_metrics_defaults():
    metrics = PerformanceMetrics()

    assert metrics.retrieval_ms == 0.0
    assert metrics.total_ms == 0.0
    assert metrics.retrieved_notes == 0


def test_performance_metrics_rejects_negative_duration():
    with pytest.raises(ValueError):
        PerformanceMetrics(retrieval_ms=-1.0)


def test_performance_metrics_rejects_invalid_counter():
    with pytest.raises(ValueError):
        PerformanceMetrics(retrieved_notes=-1)


def test_performance_metrics_is_immutable():
    metrics = PerformanceMetrics()

    with pytest.raises(AttributeError):
        metrics.total_ms = 10.0


def test_collector_records_duration(monkeypatch):
    timestamps = iter([10.0, 10.025])

    monkeypatch.setattr(
        performance_module,
        "perf_counter",
        lambda: next(timestamps),
    )

    collector = PerformanceCollector()

    with collector.measure("retrieval"):
        pass

    metrics = collector.snapshot()

    assert metrics.retrieval_ms == pytest.approx(25.0)


def test_collector_accumulates_repeated_stage():
    collector = PerformanceCollector()

    collector.record_duration("retrieval", 10.0)
    collector.record_duration("retrieval", 15.0)

    assert collector.snapshot().retrieval_ms == 25.0


def test_collector_records_counters():
    collector = PerformanceCollector()

    collector.record_count("retrieved_notes", 5)
    collector.record_count("context_items", 3)
    collector.record_count("estimated_context_tokens", 1200)
    collector.record_count("estimated_prompt_tokens", 1500)

    metrics = collector.snapshot()

    assert metrics.retrieved_notes == 5
    assert metrics.context_items == 3
    assert metrics.estimated_context_tokens == 1200
    assert metrics.estimated_prompt_tokens == 1500


def test_collector_records_duration_when_stage_fails():
    collector = PerformanceCollector()

    with pytest.raises(RuntimeError):
        with collector.measure("llm_inference"):
            raise RuntimeError("Simulated provider failure")

    assert collector.snapshot().llm_inference_ms >= 0.0


def test_collector_rejects_invalid_counter():
    collector = PerformanceCollector()

    with pytest.raises(ValueError):
        collector.record_count("retrieved_notes", -1)


def test_collector_rejects_invalid_stage():
    collector = PerformanceCollector()

    with pytest.raises(ValueError):
        collector.record_duration("unknown", 10.0)

def test_collector_rejects_non_finite_duration():
    collector = PerformanceCollector()

    with pytest.raises(ValueError):
        collector.record_duration("retrieval", float("nan"))

    with pytest.raises(ValueError):
        collector.record_duration("retrieval", float("inf"))

    with pytest.raises(ValueError):
        collector.record_duration("retrieval", float("-inf"))