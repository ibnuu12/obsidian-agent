
from __future__ import annotations

from agent_obsidian.observability.models import PerformanceMetrics

from agent_obsidian.application.agent_models import (
    AgentResponse,
    RetrievalOptions,
)
from agent_obsidian.application.context_models import ContextItem
from agent_obsidian.application.llm.models import (
    LLMRuntimeMetrics,
    LLMUsage,
)
from agent_obsidian.interfaces.cli.commands import ask as ask_command
from agent_obsidian.interfaces.cli.presenters.console import (
    present_agent_response,
)


def make_response(performance=None, llm_runtime_metrics=None):
    source = ContextItem(
        path="30 Notes/Knowledge Graph.md",
        title="Knowledge Graph",
        note_type="permanent",
        status="evergreen",
        tags=("knowledge",),
        source="fts",
        score=-1.5,
        excerpt="A graph of connected knowledge.",
        estimated_tokens=10,
    )

    return AgentResponse(
        answer="A knowledge graph connects related notes.",
        sources=(source,),
        usage=LLMUsage(
            input_tokens=100,
            output_tokens=30,
            total_tokens=130,
        ),
        provider="mock",
        model="mock-model",
        performance=performance,
        llm_runtime_metrics=llm_runtime_metrics,
    )

class FakeOrchestrator:
    def __init__(self, response):
        self.response = response
        self.requests = []

    async def run(self, request):
        self.requests.append(request)
        return self.response


class FakeRuntime:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.entered = False
        self.exited = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.exited = True
        return False


def test_execute_ask_forwards_request_and_closes_runtime(
    monkeypatch,
):
    response = make_response()
    orchestrator = FakeOrchestrator(response)
    runtime = FakeRuntime(orchestrator)
    settings = object()

    presented = {}

    monkeypatch.setattr(
        ask_command,
        "load_settings",
        lambda: settings,
    )

    monkeypatch.setattr(
        ask_command,
        "create_agent_runtime",
        lambda received_settings: runtime,
    )

    def fake_presenter(query, received_response):
        presented["query"] = query
        presented["response"] = received_response

    monkeypatch.setattr(
        ask_command,
        "present_agent_response",
        fake_presenter,
    )

    retrieval = RetrievalOptions(
        limit=4,
        expand_graph=False,
        graph_limit=2,
        note_types=frozenset({"permanent"}),
        statuses=frozenset({"evergreen"}),
        tags=frozenset({"knowledge"}),
    )

    result = ask_command.execute_ask(
        query="Explain knowledge graphs",
        retrieval=retrieval,
    )

    assert result == 0

    assert runtime.entered is True
    assert runtime.exited is True

    assert len(orchestrator.requests) == 1

    request = orchestrator.requests[0]

    assert request.query == "Explain knowledge graphs"
    assert request.retrieval is retrieval

    assert presented["query"] == "Explain knowledge graphs"
    assert presented["response"] is response


def test_present_agent_response_displays_answer_and_sources(
    capsys,
):
    response = make_response()

    present_agent_response(
        query="Explain knowledge graphs",
        response=response,
    )

    output = capsys.readouterr().out

    assert "Agent Obsidian" in output
    assert "Explain knowledge graphs" in output
    assert "A knowledge graph connects related notes." in output

    assert "Knowledge Graph" in output
    assert "30 Notes/Knowledge Graph.md" in output
    assert "permanent" in output
    assert "fts" in output

    assert "Provider: mock" in output
    assert "Model: mock-model" in output
    assert "Input tokens: 100" in output
    assert "Output tokens: 30" in output
    assert "Total tokens: 130" in output
    assert "Performance:" not in output

def test_present_agent_response_handles_empty_sources(
    capsys,
):
    response = AgentResponse(
        answer="No relevant information was found.",
        sources=(),
        usage=None,
        provider="mock",
        model="mock-model",
    )

    present_agent_response(
        query="Unknown topic",
        response=response,
    )

    output = capsys.readouterr().out

    assert "No relevant sources found." in output
    assert "No relevant information was found." in output
    assert "Provider: mock" in output
    assert "Model: mock-model" in output
    assert "Input tokens:" not in output


def test_execute_ask_closes_runtime_when_orchestrator_fails(
    monkeypatch,
):
    class FailingOrchestrator:
        async def run(self, request):
            raise RuntimeError("Agent execution failed")

    runtime = FakeRuntime(FailingOrchestrator())
    presenter_called = False

    monkeypatch.setattr(
        ask_command,
        "load_settings",
        lambda: object(),
    )

    monkeypatch.setattr(
        ask_command,
        "create_agent_runtime",
        lambda settings: runtime,
    )

    def fake_presenter(query, response):
        nonlocal presenter_called
        presenter_called = True

    monkeypatch.setattr(
        ask_command,
        "present_agent_response",
        fake_presenter,
    )

    import pytest

    with pytest.raises(
        RuntimeError,
        match="Agent execution failed",
    ):
        ask_command.execute_ask(
            query="Test failure",
            retrieval=RetrievalOptions(),
        )

    assert runtime.entered is True
    assert runtime.exited is True
    assert presenter_called is False

def test_present_agent_response_displays_performance_metrics(
    capsys,
):
    metrics = PerformanceMetrics(
        retrieval_ms=12.45,
        context_build_ms=8.32,
        prompt_build_ms=1.15,
        llm_inference_ms=2450.80,
        total_ms=2472.72,
        retrieved_notes=5,
        context_items=4,
        estimated_context_tokens=850,
        estimated_prompt_tokens=1250,
    )

    response = make_response(performance=metrics)

    present_agent_response(
        query="Explain knowledge graphs",
        response=response,
    )

    output = capsys.readouterr().out

    assert "Performance:" in output

    assert "Retrieval: 12.45 ms" in output
    assert "Context Build: 8.32 ms" in output
    assert "Prompt Build: 1.15 ms" in output
    assert "LLM Inference: 2450.80 ms" in output
    assert "Total: 2472.72 ms" in output

    assert "Retrieved Notes: 5" in output
    assert "Context Items: 4" in output
    assert "Estimated Context Tokens: 850" in output
    assert "Estimated Prompt Tokens: 1250" in output

def test_present_agent_response_displays_llm_runtime_metrics(
    capsys,
):
    runtime_metrics = LLMRuntimeMetrics(
        total_duration_ms=5000.0,
        load_duration_ms=500.0,
        prompt_eval_duration_ms=1000.0,
        eval_duration_ms=2500.0,
        tokens_per_second=10.0,
    )

    response = make_response(
        llm_runtime_metrics=runtime_metrics,
    )

    present_agent_response(
        query="Explain knowledge graphs",
        response=response,
    )

    output = capsys.readouterr().out

    assert "LLM Runtime Diagnostics:" in output
    assert "Total Duration: 5000.00 ms" in output
    assert "Model Load: 500.00 ms" in output
    assert "Prompt Evaluation: 1000.00 ms" in output
    assert "Token Generation: 2500.00 ms" in output
    assert "Generation Speed: 10.00 tokens/s" in output


def test_present_agent_response_omits_llm_runtime_metrics_when_unavailable(
    capsys,
):
    response = make_response(
        llm_runtime_metrics=None,
    )

    present_agent_response(
        query="Explain knowledge graphs",
        response=response,
    )

    output = capsys.readouterr().out

    assert "LLM Runtime Diagnostics:" not in output