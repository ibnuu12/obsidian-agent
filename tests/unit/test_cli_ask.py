
from __future__ import annotations

from agent_obsidian.application.agent_models import (
    AgentResponse,
    RetrievalOptions,
)
from agent_obsidian.application.context_models import ContextItem
from agent_obsidian.application.llm.models import LLMUsage
from agent_obsidian.interfaces.cli.commands import ask as ask_command
from agent_obsidian.interfaces.cli.presenters.console import (
    present_agent_response,
)


def make_response():
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
