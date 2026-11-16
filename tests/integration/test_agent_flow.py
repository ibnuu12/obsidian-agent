
from __future__ import annotations

import asyncio
from agent_obsidian.infrastructure.retrieval.note_parser import parse_note

from dataclasses import replace

import pytest

from agent_obsidian.application.agent_models import (
    AgentRequest,
    RetrievalOptions,
)
from agent_obsidian.application.llm.exceptions import (
    LLMNotConfiguredError,
)
from agent_obsidian.application.llm.models import (
    LLMRequest,
    LLMResponse,
    LLMUsage,
)
from agent_obsidian.infrastructure.agent import (
    create_agent_runtime,
)
from agent_obsidian.infrastructure.agent import factory as agent_factory
from agent_obsidian.infrastructure.retrieval.note_parser import (
    parse_note,
)
from agent_obsidian.shared.config import load_settings


class IntegrationTestProvider:
    """Deterministic provider for end-to-end integration tests."""

    provider_name = "integration-test"

    def __init__(self):
        self.requests: list[LLMRequest] = []
        self.usage = LLMUsage(
            input_tokens=120,
            output_tokens=35,
            total_tokens=155,
        )

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)

        return LLMResponse(
            content="A knowledge graph connects related knowledge notes.",
            provider=self.provider_name,
            model="integration-test-model",
            finish_reason="stop",
            usage=self.usage,
        )


def make_settings(**overrides):
    settings = load_settings()
    return replace(settings, **overrides)


def seed_knowledge_note(runtime):
    note = parse_note(
        "30 Notes/Knowledge Graph.md",
        """---
id: 202611130001
title: Knowledge Graph
aliases:
  - Knowledge Network
type: permanent
status: evergreen
tags:
  - knowledge
  - graph
created: 2026-11-13
updated: 2026-11-13
---

A knowledge graph represents entities and relationships
between connected knowledge notes.

It helps organize information and discover connections
between concepts in an Obsidian vault.
""",
    )

    runtime.index.upsert_note(note)
    runtime.index.rebuild_relationship_resolution()


def test_agent_end_to_end_flow(tmp_path, monkeypatch):
    provider = IntegrationTestProvider()

    settings = make_settings(
        database_path=tmp_path / "agent.db",
        llm_provider="mock",
        llm_model="integration-test-model",
    )

    monkeypatch.setattr(
        agent_factory,
        "create_llm_provider",
        lambda settings: provider,
    )

    with create_agent_runtime(settings) as runtime:
        seed_knowledge_note(runtime)

        response = asyncio.run(
            runtime.orchestrator.run(
                AgentRequest(
                    query="What is a knowledge graph?",
                    retrieval=RetrievalOptions(
                        limit=5,
                        expand_graph=False,
                        tags=frozenset({"knowledge"}),
                    ),
                )
            )
        )

    # Answer propagation
    assert response.answer == (
        "A knowledge graph connects related knowledge notes."
    )

    # Source propagation
    assert len(response.sources) == 1

    source = response.sources[0]

    assert source.path == "30 Notes/Knowledge Graph.md"
    assert source.title == "Knowledge Graph"
    assert source.source == "fts"
    assert "knowledge" in source.excerpt.lower()
    assert "graph" in source.excerpt.lower()

    # Provider metadata propagation
    assert response.provider == "integration-test"
    assert response.model == "integration-test-model"

    # Token usage propagation
    assert response.usage == provider.usage

    # Ensure the LLM received the retrieved context
    assert len(provider.requests) == 1

    user_message = provider.requests[0].messages[1].content

    assert "Knowledge Graph" in user_message
    assert "connected" in user_message
    assert "[knowledge]" in user_message
    assert "notes" in user_message


def test_agent_without_configured_provider(tmp_path):
    settings = make_settings(
        database_path=tmp_path / "unconfigured.db",
        llm_provider="none",
    )

    with create_agent_runtime(settings) as runtime:
        with pytest.raises(LLMNotConfiguredError):
            asyncio.run(
                runtime.orchestrator.run(
                    AgentRequest(query="What is a knowledge graph?")
                )
            )

def test_agent_end_to_end_with_graph_expansion(
    tmp_path,
    monkeypatch,
):
    provider = IntegrationTestProvider()

    settings = make_settings(
        database_path=tmp_path / "agent.db",
        llm_provider="mock",
        llm_model="integration-test-model",
    )

    monkeypatch.setattr(
        agent_factory,
        "create_llm_provider",
        lambda _: provider,
    )

    with agent_factory.create_agent_runtime(settings) as runtime:
        runtime.index.upsert_note(
            parse_note(
                "30 Notes/Atomic Knowledge.md",
                """---
        title: Atomic Knowledge
        type: permanent
        status: evergreen
        tags:
        - knowledge
        ---
        # Atomic Knowledge

        Atomic knowledge is a fundamental concept.

        Related to [[Knowledge Graph]].
        """,
            )
        )

        runtime.index.upsert_note(
            parse_note(
                "30 Notes/Knowledge Graph.md",
                """---
        title: Knowledge Graph
        type: permanent
        status: evergreen
        tags:
        - knowledge
        ---
        # Knowledge Graph

        A graph of connected knowledge notes.
        """,
            )
        )

        runtime.index.rebuild_relationship_resolution()

        response = asyncio.run(
            runtime.orchestrator.run(
                AgentRequest(
                    query="atomic knowledge",
                    retrieval=RetrievalOptions(
                        limit=5,
                        expand_graph=True,
                        graph_limit=3,
                    ),
                )
            )
        )

    assert response.answer
    assert len(response.sources) == 2

    direct = next(
        source
        for source in response.sources
        if source.path == "30 Notes/Atomic Knowledge.md"
    )

    graph = next(
        source
        for source in response.sources
        if source.path == "30 Notes/Knowledge Graph.md"
    )

    assert direct.source == "fts"
    assert graph.source == "graph"

    assert len(provider.requests) == 1

    user_message = provider.requests[0].messages[1].content

    assert "Atomic Knowledge" in user_message
    assert "Knowledge Graph" in user_message