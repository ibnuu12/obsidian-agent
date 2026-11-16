
from __future__ import annotations

import asyncio
import sqlite3

import pytest

from agent_obsidian.application.agent_models import (
    AgentRequest,
    RetrievalOptions,
)
from agent_obsidian.application.agent_orchestrator import (
    AgentOrchestrator,
)
from agent_obsidian.application.context_builder import ContextBuilder
from agent_obsidian.application.hybrid_retrieval import (
    HybridRetrievalService,
)
from agent_obsidian.application.llm.models import (
    LLMRequest,
    LLMResponse,
    LLMUsage,
)
from agent_obsidian.infrastructure.retrieval.sqlite_index import (
    SQLiteIndex,
)
from agent_obsidian.shared.config import load_settings


class LiveVaultTestProvider:
    """Deterministic provider for live-vault integration testing."""

    provider_name = "live-vault-test"

    def __init__(self):
        self.requests: list[LLMRequest] = []
        self.usage = LLMUsage(
            input_tokens=150,
            output_tokens=40,
            total_tokens=190,
        )

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)

        return LLMResponse(
            content=(
                "The retrieved vault notes contain information "
                "about the Master Map of Content."
            ),
            provider=self.provider_name,
            model="live-vault-test-model",
            finish_reason="stop",
            usage=self.usage,
        )


def read_database_counts(database_path):
    """Read database counts using a read-only SQLite connection."""

    database_uri = (
        database_path.resolve().as_uri() + "?mode=ro"
    )

    connection = sqlite3.connect(
        database_uri,
        uri=True,
    )

    try:
        return {
            "notes": connection.execute(
                "SELECT COUNT(*) FROM notes"
            ).fetchone()[0],
            "fts": connection.execute(
                "SELECT COUNT(*) FROM notes_fts"
            ).fetchone()[0],
            "relationships": connection.execute(
                "SELECT COUNT(*) FROM note_relationships"
            ).fetchone()[0],
        }
    finally:
        connection.close()


def test_agent_end_to_end_with_live_vault():
    """Test the complete agent flow using the actual project database."""

    settings = load_settings()
    database_path = settings.database_path

    if not database_path.is_file():
        pytest.skip(
            f"Project database not found: {database_path}"
        )

    provider = LiveVaultTestProvider()

    with SQLiteIndex(database_path) as index:
        counts_before = read_database_counts(database_path)

        orchestrator = AgentOrchestrator(
            retrieval_service=HybridRetrievalService(index),
            context_builder=ContextBuilder(),
            provider=provider,
            settings=settings,
        )

        response = asyncio.run(
            orchestrator.run(
                AgentRequest(
                    query="What is the Master Map of Content?",
                    retrieval=RetrievalOptions(
                        limit=10,
                        expand_graph=False,
                    ),
                )
            )
        )

        counts_after = read_database_counts(database_path)

    # Answer propagation
    assert response.answer == (
        "The retrieved vault notes contain information "
        "about the Master Map of Content."
    )

    # Source propagation
    assert response.sources

    assert any(
        source.title == "Master Map of Content"
        for source in response.sources
    )

    assert all(
        source.source == "fts"
        for source in response.sources
    )

    # Provider metadata
    assert response.provider == "live-vault-test"
    assert response.model == "live-vault-test-model"
    assert response.usage == provider.usage

    # Provider received the actual retrieved context
    assert len(provider.requests) == 1

    llm_request = provider.requests[0]

    assert len(llm_request.messages) == 2
    assert llm_request.messages[0].role == "system"
    assert llm_request.messages[1].role == "user"

    user_message = llm_request.messages[1].content

    assert "Master Map of Content" in user_message
    assert "Retrieved context from the Obsidian vault" in user_message

    # Context source consistency
    for source in response.sources:
        assert source.title in user_message
        assert source.path in user_message
        assert source.excerpt

    # Database content must remain unchanged
    assert counts_after == counts_before
