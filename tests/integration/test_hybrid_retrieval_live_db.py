
from __future__ import annotations

import pytest

from agent_obsidian.application.hybrid_retrieval import (
    HybridRetrievalService,
)
from agent_obsidian.infrastructure.retrieval.sqlite_index import (
    SQLiteIndex,
)
from agent_obsidian.shared.config import load_settings


@pytest.fixture
def live_index():
    """Open the configured project database for integration testing."""

    settings = load_settings()
    database_path = settings.database_path

    if not database_path.is_file():
        pytest.skip(
            f"Project database not found: {database_path}"
        )

    with SQLiteIndex(database_path) as index:
        yield index


def test_live_database_fts_retrieval(live_index):
    """Verify FTS retrieval against the actual indexed vault."""

    service = HybridRetrievalService(live_index)

    results = service.retrieve(
        query="Master Map of Content",
        limit=10,
        expand_graph=False,
    )

    assert results, "No results returned from the live database."

    assert all(
        item["source"] == "fts"
        for item in results
    )

    assert any(
        item["title"] == "Master Map of Content"
        for item in results
    )

    for item in results:
        assert item["path"]
        assert item["title"]
        assert isinstance(item["tags"], list)
        assert isinstance(item["aliases"], list)
        assert isinstance(item["frontmatter"], dict)
        assert isinstance(item["content"], str)


def test_live_database_graph_expansion(live_index):
    """Verify hybrid retrieval and graph metadata against live data."""

    service = HybridRetrievalService(live_index)

    results = service.retrieve(
        query="Master Map of Content",
        limit=10,
        expand_graph=True,
        graph_limit=20,
    )

    assert results, "No results returned from hybrid retrieval."

    direct_results = [
        item for item in results
        if item["source"] == "fts"
    ]

    graph_results = [
        item for item in results
        if item["source"] == "graph"
    ]

    assert direct_results, "No direct FTS results found."

    paths = [item["path"] for item in results]

    assert len(paths) == len(set(paths)), (
        "Hybrid retrieval returned duplicate note paths."
    )

    for item in direct_results:
        relationships = item["relationships"]

        assert isinstance(relationships, dict)
        assert isinstance(relationships["outgoing"], list)
        assert isinstance(relationships["backlinks"], list)

    for item in graph_results:
        assert item["path"]
        assert item["title"]
        assert item["source"] == "graph"
