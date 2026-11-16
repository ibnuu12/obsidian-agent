
import pytest

from agent_obsidian.infrastructure.retrieval.note_parser import (
    parse_note,
)
from agent_obsidian.infrastructure.retrieval.sqlite_index import (
    SQLiteIndex,
)


@pytest.mark.parametrize(
    "query",
    [
        "What is a knowledge graph?",
        "Apa itu knowledge graph?",
    ],
)
def test_search_handles_natural_language_queries(tmp_path, query):
    note = parse_note(
        "30 Notes/Knowledge Graph.md",
        """---
title: Knowledge Graph
type: permanent
status: evergreen
tags:
  - knowledge
  - graph
---

A knowledge graph represents entities and relationships
between connected knowledge notes.
""",
    )

    with SQLiteIndex(tmp_path / "test.db") as index:
        index.upsert_note(note)

        results = index.search(query)

    assert len(results) == 1
    assert results[0]["path"] == "30 Notes/Knowledge Graph.md"
