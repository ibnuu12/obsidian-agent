
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


@pytest.mark.parametrize(
    "query",
    [
        "Jelaskan fungsi Master Map of Content dalam vault Obsidian saya.",
        "Apa fungsi Master Map of Content?",
        "Explain the function of Master Map of Content.",
        "Master Map of Content",
    ],
)
def test_search_handles_contextual_questions(tmp_path, query):
    note = parse_note(
        "K002_Tech/20 MOCs/Master Map of Content.md",
        """---
title: Master Map of Content
type: moc
status: active
tags:
  - zettel/moc
  - zettel/master
---

The Master Map of Content provides an overview
of the knowledge structure and navigation links.
""",
    )

    with SQLiteIndex(tmp_path / "test.db") as index:
        index.upsert_note(note)
        results = index.search(query)

    assert len(results) == 1
    assert (
        results[0]["path"]
        == "K002_Tech/20 MOCs/Master Map of Content.md"
    )


def test_search_preserves_stopword_only_fallback(tmp_path):
    note = parse_note(
        "30 Notes/Knowledge Graph.md",
        """---
title: Knowledge Graph
---

Knowledge graph connects related notes.
""",
    )

    with SQLiteIndex(tmp_path / "test.db") as index:
        index.upsert_note(note)
        results = index.search("apa itu dan yang")

    assert len(results) == 0
