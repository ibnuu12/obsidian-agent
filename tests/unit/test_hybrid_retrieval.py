
from agent_obsidian.infrastructure.retrieval.note_parser import parse_note
from agent_obsidian.infrastructure.retrieval.sqlite_index import SQLiteIndex
from agent_obsidian.application.hybrid_retrieval import HybridRetrievalService


def _add_note(index, path, markdown):
    note = parse_note(path, markdown)
    index.upsert_note(note)


def test_hybrid_retrieval_with_graph_expansion(tmp_path):
    with SQLiteIndex(tmp_path / "test.db") as index:
        _add_note(
            index,
            "30 Notes/Atomic Knowledge.md",
            """---
title: Atomic Knowledge
type: permanent
status: evergreen
tags:
  - zettel/permanent
  - knowledge
---
# Atomic Knowledge

Atomic knowledge is a fundamental concept.

Related to [[Knowledge Graph]].
""",
        )

        _add_note(
            index,
            "30 Notes/Knowledge Graph.md",
            """---
title: Knowledge Graph
type: permanent
status: evergreen
tags:
  - zettel/permanent
  - knowledge
---
# Knowledge Graph

A graph of connected knowledge notes.
""",
        )

        index.rebuild_relationship_resolution()

        service = HybridRetrievalService(index)

        results = service.retrieve(
            query="atomic knowledge",
            limit=5,
            expand_graph=True,
        )

        assert len(results) == 2

        direct = next(
            item for item in results
            if item["path"] == "30 Notes/Atomic Knowledge.md"
        )

        graph = next(
            item for item in results
            if item["path"] == "30 Notes/Knowledge Graph.md"
        )

        assert direct["source"] == "fts"
        assert graph["source"] == "graph"

        assert direct["title"] == "Atomic Knowledge"
        assert graph["title"] == "Knowledge Graph"

        assert direct["relationships"]["outgoing"]


def test_hybrid_retrieval_metadata_filter(tmp_path):
    with SQLiteIndex(tmp_path / "test.db") as index:
        _add_note(
            index,
            "30 Notes/Permanent Note.md",
            """---
title: Permanent Note
type: permanent
status: evergreen
tags:
  - zettel/permanent
---
# Knowledge Architecture

Knowledge architecture and permanent notes.
""",
        )

        _add_note(
            index,
            "30 Notes/Literature Note.md",
            """---
title: Literature Note
type: literature
status: reading
tags:
  - zettel/literature
---
# Knowledge Literature

Knowledge architecture from a literature source.
""",
        )

        service = HybridRetrievalService(index)

        results = service.retrieve(
            query="knowledge architecture",
            note_types={"permanent"},
            statuses={"evergreen"},
            tags={"zettel/permanent"},
            expand_graph=False,
        )

        assert len(results) == 1
        assert results[0]["path"] == "30 Notes/Permanent Note.md"


def test_hybrid_retrieval_backlink_expansion(tmp_path):
    with SQLiteIndex(tmp_path / "test.db") as index:
        _add_note(
            index,
            "30 Notes/Atomic Knowledge.md",
            """---
title: Atomic Knowledge
type: permanent
status: evergreen
tags:
  - knowledge
---
# Atomic Knowledge

Atomic knowledge is the central concept.
""",
        )

        _add_note(
            index,
            "30 Notes/Knowledge Graph.md",
            """---
title: Knowledge Graph
type: permanent
status: evergreen
tags:
  - knowledge
---
# Knowledge Graph

This note links to [[Atomic Knowledge]].
""",
        )

        index.rebuild_relationship_resolution()

        service = HybridRetrievalService(index)

        results = service.retrieve(
            query="central concept",
            expand_graph=True,
        )

        graph_paths = {
            item["path"]
            for item in results
            if item["source"] == "graph"
        }

        assert "30 Notes/Knowledge Graph.md" in graph_paths


def test_hybrid_retrieval_without_graph_expansion(tmp_path):
    with SQLiteIndex(tmp_path / "test.db") as index:
        _add_note(
            index,
            "30 Notes/Atomic Knowledge.md",
            """---
title: Atomic Knowledge
type: permanent
status: evergreen
---
# Atomic Knowledge

Atomic knowledge.
""",
        )

        _add_note(
            index,
            "30 Notes/Knowledge Graph.md",
            """---
title: Knowledge Graph
type: permanent
status: evergreen
---
# Knowledge Graph

A graph of knowledge.
""",
        )

        index.rebuild_relationship_resolution()

        service = HybridRetrievalService(index)

        results = service.retrieve(
            query="atomic knowledge",
            expand_graph=False,
        )

        assert all(item["source"] == "fts" for item in results)
        assert len(results) == 1

def test_graph_expansion_with_metadata_filter(tmp_path):
    with SQLiteIndex(tmp_path / "test.db") as index:
        _add_note(
            index,
            "30 Notes/Atomic Knowledge.md",
            """---
title: Atomic Knowledge
type: permanent
status: evergreen
tags:
  - knowledge
---
# Atomic Knowledge

This is the central concept.

Related to [[Knowledge Graph]].
""",
        )

        _add_note(
            index,
            "30 Notes/Knowledge Graph.md",
            """---
title: Knowledge Graph
type: permanent
status: evergreen
tags:
  - knowledge
---
# Knowledge Graph

A graph of connected notes.
""",
        )

        index.rebuild_relationship_resolution()

        service = HybridRetrievalService(index)

        results = service.retrieve(
            query="central concept",
            note_types={"permanent"},
            statuses={"evergreen"},
            tags={"knowledge"},
            expand_graph=True,
        )

        direct_paths = {
            item["path"]
            for item in results
            if item["source"] == "fts"
        }

        graph_paths = {
            item["path"]
            for item in results
            if item["source"] == "graph"
        }

        assert "30 Notes/Atomic Knowledge.md" in direct_paths
        assert "30 Notes/Knowledge Graph.md" in graph_paths

def test_graph_limit(tmp_path):
    with SQLiteIndex(tmp_path / "test.db") as index:
        _add_note(
            index,
            "30 Notes/Source.md",
            """---
title: Source
type: permanent
status: evergreen
---
# Source

This is a unique anchor concept.

Related to [[Note A]].
Related to [[Note B]].
Related to [[Note C]].
""",
        )

        for name in ["Note A", "Note B", "Note C"]:
            _add_note(
                index,
                f"30 Notes/{name}.md",
                f"""---
title: {name}
type: permanent
status: evergreen
---
# {name}

A related knowledge note.
""",
            )

        index.rebuild_relationship_resolution()

        service = HybridRetrievalService(index)

        results = service.retrieve(
            query="unique anchor concept",
            limit=5,
            expand_graph=True,
            graph_limit=2,
        )

        graph_results = [
            item for item in results
            if item["source"] == "graph"
        ]

        assert len(graph_results) == 2

def test_no_duplicate_notes(tmp_path):
    with SQLiteIndex(tmp_path / "test.db") as index:
        _add_note(
            index,
            "30 Notes/Source.md",
            """---
title: Source
type: permanent
status: evergreen
---
# Source

Unique anchor concept.

Related to [[Knowledge Graph]].
""",
        )

        _add_note(
            index,
            "30 Notes/Knowledge Graph.md",
            """---
title: Knowledge Graph
type: permanent
status: evergreen
---
# Knowledge Graph

Unique anchor concept and connected knowledge.
""",
        )

        index.rebuild_relationship_resolution()

        service = HybridRetrievalService(index)

        results = service.retrieve(
            query="unique anchor concept",
            limit=5,
            expand_graph=True,
        )

        paths = [item["path"] for item in results]

        assert len(paths) == len(set(paths))

def test_direct_results_limit(tmp_path):
    with SQLiteIndex(tmp_path / "test.db") as index:
        for name in ["Note A", "Note B", "Note C"]:
            _add_note(
                index,
                f"30 Notes/{name}.md",
                f"""---
title: {name}
type: permanent
status: evergreen
---
# {name}

Shared searchable knowledge concept.
""",
            )

        service = HybridRetrievalService(index)

        results = service.retrieve(
            query="shared searchable knowledge",
            limit=2,
            expand_graph=False,
        )

        assert len(results) == 2
        assert all(item["source"] == "fts" for item in results)