
from pathlib import Path

from agent_obsidian.infrastructure.retrieval.note_parser import parse_note
from agent_obsidian.infrastructure.retrieval.sqlite_index import SQLiteIndex


def test_knowledge_graph_resolution():
    atomic_note = parse_note(
        "30 Notes/Atomic Knowledge.md",
        """---
title: Atomic Knowledge
type: permanent
aliases:
  - Atomic Note
related:
  - "[[Knowledge Graph]]"
---

This note links to [[Knowledge Graph|the graph]].
""",
    )

    graph_note = parse_note(
        "30 Notes/Knowledge Graph.md",
        """---
title: Knowledge Graph
type: permanent
---

Knowledge Graph represents connected notes.
""",
    )

    with SQLiteIndex(Path(":memory:")) as index:
        index.upsert_note(atomic_note)
        index.upsert_note(graph_note)

        result = index.rebuild_relationship_resolution()

        assert result["resolved"] == 2
        assert result["unresolved"] == 0
        assert result["ambiguous"] == 0

        outgoing = index.get_outgoing_links(
            "30 Notes/Atomic Knowledge.md"
        )

        assert len(outgoing) == 2
        assert all(
            item["resolution_status"] == "resolved"
            for item in outgoing
        )

        assert all(
            item["target_path"] == "30 Notes/Knowledge Graph.md"
            for item in outgoing
        )

        backlinks = index.get_backlinks(
            "30 Notes/Knowledge Graph.md"
        )

        assert len(backlinks) == 2
        assert all(
            item["source_path"] == "30 Notes/Atomic Knowledge.md"
            for item in backlinks
        )


def test_unresolved_relationship():
    note = parse_note(
        "30 Notes/Atomic Knowledge.md",
        """---
title: Atomic Knowledge
related:
  - "[[Missing Note]]"
---
Content.
""",
    )

    with SQLiteIndex(Path(":memory:")) as index:
        index.upsert_note(note)

        result = index.rebuild_relationship_resolution()

        assert result["unresolved"] == 1

        unresolved = index.get_unresolved_links(
            status="unresolved"
        )

        assert len(unresolved) == 1
        assert unresolved[0]["target"] == "Missing Note"


def test_ambiguous_relationship():
    source = parse_note(
        "30 Notes/Source.md",
        """---
title: Source
related:
  - "[[Shared Alias]]"
---
Content.
""",
    )

    first = parse_note(
        "30 Notes/First.md",
        """---
title: First
aliases:
  - Shared Alias
---
Content.
""",
    )

    second = parse_note(
        "30 Notes/Second.md",
        """---
title: Second
aliases:
  - Shared Alias
---
Content.
""",
    )

    with SQLiteIndex(Path(":memory:")) as index:
        index.upsert_note(source)
        index.upsert_note(first)
        index.upsert_note(second)

        result = index.rebuild_relationship_resolution()

        assert result["ambiguous"] == 1

        ambiguous = index.get_unresolved_links(
            status="ambiguous"
        )

        assert len(ambiguous) == 1
        assert ambiguous[0]["target"] == "Shared Alias"
