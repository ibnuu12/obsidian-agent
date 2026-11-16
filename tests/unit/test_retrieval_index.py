
from pathlib import Path

from agent_obsidian.infrastructure.retrieval.note_parser import parse_note
from agent_obsidian.infrastructure.retrieval.sqlite_index import SQLiteIndex


SAMPLE_NOTE = """---
id: 20261111100001
title: Atomic Knowledge
aliases:
  - Atomic Note
type: permanent
status: evergreen
tags:
  - zettel/permanent
  - knowledge
created: "2026-11-11 10:00"
updated: "2026-11-11 10:00"
related:
  - "[[Knowledge Graph]]"
author: Test Author
---

# Atomic Knowledge

One note represents one idea.

A permanent note should connect to [[Knowledge Graph|the graph]].
It may also reference ![[Note Attachment#Summary]].
"""


def main():
    note = parse_note(
        "30 Notes/Atomic Knowledge.md",
        SAMPLE_NOTE,
    )

    assert note.title == "Atomic Knowledge"
    assert note.note_id == "20261111100001"
    assert note.note_type == "permanent"
    assert note.status == "evergreen"
    assert "Atomic Note" in note.aliases
    assert "zettel/permanent" in note.tags
    assert note.frontmatter["author"] == "Test Author"

    assert "Knowledge Graph" in note.wikilinks
    assert "Note Attachment" in note.wikilinks


    print("\n=== DEBUG NOTE RELATED ===")
    print("Type   :", type(note.related))
    print("Value  :", repr(note.related))
    print("Note   :", note.path)

    assert "Knowledge Graph" in note.related

    with SQLiteIndex(Path(":memory:")) as index:
        index.upsert_note(note, mtime=1000, size=500)

        assert index.count_notes() == 1

        result = index.search("atomic knowledge")

        assert len(result) == 1
        assert result[0]["path"] == "30 Notes/Atomic Knowledge.md"

        # Verify upsert updates rather than duplicates the note.
        updated_note = parse_note(
            "30 Notes/Atomic Knowledge.md",
            SAMPLE_NOTE.replace(
                "One note represents one idea.",
                "One atomic note represents one clear idea.",
            ),
        )

        index.upsert_note(updated_note, mtime=2000, size=550)

        assert index.count_notes() == 1

        stored = index.get_note("30 Notes/Atomic Knowledge.md")
        assert stored is not None
        assert stored["mtime"] == 2000
        assert stored["frontmatter"]["author"] == "Test Author"

        updated_result = index.search("atomic clear idea")
        assert len(updated_result) == 1

    print("Note parser: OK")
    print("Frontmatter preservation: OK")
    print("Wikilink extraction: OK")
    print("Related extraction: OK")
    print("SQLite FTS5 indexing: OK")
    print("Upsert without duplication: OK")
    print("Retrieval search: OK")
    print("\nStage 4A/4B: PASSED")

if __name__ == "__main__":
    main()