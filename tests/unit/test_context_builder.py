
import pytest

from agent_obsidian.application.context_builder import (
    ContextBuilder,
    estimate_tokens,
)


def _note(
    title="Test Note",
    path="30 Notes/Test Note.md",
    source="fts",
    excerpt="This is a test excerpt.",
    **kwargs,
):
    return {
        "title": title,
        "path": path,
        "type": "permanent",
        "status": "evergreen",
        "tags": ["zettel/permanent"],
        "aliases": [],
        "source": source,
        "score": -1.0,
        "excerpt": excerpt,
        "content": "FULL CONTENT MUST NOT BE USED",
        **kwargs,
    }


def test_build_context_from_single_note():
    builder = ContextBuilder()

    result = builder.build(
        notes=[_note()],
        max_tokens=500,
    )

    assert len(result.items) == 1
    assert "Test Note" in result.context
    assert "This is a test excerpt." in result.context
    assert result.items[0].excerpt == "This is a test excerpt."
    assert result.excluded_count == 0
    assert result.truncated_count == 0


def test_build_context_from_multiple_notes():
    builder = ContextBuilder()

    result = builder.build(
        notes=[
            _note(title="Note A", path="Note A.md"),
            _note(title="Note B", path="Note B.md"),
        ],
        max_tokens=500,
    )

    assert len(result.items) == 2
    assert "Note A" in result.context
    assert "Note B" in result.context


def test_excerpt_first_never_uses_full_content():
    builder = ContextBuilder()

    result = builder.build(
        notes=[_note()],
        max_tokens=500,
    )

    assert "FULL CONTENT MUST NOT BE USED" not in result.context


def test_missing_excerpt_is_excluded_without_content_fallback():
    builder = ContextBuilder()

    result = builder.build(
        notes=[
            _note(excerpt=None),
        ],
        max_tokens=500,
    )

    assert result.context == ""
    assert result.items == ()
    assert result.excluded_count == 1


def test_moc_uses_full_content_instead_of_excerpt():
    builder = ContextBuilder()

    full_content = """
# Master Map of Content

## Core Purpose
Central dashboard for knowledge navigation.

## Core Domain Navigation
- [[Software Engineering]]
- [[System Architecture]]
- [[AI Development]]
- [[DevOps & Infra]]

## Cross-Vault Navigator
- [[Master Map Finance]]
- [[Master Map Personal]]
"""

    result = builder.build(
        notes=[
            _note(
                title="Master Map Tech",
                path="20 MOCs/Master Map Tech.md",
                excerpt="Short FTS excerpt.",
                content=full_content,
                type="moc",
            ),
        ],
        max_tokens=1000,
    )

    assert len(result.items) == 1
    assert "Central dashboard" in result.context
    assert "Software Engineering" in result.context
    assert "Cross-Vault Navigator" in result.context
    assert "Short FTS excerpt." not in result.context
    assert result.items[0].excerpt.strip() == full_content.strip()


def test_moc_uses_excerpt_when_full_content_is_missing():
    builder = ContextBuilder()

    result = builder.build(
        notes=[
            _note(
                excerpt="MOC fallback excerpt.",
                content=None,
                type="moc",
            ),
        ],
        max_tokens=500,
    )

    assert len(result.items) == 1
    assert "MOC fallback excerpt." in result.context


def test_moc_full_content_is_truncated_when_budget_is_limited():
    builder = ContextBuilder()

    full_content = "# Master MOC\n\n" + (
        "Knowledge domain navigation. " * 500
    )

    result = builder.build(
        notes=[
            _note(
                title="Master MOC",
                path="20 MOCs/Master MOC.md",
                excerpt="Short excerpt.",
                content=full_content,
                type="moc",
            ),
            _note(
                title="Lower Priority Note",
                path="30 Notes/Other.md",
                excerpt="Additional context.",
            ),
        ],
        max_tokens=100,
    )

    assert len(result.items) == 1
    assert result.items[0].title == "Master MOC"
    assert result.truncated_count == 1
    assert result.excluded_count == 1
    assert result.estimated_tokens <= result.budget_tokens
    assert result.items[0].excerpt.endswith("…")


def test_regular_note_still_uses_excerpt_when_content_exists():
    builder = ContextBuilder()

    result = builder.build(
        notes=[
            _note(
                excerpt="Relevant excerpt.",
                content="FULL CONTENT MUST NOT BE USED",
                type="permanent",
            ),
        ],
        max_tokens=500,
    )

    assert "Relevant excerpt." in result.context
    assert "FULL CONTENT MUST NOT BE USED" not in result.context

def test_fts_results_are_prioritized_over_graph_results():
    builder = ContextBuilder()

    result = builder.build(
        notes=[
            _note(
                title="Graph Note",
                path="Graph Note.md",
                source="graph",
                excerpt="Graph context.",
            ),
            _note(
                title="Direct Note",
                path="Direct Note.md",
                source="fts",
                excerpt="Direct search result.",
            ),
        ],
        max_tokens=500,
    )

    assert result.items[0].title == "Direct Note"
    assert result.items[1].title == "Graph Note"


def test_metadata_is_preserved():
    builder = ContextBuilder()

    result = builder.build(
        notes=[
            _note(
                aliases=["Test Alias"],
                tags=["zettel/permanent", "knowledge"],
            ),
        ],
        max_tokens=500,
    )

    item = result.items[0]

    assert item.path == "30 Notes/Test Note.md"
    assert item.title == "Test Note"
    assert item.note_type == "permanent"
    assert item.status == "evergreen"
    assert item.tags == ("zettel/permanent", "knowledge")

    assert "Type: permanent" in result.context
    assert "Status: evergreen" in result.context
    assert "Aliases: Test Alias" in result.context


def test_excerpt_is_truncated_when_budget_is_limited():
    builder = ContextBuilder()

    long_excerpt = "knowledge " * 500

    result = builder.build(
        notes=[
            _note(excerpt=long_excerpt),
            _note(
                title="Lower Priority Note",
                path="Lower Priority Note.md",
                excerpt="Additional context.",
            ),
        ],
        max_tokens=80,
    )

    assert len(result.items) == 1
    assert result.truncated_count == 1
    assert result.excluded_count == 1
    assert len(result.items[0].excerpt) < len(long_excerpt)
    assert result.estimated_tokens <= result.budget_tokens
    assert result.items[0].excerpt.endswith("…")


def test_empty_retrieval_returns_empty_context():
    builder = ContextBuilder()

    result = builder.build(
        notes=[],
        max_tokens=100,
    )

    assert result.context == ""
    assert result.items == ()
    assert result.estimated_tokens == 0
    assert result.excluded_count == 0


@pytest.mark.parametrize(
    "budget",
    [0, -1, True, 1.5, "100"],
)
def test_invalid_token_budget_raises_value_error(budget):
    builder = ContextBuilder()

    with pytest.raises(ValueError):
        builder.build([], max_tokens=budget)


def test_token_estimation_is_deterministic():
    text = "A deterministic token estimation test."

    assert estimate_tokens(text) == estimate_tokens(text)
    assert estimate_tokens("") == 0
    assert estimate_tokens("1234") == 1
    assert estimate_tokens("12345") == 2


def test_context_never_exceeds_token_budget():
    builder = ContextBuilder()
    budget = 100

    result = builder.build(
        notes=[
            _note(
                title=f"Note {index}",
                path=f"Note {index}.md",
                excerpt="Relevant knowledge. " * 100,
            )
            for index in range(5)
        ],
        max_tokens=budget,
    )

    assert result.estimated_tokens <= budget


def test_duplicate_paths_keep_higher_priority_fts_result():
    builder = ContextBuilder()

    result = builder.build(
        notes=[
            _note(
                title="Graph Version",
                path="20 MOCs/Master MOC.md",
                source="graph",
                excerpt="Graph excerpt.",
                type="moc",
                content="# Graph Version",
            ),
            _note(
                title="FTS Version",
                path="20 MOCs/Master MOC.md",
                source="fts",
                excerpt="FTS excerpt.",
                type="moc",
                content="# FTS Version",
            ),
        ],
        max_tokens=500,
    )

    assert len(result.items) == 1
    assert result.items[0].source == "fts"
    assert "FTS Version" in result.context
    assert "Graph Version" not in result.context


def test_moc_context_has_explicit_section_boundaries():
    builder = ContextBuilder()

    content = """# Master Map of Content

## Core Purpose
Central knowledge dashboard.

## Core Domain Navigation
- [[Software Engineering]]
- [[AI Development]]

## Cross-Vault Navigator
- [[Master Map Finance]]

## Active Indexing
TABLE type AS Type FROM #zettel/moc
"""

    result = builder.build(
        notes=[
            _note(
                title="Master Map Tech",
                path="20 MOCs/Master Map Tech.md",
                type="moc",
                content=content,
            ),
        ],
        max_tokens=1000,
    )

    assert "<MOC_NOTE>" in result.context
    assert "<SECTION: Core Purpose>" in result.context
    assert "<SECTION: Core Domain Navigation>" in result.context
    assert "<SECTION: Cross-Vault Navigator>" in result.context
    assert "<SECTION: Active Indexing>" in result.context
    assert "</MOC_NOTE>" in result.context


def test_moc_preserves_later_sections_when_context_is_limited():
    builder = ContextBuilder()

    content = """# Master Map of Content

## Core Purpose
""" + ("General dashboard description. " * 100) + """

## Core Domain Navigation
- [[Software Engineering]]
- [[System Architecture]]
- [[AI Development]]
- [[DevOps & Infra]]

## Cross-Vault Navigator
- [[Master Map Finance]]
- [[Master Map Personal]]

## Active Indexing
TABLE type AS Type, file.mtime AS "Last Modified"
FROM #zettel/moc
SORT file.name ASC
"""

    result = builder.build(
        notes=[
            _note(
                title="Master Map Tech",
                path="20 MOCs/Master Map Tech.md",
                type="moc",
                content=content,
            ),
        ],
        max_tokens=300,
    )

    assert len(result.items) == 1
    assert "Core Domain Navigation" in result.context
    assert "Software Engineering" in result.context
    assert "Active Indexing" in result.context
    assert result.estimated_tokens <= result.budget_tokens
    assert result.truncated_count == 1

def test_moc_preserves_nested_heading_hierarchy():
    builder = ContextBuilder()

    content = """# Master Map of Content

## Core Domain Overview

### Software Engineering

Software Engineering domain description.

### System Architecture

System Architecture domain description.

## Cross-Vault Navigator

- [[Master Map Finance]]
"""

    result = builder.build(
        notes=[
            _note(
                title="Master Map Tech",
                path="20 MOCs/Master Map Tech.md",
                type="moc",
                content=content,
            ),
        ],
        max_tokens=1000,
    )

    context = result.context

    assert "<SECTION: Core Domain Overview>" in context
    assert "### Software Engineering" in context
    assert "### System Architecture" in context

    assert "<SECTION: Software Engineering>" not in context
    assert "<SECTION: System Architecture>" not in context

    assert "Software Engineering domain description." in context
    assert "System Architecture domain description." in context

    assert "<SECTION: Cross-Vault Navigator>" in context