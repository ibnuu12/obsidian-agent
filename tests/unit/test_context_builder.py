
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
