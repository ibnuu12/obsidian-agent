
import pytest

from agent_obsidian.infrastructure.retrieval.note_parser import (
    _normalize_link_target,
)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("Knowledge Graph", "Knowledge Graph"),
        ("[[Knowledge Graph]]", "Knowledge Graph"),
        ("[[Knowledge Graph|Graph]]", "Knowledge Graph"),
        ("[[Knowledge Graph#Overview]]", "Knowledge Graph"),
        (
            "[[Knowledge Graph#Overview|Graph]]",
            "Knowledge Graph",
        ),
        (
            "![[Note Attachment#Summary]]",
            "Note Attachment",
        ),
        (
            r"[[Folder\Knowledge Graph]]",
            "Folder/Knowledge Graph",
        ),
    ],
)
def test_normalize_link_target(source, expected):
    assert _normalize_link_target(source) == expected
