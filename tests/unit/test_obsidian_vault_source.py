
import asyncio
import json
from types import SimpleNamespace

import pytest

from agent_obsidian.infrastructure.mcp.obsidian_vault_source import (
    ObsidianVaultSource,
)


class FakeMCPClient:
    def __init__(self, listing_result=None, note_result=None):
        self.listing_result = listing_result
        self.note_result = note_result

    async def list_vault(self, path=""):
        return self.listing_result

    async def read_note(self, path):
        return self.note_result


def make_result(payload, structured_content=None, is_error=False):
    return SimpleNamespace(
        structured_content=structured_content,
        content=[
            SimpleNamespace(
                text=json.dumps(payload, ensure_ascii=False)
            )
        ] if payload is not None else [],
        is_error=is_error,
    )


def test_list_vault_decodes_text_content():
    result = make_result({
        "files": [
            "20 MOCs/",
            "30 Notes/",
            "Master Map of Content.md",
        ]
    })

    source = ObsidianVaultSource(
        FakeMCPClient(listing_result=result)
    )

    files = asyncio.run(source.list_vault("K003_Personal"))

    assert files == [
        "20 MOCs/",
        "30 Notes/",
        "Master Map of Content.md",
    ]


def test_list_vault_prefers_structured_content():
    result = make_result(
        payload={"files": ["wrong.md"]},
        structured_content={"files": ["correct.md"]},
    )

    source = ObsidianVaultSource(
        FakeMCPClient(listing_result=result)
    )

    files = asyncio.run(source.list_vault())

    assert files == ["correct.md"]


def test_read_note_decodes_text_content():
    payload = {
        "path": "K003_Personal/20 MOCs/Master Map of Content.md",
        "content": "---\ntitle: Master Map of Content\n---\n# Overview",
        "stat": {
            "mtime": 1794295719257,
            "size": 2240,
        },
    }

    source = ObsidianVaultSource(
        FakeMCPClient(note_result=make_result(payload))
    )

    note = asyncio.run(
        source.read_note(
            "K003_Personal/20 MOCs/Master Map of Content.md"
        )
    )

    assert note["path"] == payload["path"]
    assert note["content"] == payload["content"]
    assert note["stat"]["size"] == 2240


def test_list_vault_rejects_invalid_json():
    result = SimpleNamespace(
        structured_content=None,
        content=[SimpleNamespace(text="not valid json")],
        is_error=False,
    )

    source = ObsidianVaultSource(
        FakeMCPClient(listing_result=result)
    )

    with pytest.raises(ValueError, match="Invalid MCP response"):
        asyncio.run(source.list_vault())


def test_list_vault_rejects_missing_files():
    result = make_result({"items": ["note.md"]})

    source = ObsidianVaultSource(
        FakeMCPClient(listing_result=result)
    )

    with pytest.raises(ValueError, match="'files' must be a list"):
        asyncio.run(source.list_vault())


def test_read_note_rejects_missing_content():
    result = make_result({
        "path": "K003_Personal/20 MOCs/Example.md",
        "stat": {},
    })

    source = ObsidianVaultSource(
        FakeMCPClient(note_result=result)
    )

    with pytest.raises(ValueError, match="'content' must be a string"):
        asyncio.run(
            source.read_note("K003_Personal/20 MOCs/Example.md")
        )


def test_list_vault_rejects_mcp_error():
    result = make_result(
        payload=None,
        is_error=True,
    )

    source = ObsidianVaultSource(
        FakeMCPClient(listing_result=result)
    )

    with pytest.raises(RuntimeError, match="MCP operation failed"):
        asyncio.run(source.list_vault())
