
import asyncio

import pytest

from agent_obsidian.application.indexing.vault_indexer import (
    VaultIndexer,
)
from agent_obsidian.infrastructure.retrieval.sqlite_index import (
    SQLiteIndex,
)


class FakeVaultSource:
    def __init__(self, directories=None, documents=None, failures=None):
        self.directories = directories or {}
        self.documents = documents or {}
        self.failures = failures or set()
        self.listed_paths = []
        self.read_paths = []

    async def list_vault(self, path=""):
        self.listed_paths.append(path)

        if path not in self.directories:
            raise FileNotFoundError(path)

        return self.directories[path]

    async def read_note(self, path):
        self.read_paths.append(path)

        if path in self.failures:
            raise RuntimeError("Simulated read failure")

        if path not in self.documents:
            raise FileNotFoundError(path)

        return self.documents[path]


def make_document(path, content, mtime=1794295719257, size=None):
    return {
        "path": path,
        "content": content,
        "stat": {
            "mtime": mtime,
            "size": size if size is not None else len(content),
        },
    }



def test_index_vault_recursively_indexes_markdown_and_resolves_links(
    tmp_path,
):
    alpha_path = "K003_Personal/30 Notes/Alpha.md"
    beta_path = "K003_Personal/30 Notes/Beta.md"

    root = "K003_Personal"

    source = FakeVaultSource(
        directories={
            root: ["30 Notes/"],
            f"{root}/30 Notes": [
                "Alpha.md",
                "Beta.md",
            ],
        },
        documents={
            alpha_path: make_document(
                alpha_path,
                "---\ntitle: Alpha\ntype: note\n---\nLinks to [[Beta]].",
            ),
            beta_path: make_document(
                beta_path,
                "---\ntitle: Beta\ntype: note\n---\nBeta content.",
            ),
        },
    )

    with SQLiteIndex(tmp_path / "agent.db") as index:
        indexer = VaultIndexer(
            source,
            index,
            content_roots=(root,),
        )

        result = asyncio.run(indexer.index_vault())

        assert result.discovered == 2
        assert result.indexed == 2
        assert result.failed == 0
        assert index.count_notes() == 2

        outgoing = index.get_outgoing_links(alpha_path)

        assert len(outgoing) == 1
        assert outgoing[0]["target"] == "Beta"
        assert outgoing[0]["target_path"] == beta_path
        assert outgoing[0]["resolution_status"] == "resolved"

def test_index_vault_excludes_directories_and_non_markdown_files(
    tmp_path,
):
    root = "K001_Finance"

    metadata_path = f"{root}/00 Meta/metadata.md"
    note_path = f"{root}/30 Notes/note.md"

    source = FakeVaultSource(
        directories={
            root: [
                ".obsidian/",
                "00 Meta/",
                "30 Notes/",
            ],
            f"{root}/00 Meta": [
                "Templates/",
                "Attachments/",
                "metadata.md",
            ],
            f"{root}/30 Notes": [
                "note.md",
                "image.png",
                "document.pdf",
            ],
        },
        documents={
            metadata_path: make_document(
                metadata_path,
                "# Metadata note",
            ),
            note_path: make_document(
                note_path,
                "# Regular note",
            ),
        },
    )

    with SQLiteIndex(tmp_path / "agent.db") as index:
        indexer = VaultIndexer(
            source,
            index,
            content_roots=(root,),
        )

        result = asyncio.run(indexer.index_vault())

        assert result.discovered == 2
        assert result.indexed == 2
        assert result.failed == 0
        assert result.skipped == 5
        assert index.count_notes() == 2

        assert f"{root}/.obsidian" not in source.listed_paths
        assert f"{root}/00 Meta/Templates" not in source.listed_paths
        assert f"{root}/00 Meta/Attachments" not in source.listed_paths

def test_index_vault_is_idempotent(tmp_path):
    path = "K002_Tech/30 Notes/Python.md"

    source = FakeVaultSource(
        directories={
            "K002_Tech": ["30 Notes/"],
            "K002_Tech/30 Notes": ["Python.md"],
        },
        documents={
            path: make_document(path, "# Python"),
        },
    )

    with SQLiteIndex(tmp_path / "agent.db") as index:
        indexer = VaultIndexer(
            source,
            index,
            content_roots=("K002_Tech",),
        )

        first_result = asyncio.run(indexer.index_vault())
        second_result = asyncio.run(indexer.index_vault())

        assert first_result.indexed == 1
        assert second_result.indexed == 1
        assert index.count_notes() == 1


def test_index_vault_continues_after_note_read_failure(tmp_path):
    good_path = "K003_Personal/30 Notes/Good.md"
    bad_path = "K003_Personal/30 Notes/Bad.md"

    source = FakeVaultSource(
        directories={
            "K003_Personal": ["30 Notes/"],
            "K003_Personal/30 Notes": [
                "Bad.md",
                "Good.md",
            ],
        },
        documents={
            good_path: make_document(good_path, "# Good note"),
        },
        failures={bad_path},
    )

    with SQLiteIndex(tmp_path / "agent.db") as index:
        indexer = VaultIndexer(
            source,
            index,
            content_roots=("K003_Personal",),
        )

        result = asyncio.run(indexer.index_vault())

        assert result.discovered == 2
        assert result.indexed == 1
        assert result.failed == 1
        assert index.count_notes() == 1
        assert any(bad_path in error for error in result.errors)


def test_index_vault_continues_with_next_root_after_listing_failure(
    tmp_path,
):
    good_path = "K002_Tech/30 Notes/Good.md"

    source = FakeVaultSource(
        directories={
            "K002_Tech": ["30 Notes/"],
            "K002_Tech/30 Notes": ["Good.md"],
        },
        documents={
            good_path: make_document(good_path, "# Good note"),
        },
    )

    with SQLiteIndex(tmp_path / "agent.db") as index:
        indexer = VaultIndexer(
            source,
            index,
            content_roots=(
                "K001_Finance",
                "K002_Tech",
            ),
        )

        result = asyncio.run(indexer.index_vault())

        assert result.indexed == 1
        assert result.failed == 1
        assert index.count_notes() == 1
        assert "K002_Tech" in source.listed_paths


def test_index_vault_rejects_invalid_entry(tmp_path):
    source = FakeVaultSource(
        directories={
            "K003_Personal": [
                "../outside.md",
                "valid.md",
            ],
        },
        documents={
            "K003_Personal/valid.md": make_document(
                "K003_Personal/valid.md",
                "# Valid note",
            ),
        },
    )

    with SQLiteIndex(tmp_path / "agent.db") as index:
        indexer = VaultIndexer(
            source,
            index,
            content_roots=("K003_Personal",),
        )

        result = asyncio.run(indexer.index_vault())

        assert result.indexed == 1
        assert result.failed == 1
        assert index.count_notes() == 1
        assert any("Invalid vault entry" in error for error in result.errors)


def test_indexer_requires_at_least_one_content_root():
    source = FakeVaultSource()

    with pytest.raises(ValueError, match="At least one content root"):
        VaultIndexer(
            source,
            index=None,
            content_roots=(),
        )
