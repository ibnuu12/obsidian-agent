from __future__ import annotations

import asyncio

from agent_obsidian.application.indexing.vault_indexer import (
    DEFAULT_CONTENT_ROOTS,
    IndexingResult,
    VaultIndexer,
)
from agent_obsidian.infrastructure.mcp.obsidian_client import (
    ObsidianMCPClient,
)
from agent_obsidian.infrastructure.mcp.obsidian_vault_source import (
    ObsidianVaultSource,
)
from agent_obsidian.infrastructure.retrieval.sqlite_index import (
    SQLiteIndex,
)
from agent_obsidian.shared.config import load_settings


async def _run_indexing(
    root: str,
    settings,
) -> IndexingResult:
    async with ObsidianMCPClient(settings) as client:
        source = ObsidianVaultSource(client)

        with SQLiteIndex(settings.database_path) as index:
            indexer = VaultIndexer(
                source=source,
                index=index,
                content_roots=(root,),
            )

            return await indexer.index_vault()


def _present_indexing_result(
    root: str,
    result: IndexingResult,
) -> None:
    print()
    print("Agent Obsidian - Indexing")
    print("=" * 40)
    print(f"Vault root : {root}")
    print(f"Discovered : {result.discovered}")
    print(f"Indexed    : {result.indexed}")
    print(f"Skipped    : {result.skipped}")
    print(f"Failed     : {result.failed}")

    print()
    print("Relationship resolution")

    if result.relationship_resolution:
        for status, count in sorted(
            result.relationship_resolution.items()
        ):
            print(f"  {status}: {count}")
    else:
        print("  No relationship resolution performed.")

    if result.errors:
        print()
        print("Errors")

        for error in result.errors:
            print(f"  - {error}")


def execute_index(root: str) -> int:
    if root not in DEFAULT_CONTENT_ROOTS:
        print(f"Error: unsupported vault root: {root}")
        return 1

    try:
        settings = load_settings()

        result = asyncio.run(
            _run_indexing(
                root=root,
                settings=settings,
            )
        )

        _present_indexing_result(root, result)

        return 1 if result.failed else 0

    except Exception as exc:
        print(
            f"Indexing error: {type(exc).__name__}: {exc}"
        )
        return 1