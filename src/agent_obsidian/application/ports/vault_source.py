
from __future__ import annotations

from typing import Any, Protocol


class VaultSource(Protocol):
    """
    Application-level contract for reading an Obsidian vault.

    Implementations may use MCP, REST API, or another
    read-only source.
    """

    async def list_vault(
        self,
        path: str = "",
    ) -> list[str]:
        """
        Return immediate child names inside a vault directory.

        Directory names must end with '/'.
        File names must not end with '/'.
        """
        ...

    async def read_note(
        self,
        path: str,
    ) -> dict[str, Any]:
        """
        Return a normalized note document.

        Expected fields:
        - path: vault-relative note path
        - content: Markdown content
        - stat: optional file metadata
        """
        ...
