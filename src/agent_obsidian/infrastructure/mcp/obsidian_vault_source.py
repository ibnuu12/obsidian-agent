
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from agent_obsidian.infrastructure.mcp.obsidian_client import (
    ObsidianMCPClient,
)


class ObsidianVaultSource:
    """
    Adapts raw Obsidian MCP responses into normalized Python data.
    """

    def __init__(self, client: ObsidianMCPClient):
        self.client = client

    @staticmethod
    def _decode_result(result: Any, operation: str) -> dict[str, Any]:
        if getattr(result, "is_error", False):
            raise RuntimeError(
                f"Obsidian MCP operation failed: {operation}"
            )

        structured = getattr(result, "structured_content", None)

        if isinstance(structured, Mapping):
            return dict(structured)

        content_items = getattr(result, "content", [])

        for item in content_items:
            text = getattr(item, "text", None)

            if not isinstance(text, str):
                continue

            try:
                decoded = json.loads(text)
            except json.JSONDecodeError:
                continue

            if isinstance(decoded, Mapping):
                return dict(decoded)

        raise ValueError(
            f"Invalid MCP response for {operation}: "
            "expected a JSON object"
        )

    async def list_vault(self, path: str = "") -> list[str]:
        result = await self.client.list_vault(path)

        payload = self._decode_result(
            result,
            "vault_list",
        )

        files = payload.get("files")

        if not isinstance(files, list):
            raise ValueError(
                "Invalid vault_list response: "
                "'files' must be a list"
            )

        if not all(
            isinstance(item, str) and item.strip()
            for item in files
        ):
            raise ValueError(
                "Invalid vault_list response: "
                "every entry must be a non-empty string"
            )

        return files

    async def read_note(self, path: str) -> dict[str, Any]:
        result = await self.client.read_note(path)

        payload = self._decode_result(
            result,
            "vault_read",
        )

        note_path = payload.get("path")
        content = payload.get("content")
        stat = payload.get("stat", {})

        if not isinstance(note_path, str) or not note_path.strip():
            raise ValueError(
                "Invalid vault_read response: "
                "'path' must be a non-empty string"
            )

        if not isinstance(content, str):
            raise ValueError(
                "Invalid vault_read response: "
                "'content' must be a string"
            )

        if not isinstance(stat, Mapping):
            raise ValueError(
                "Invalid vault_read response: "
                "'stat' must be an object"
            )

        return {
            "path": note_path,
            "content": content,
            "stat": dict(stat),
        }
