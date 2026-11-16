
from __future__ import annotations

from typing import Any, Protocol


class RetrievalIndex(Protocol):
    """Contract for the index operations required by hybrid retrieval."""

    def search(
        self,
        query: str,
        limit: int = 10,
        notes_prefix: str | None = None,
    ) -> list[dict[str, Any]]:
        ...

    def get_note(self, path: str) -> dict[str, Any] | None:
        ...

    def get_outgoing_links(
        self,
        source_path: str,
        relationship_type: str | None = None,
    ) -> list[dict[str, Any]]:
        ...

    def get_backlinks(
        self,
        target_path: str,
    ) -> list[dict[str, Any]]:
        ...