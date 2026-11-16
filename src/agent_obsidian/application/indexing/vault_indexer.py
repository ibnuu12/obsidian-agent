
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath

from agent_obsidian.application.ports.vault_source import VaultSource
from agent_obsidian.infrastructure.retrieval.note_parser import parse_note
from agent_obsidian.infrastructure.retrieval.sqlite_index import SQLiteIndex


DEFAULT_CONTENT_ROOTS = (
    "K001_Finance",
    "K002_Tech",
    "K003_Personal",
)

EXCLUDED_DIRECTORY_NAMES = {
    ".obsidian",
}

EXCLUDED_DIRECTORY_SUFFIXES = {
    ("00 Meta", "Templates"),
    ("00 Meta", "Attachments"),
}


@dataclass
class IndexingResult:
    discovered: int = 0
    indexed: int = 0
    skipped: int = 0
    failed: int = 0
    relationship_resolution: dict[str, int] = field(
        default_factory=dict
    )
    errors: list[str] = field(default_factory=list)


class VaultIndexer:
    """
    Recursively indexes Markdown notes from configured vault roots.

    The indexer is read-only with respect to the Obsidian vault.
    It only writes to the local SQLite index.
    """

    def __init__(
        self,
        source: VaultSource,
        index: SQLiteIndex,
        content_roots: tuple[str, ...] = DEFAULT_CONTENT_ROOTS,
    ):
        if not content_roots:
            raise ValueError("At least one content root is required")

        self.source = source
        self.index = index
        self.content_roots = tuple(
            self._normalize_relative_path(root)
            for root in content_roots
        )

    @staticmethod
    def _normalize_relative_path(path: str) -> str:
        if not isinstance(path, str) or not path.strip():
            raise ValueError("Vault path must be a non-empty string")

        normalized = path.strip().replace("\\", "/")
        parsed = PurePosixPath(normalized)

        if (
            parsed.is_absolute()
            or ".." in parsed.parts
            or not parsed.parts
        ):
            raise ValueError(f"Invalid vault-relative path: {path}")

        return parsed.as_posix()

    @staticmethod
    def _is_excluded_directory(path: str) -> bool:
        parts = PurePosixPath(path).parts

        if any(
            part in EXCLUDED_DIRECTORY_NAMES
            for part in parts
        ):
            return True

        return any(
            len(parts) >= len(suffix)
            and parts[-len(suffix):] == suffix
            for suffix in EXCLUDED_DIRECTORY_SUFFIXES
        )

    @staticmethod
    def _is_markdown_file(path: str) -> bool:
        return PurePosixPath(path).suffix.casefold() == ".md"

    @staticmethod
    def _extract_file_metadata(stat: dict) -> tuple[float | None, int | None]:
        raw_mtime = stat.get("mtime")
        raw_size = stat.get("size")

        mtime = None
        size = None

        if isinstance(raw_mtime, (int, float)):
            # Obsidian Local REST API reports timestamps in milliseconds.
            mtime = float(raw_mtime) / 1000.0

        if isinstance(raw_size, int) and raw_size >= 0:
            size = raw_size

        return mtime, size

    async def _walk_directory(
        self,
        directory: str,
        result: IndexingResult,
    ) -> None:
        entries = await self.source.list_vault(directory)

        for entry in entries:
            if not isinstance(entry, str) or not entry.strip():
                result.skipped += 1
                continue

            is_directory = entry.endswith("/")
            name = entry.rstrip("/")

            if not name:
                result.skipped += 1
                continue

            # MCP entries are immediate children, not arbitrary paths.
            if (
                "/" in name
                or "\\" in name
                or name in {".", ".."}
            ):
                result.failed += 1
                result.errors.append(
                    f"Invalid vault entry in {directory}: {entry!r}"
                )
                continue

            child_path = f"{directory}/{name}"

            if is_directory:
                if self._is_excluded_directory(child_path):
                    result.skipped += 1
                    continue

                await self._walk_directory(child_path, result)
                continue

            if not self._is_markdown_file(child_path):
                result.skipped += 1
                continue

            result.discovered += 1

            try:
                document = await self.source.read_note(child_path)

                returned_path = document["path"]
                markdown = document["content"]
                stat = document.get("stat", {})

                normalized_returned_path = (
                    self._normalize_relative_path(returned_path)
                )

                if normalized_returned_path != child_path:
                    raise ValueError(
                        "MCP returned a different note path: "
                        f"{normalized_returned_path!r}"
                    )

                if not isinstance(markdown, str):
                    raise ValueError(
                        "MCP note content must be a string"
                    )

                if not isinstance(stat, dict):
                    raise ValueError(
                        "MCP note stat must be an object"
                    )

                note = parse_note(child_path, markdown)
                mtime, size = self._extract_file_metadata(stat)

                self.index.upsert_note(
                    note,
                    mtime=mtime,
                    size=size,
                )

                result.indexed += 1

            except Exception as exc:
                result.failed += 1
                result.errors.append(
                    f"{child_path}: {type(exc).__name__}: {exc}"
                )

    async def index_vault(self) -> IndexingResult:
        result = IndexingResult()

        for root in self.content_roots:
            try:
                await self._walk_directory(root, result)
            except Exception as exc:
                result.failed += 1
                result.errors.append(
                    f"{root}: {type(exc).__name__}: {exc}"
                )

        if result.indexed:
            result.relationship_resolution = (
                self.index.rebuild_relationship_resolution()
            )

        return result
