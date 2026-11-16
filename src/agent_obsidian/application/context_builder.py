
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from agent_obsidian.application.context_models import (
    ContextBuildResult,
    ContextItem,
)


def estimate_tokens(text: str) -> int:
    """
    Estimate token count using a deterministic character heuristic.

    This is a conservative approximation, not a provider-specific tokenizer.
    """
    if not text:
        return 0

    return math.ceil(len(text) / 4)


class ContextBuilder:
    """
    Build an excerpt-first context from hybrid retrieval results.

    Direct FTS results are prioritized over graph results.
    The builder does not access the vault, SQLite, or an LLM provider.
    """

    SEPARATOR = "\n\n"

    def build(
        self,
        notes: Sequence[Mapping[str, Any]],
        max_tokens: int,
    ) -> ContextBuildResult:
        if (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or max_tokens < 1
        ):
            raise ValueError("max_tokens must be a positive integer")

        candidates = self._prioritize_notes(notes)

        selected_items: list[ContextItem] = []
        selected_blocks: list[str] = []
        excluded_count = 0
        truncated_count = 0

        for index, note in enumerate(candidates):
            excerpt = note.get("excerpt")

            # Excerpt-first: never fall back to the full note content.
            if not isinstance(excerpt, str) or not excerpt.strip():
                excluded_count += 1
                continue

            excerpt = excerpt.strip()
            full_block = self._format_note(note, excerpt)
            candidate_context = self._join_blocks(
                [*selected_blocks, full_block]
            )

            if estimate_tokens(candidate_context) <= max_tokens:
                selected_blocks.append(full_block)
                selected_items.append(
                    self._make_item(note, excerpt, full_block)
                )
                continue

            # Try to fit a shortened excerpt for this note.
            truncated_excerpt = self._fit_excerpt(
                note=note,
                excerpt=excerpt,
                existing_blocks=selected_blocks,
                max_tokens=max_tokens,
            )

            if truncated_excerpt is None:
                excluded_count += 1
                continue

            truncated_block = self._format_note(
                note,
                truncated_excerpt,
            )

            selected_blocks.append(truncated_block)
            selected_items.append(
                self._make_item(
                    note,
                    truncated_excerpt,
                    truncated_block,
                )
            )
            truncated_count += 1

            # Lower-priority notes must not displace this partial result.
            excluded_count += len(candidates) - index - 1
            break

        context = self._join_blocks(selected_blocks)

        return ContextBuildResult(
            context=context,
            items=tuple(selected_items),
            estimated_tokens=estimate_tokens(context),
            budget_tokens=max_tokens,
            excluded_count=excluded_count,
            truncated_count=truncated_count,
        )

    @staticmethod
    def _prioritize_notes(
        notes: Sequence[Mapping[str, Any]],
    ) -> list[Mapping[str, Any]]:
        """
        Keep input order within each source priority.

        FTS results come first, graph results second, and unknown sources last.
        """
        priorities = {
            "fts": 0,
            "graph": 1,
        }

        return sorted(
            notes,
            key=lambda note: priorities.get(
                note.get("source"),
                2,
            ),
        )

    @staticmethod
    def _format_note(
        note: Mapping[str, Any],
        excerpt: str,
    ) -> str:
        title = str(note.get("title") or "Untitled")
        path = str(note.get("path") or "")
        source = str(note.get("source") or "unknown")

        lines = [
            f"### {title}",
            f"Path: {path}",
        ]

        note_type = note.get("type", note.get("note_type"))
        status = note.get("status")
        tags = note.get("tags", [])
        aliases = note.get("aliases", [])

        if note_type:
            lines.append(f"Type: {note_type}")

        if status:
            lines.append(f"Status: {status}")

        if tags:
            lines.append(f"Tags: {', '.join(map(str, tags))}")

        if aliases:
            lines.append(
                f"Aliases: {', '.join(map(str, aliases))}"
            )

        lines.append(f"Source: {source}")
        lines.extend(["", excerpt])

        return "\n".join(lines)

    @staticmethod
    def _join_blocks(blocks: list[str]) -> str:
        return ContextBuilder.SEPARATOR.join(blocks)

    def _fit_excerpt(
        self,
        note: Mapping[str, Any],
        excerpt: str,
        existing_blocks: list[str],
        max_tokens: int,
    ) -> str | None:
        """
        Find the longest excerpt prefix that fits the remaining budget.
        """
        suffix = " …"
        low = 1
        high = len(excerpt)
        best: str | None = None

        while low <= high:
            middle = (low + high) // 2
            shortened = excerpt[:middle].rstrip()

            if not shortened:
                low = middle + 1
                continue

            candidate_excerpt = (
                shortened + suffix
                if middle < len(excerpt)
                else shortened
            )

            block = self._format_note(note, candidate_excerpt)
            context = self._join_blocks(
                [*existing_blocks, block]
            )

            if estimate_tokens(context) <= max_tokens:
                best = candidate_excerpt
                low = middle + 1
            else:
                high = middle - 1

        return best

    @staticmethod
    def _make_item(
        note: Mapping[str, Any],
        excerpt: str,
        block: str,
    ) -> ContextItem:
        note_type = note.get("type", note.get("note_type"))
        tags = note.get("tags", [])
        score = note.get("score")

        return ContextItem(
            path=str(note.get("path") or ""),
            title=str(note.get("title") or "Untitled"),
            note_type=note_type,
            status=note.get("status"),
            tags=tuple(str(tag) for tag in tags),
            source=str(note.get("source") or "unknown"),
            score=score,
            excerpt=excerpt,
            estimated_tokens=estimate_tokens(block),
        )
