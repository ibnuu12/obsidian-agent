
from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any


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
    Build a token-budgeted context from hybrid retrieval results.

    Direct FTS results are prioritized over graph results.
    Duplicate note paths are included only once.

    MOC notes use full content and section-aware formatting.
    Regular notes remain excerpt-first.

    The builder does not access the vault, SQLite, or an LLM provider.
    """

    SEPARATOR = "\n\n"
    MOC_NOTE_TYPE = "moc"
    SECTION_PATTERN = re.compile(
        r"^(##)\s+(.+?)\s*#*\s*$",
        re.MULTILINE,
    )

    MOC_SECTION_PRIORITY = {
        "core domain navigation": 0,
        "core purpose": 1,
        "core domain overview": 2,
        "cross-vault navigator": 3,
        "active indexing": 4,
    }

    def build(
        self,
        notes: Sequence[Mapping[str, Any]],
        max_tokens: int,
    ):
        from agent_obsidian.application.context_models import (
            ContextBuildResult,
            ContextItem,
        )

        if (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or max_tokens < 1
        ):
            raise ValueError("max_tokens must be a positive integer")

        prioritized = self._prioritize_notes(notes)
        candidates, duplicate_count = self._deduplicate_notes(
            prioritized
        )

        selected_items: list[ContextItem] = []
        selected_blocks: list[str] = []
        excluded_count = duplicate_count
        truncated_count = 0

        for index, note in enumerate(candidates):
            context_text = self._get_context_text(note)

            if context_text is None:
                excluded_count += 1
                continue

            full_block = self._format_note(note, context_text)
            candidate_context = self._join_blocks(
                [*selected_blocks, full_block]
            )

            if estimate_tokens(candidate_context) <= max_tokens:
                selected_blocks.append(full_block)
                selected_items.append(
                    self._make_item(note, context_text, full_block)
                )
                continue

            if self._is_moc(note):
                truncated_text = self._fit_moc_text(
                    note=note,
                    text=context_text,
                    existing_blocks=selected_blocks,
                    max_tokens=max_tokens,
                )
            else:
                truncated_text = self._fit_text(
                    note=note,
                    text=context_text,
                    existing_blocks=selected_blocks,
                    max_tokens=max_tokens,
                )

            if truncated_text is None:
                excluded_count += 1
                continue

            truncated_block = self._format_note(
                note,
                truncated_text,
            )

            selected_blocks.append(truncated_block)
            selected_items.append(
                self._make_item(
                    note,
                    truncated_text,
                    truncated_block,
                )
            )
            truncated_count += 1

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

    @classmethod
    def _is_moc(cls, note: Mapping[str, Any]) -> bool:
        note_type = note.get("type", note.get("note_type"))

        return (
            isinstance(note_type, str)
            and note_type.strip().casefold() == cls.MOC_NOTE_TYPE
        )

    @classmethod
    def _get_context_text(
        cls,
        note: Mapping[str, Any],
    ) -> str | None:
        """
        MOC notes prefer full content, with excerpt as fallback.
        Other note types must have a valid excerpt.
        """
        if cls._is_moc(note):
            content = note.get("content")

            if isinstance(content, str) and content.strip():
                return content.strip()

        excerpt = note.get("excerpt")

        if isinstance(excerpt, str) and excerpt.strip():
            return excerpt.strip()

        return None

    @staticmethod
    def _prioritize_notes(
        notes: Sequence[Mapping[str, Any]],
    ) -> list[Mapping[str, Any]]:
        """
        Keep input order within each source priority.

        FTS results come first, graph results second,
        and unknown sources last.
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
    def _deduplicate_notes(
        notes: Sequence[Mapping[str, Any]],
    ) -> tuple[list[Mapping[str, Any]], int]:
        """
        Keep the first result for each non-empty normalized path.

        Since notes are already prioritized, an FTS result wins
        over a graph result for the same path.
        """
        unique_notes: list[Mapping[str, Any]] = []
        seen_paths: set[str] = set()
        duplicate_count = 0

        for note in notes:
            path = note.get("path")

            if not isinstance(path, str) or not path.strip():
                unique_notes.append(note)
                continue

            normalized_path = path.strip().replace("\\", "/").casefold()

            if normalized_path in seen_paths:
                duplicate_count += 1
                continue

            seen_paths.add(normalized_path)
            unique_notes.append(note)

        return unique_notes, duplicate_count

    @staticmethod
    def _format_header(note: Mapping[str, Any]) -> list[str]:
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
            lines.append(f"Aliases: {', '.join(map(str, aliases))}")

        lines.append(f"Source: {source}")

        return lines

    @classmethod
    def _format_note(
        cls,
        note: Mapping[str, Any],
        text: str,
    ) -> str:
        if cls._is_moc(note):
            return cls._format_moc_note(note, text)

        lines = cls._format_header(note)
        lines.extend(["", text])

        return "\n".join(lines)

    @classmethod
    def _parse_moc_sections(
        cls,
        text: str,
    ) -> tuple[str, list[tuple[str, str]]]:
        """
        Split MOC content into its preamble and Markdown sections.

        Heading levels 2 through 6 are treated as sections.
        The original heading text and section body are preserved.
        """
        matches = list(cls.SECTION_PATTERN.finditer(text))

        if not matches:
            return text.strip(), []

        preamble = text[:matches[0].start()].strip()
        sections: list[tuple[str, str]] = []

        for index, match in enumerate(matches):
            heading = match.group(2).strip()
            body_start = match.end()

            body_end = (
                matches[index + 1].start()
                if index + 1 < len(matches)
                else len(text)
            )

            body = text[body_start:body_end].strip()
            sections.append((heading, body))

        return preamble, sections

    @classmethod
    def _format_moc_note(
        cls,
        note: Mapping[str, Any],
        text: str,
    ) -> str:
        lines = cls._format_header(note)
        preamble, sections = cls._parse_moc_sections(text)

        lines.extend(["", "<MOC_NOTE>"])

        if preamble:
            lines.extend(["<MOC_INTRO>", preamble, "</MOC_INTRO>"])

        if sections:
            for heading, body in sections:
                lines.append(f"<SECTION: {heading}>")

                if body:
                    lines.append(body)

                lines.append("</SECTION>")
        else:
            lines.extend(["<MOC_CONTENT>", text.strip(), "</MOC_CONTENT>"])

        lines.append("</MOC_NOTE>")

        return "\n".join(lines)

    def _fit_moc_text(
        self,
        note: Mapping[str, Any],
        text: str,
        existing_blocks: list[str],
        max_tokens: int,
    ) -> str | None:
        """
        Select complete MOC sections by importance when the full
        document does not fit.

        The final text retains the original section order.
        """
        preamble, sections = self._parse_moc_sections(text)

        if not sections:
            return self._fit_text(
                note=note,
                text=text,
                existing_blocks=existing_blocks,
                max_tokens=max_tokens,
            )

        ranked_sections = sorted(
            enumerate(sections),
            key=lambda item: (
                self.MOC_SECTION_PRIORITY.get(
                    item[1][0].casefold(),
                    5,
                ),
                item[0],
            ),
        )

        selected_indices: set[int] = set()

        for index, _section in ranked_sections:
            candidate_indices = selected_indices | {index}

            candidate_text = self._assemble_moc_text(
                preamble,
                sections,
                candidate_indices,
            )

            candidate_block = self._format_moc_note(
                note,
                candidate_text,
            )

            candidate_context = self._join_blocks(
                [*existing_blocks, candidate_block]
            )

            if estimate_tokens(candidate_context) <= max_tokens:
                selected_indices.add(index)

        if not selected_indices:
            return self._fit_text(
                note=note,
                text=text,
                existing_blocks=existing_blocks,
                max_tokens=max_tokens,
            )

        selected_text = self._assemble_moc_text(
            preamble,
            sections,
            selected_indices,
        )

        omitted = len(selected_indices) < len(sections)

        if omitted:
            candidate_with_marker = selected_text.rstrip() + "\n\n…"
            marker_block = self._format_moc_note(
                note,
                candidate_with_marker,
            )
            marker_context = self._join_blocks(
                [*existing_blocks, marker_block]
            )

            if estimate_tokens(marker_context) <= max_tokens:
                selected_text = candidate_with_marker

        return selected_text

    @staticmethod
    def _assemble_moc_text(
        preamble: str,
        sections: list[tuple[str, str]],
        selected_indices: set[int],
    ) -> str:
        parts: list[str] = []

        if preamble:
            parts.append(preamble)

        for index, (heading, body) in enumerate(sections):
            if index not in selected_indices:
                continue

            section_text = f"## {heading}"

            if body:
                section_text += f"\n\n{body}"

            parts.append(section_text)

        return "\n\n".join(parts)

    @staticmethod
    def _join_blocks(blocks: list[str]) -> str:
        return ContextBuilder.SEPARATOR.join(blocks)

    def _fit_text(
        self,
        note: Mapping[str, Any],
        text: str,
        existing_blocks: list[str],
        max_tokens: int,
    ) -> str | None:
        """
        Find the longest text prefix that fits the remaining budget.
        """
        suffix = " …"
        low = 1
        high = len(text)
        best: str | None = None

        while low <= high:
            middle = (low + high) // 2
            shortened = text[:middle].rstrip()

            if not shortened:
                low = middle + 1
                continue

            candidate_text = (
                shortened + suffix
                if middle < len(text)
                else shortened
            )

            block = self._format_note(note, candidate_text)
            context = self._join_blocks(
                [*existing_blocks, block]
            )

            if estimate_tokens(context) <= max_tokens:
                best = candidate_text
                low = middle + 1
            else:
                high = middle - 1

        return best

    @staticmethod
    def _make_item(
        note: Mapping[str, Any],
        text: str,
        block: str,
    ):
        from agent_obsidian.application.context_models import ContextItem

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
            excerpt=text,
            estimated_tokens=estimate_tokens(block),
        )
