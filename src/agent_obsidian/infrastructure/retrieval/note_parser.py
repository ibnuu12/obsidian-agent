
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath
import re
from typing import Any

import yaml


WIKILINK_PATTERN = re.compile(r"!?\[\[([^\]]+)\]\]")
HEADING_PATTERN = re.compile(r"(?m)^#\s+(.+?)\s*$")


@dataclass
class ParsedNote:
    path: str
    note_id: str | None
    title: str
    aliases: list[str]
    note_type: str | None
    status: str | None
    tags: list[str]
    created: str | None
    updated: str | None
    frontmatter: dict[str, Any]
    content: str
    wikilinks: list[str] = field(default_factory=list)
    related: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _as_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (str, int, float)):
        return str(value)
    return None


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, str):
        value = [value]

    if not isinstance(value, (list, tuple, set)):
        return []

    return [
        str(item).strip()
        for item in value
        if item is not None and str(item).strip()
    ]


def _extract_frontmatter(
    markdown: str,
) -> tuple[dict[str, Any], str, list[str]]:
    lines = markdown.splitlines()

    if not lines or lines[0].strip() != "---":
        return {}, markdown, []

    closing_index = None

    for index in range(1, len(lines)):
        if lines[index].strip() in ("---", "..."):
            closing_index = index
            break

    if closing_index is None:
        return {}, markdown, ["Frontmatter opening delimiter has no closing delimiter"]

    raw_yaml = "\n".join(lines[1:closing_index])
    body = "\n".join(lines[closing_index + 1:])

    try:
        metadata = yaml.safe_load(raw_yaml) or {}
    except yaml.YAMLError as exc:
        return {}, markdown, [f"Invalid YAML frontmatter: {exc}"]

    if not isinstance(metadata, dict):
        return {}, markdown, ["Frontmatter must contain a YAML mapping"]

    return metadata, body, []



def _normalize_link_target(raw_target: str) -> str:
    target = raw_target.strip()

    # Remove Obsidian embed marker.
    if target.startswith("!"):
        target = target[1:].strip()

    # Remove Obsidian wikilink wrappers.
    if target.startswith("[[") and target.endswith("]]"):
        target = target[2:-2].strip()

    # Remove alias.
    target = target.split("|", 1)[0].strip()

    # Remove heading or block references.
    target = re.split(r"[#^]", target, maxsplit=1)[0].strip()

    # Normalize path separators.
    target = target.replace("\\", "/")

    return target


def _extract_wikilinks(content: str) -> list[str]:
    targets = []

    for match in WIKILINK_PATTERN.finditer(content):
        target = _normalize_link_target(match.group(1))

        if target and target not in targets:
            targets.append(target)

    return targets


def _extract_related(value: Any) -> list[str]:
    related = []

    for item in _as_string_list(value):
        target = _normalize_link_target(item)

        if target and target not in related:
            related.append(target)

    return related


def parse_note(path: str, markdown: str) -> ParsedNote:
    normalized_path = PurePosixPath(
        path.replace("\\", "/")
    ).as_posix()

    frontmatter, content, warnings = _extract_frontmatter(markdown)

    title = _as_string(frontmatter.get("title"))

    if not title:
        heading = HEADING_PATTERN.search(content)
        title = heading.group(1).strip() if heading else None

    if not title:
        title = PurePosixPath(normalized_path).stem

    aliases = _as_string_list(frontmatter.get("aliases"))
    tags = _as_string_list(frontmatter.get("tags"))
    related = _extract_related(frontmatter.get("related"))

    return ParsedNote(
        path=normalized_path,
        note_id=_as_string(frontmatter.get("id")),
        title=title,
        aliases=aliases,
        note_type=_as_string(frontmatter.get("type")),
        status=_as_string(frontmatter.get("status")),
        tags=tags,
        created=_as_string(frontmatter.get("created")),
        updated=_as_string(frontmatter.get("updated")),
        frontmatter=frontmatter,
        content=content,
        wikilinks=_extract_wikilinks(content),
        related=related,
        warnings=warnings,
    )
