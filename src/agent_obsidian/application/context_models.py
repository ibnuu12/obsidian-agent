
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContextItem:
    """A retrieved note selected for the LLM context."""

    path: str
    title: str
    note_type: str | None
    status: str | None
    tags: tuple[str, ...]
    source: str
    score: float | None
    excerpt: str
    estimated_tokens: int


@dataclass(frozen=True)
class ContextBuildResult:
    """The result of building a token-budgeted context."""

    context: str
    items: tuple[ContextItem, ...]
    estimated_tokens: int
    budget_tokens: int
    excluded_count: int
    truncated_count: int
