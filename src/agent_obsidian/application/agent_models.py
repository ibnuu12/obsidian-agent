
from __future__ import annotations

from dataclasses import dataclass

from agent_obsidian.application.context_models import ContextItem
from agent_obsidian.application.llm.models import LLMUsage


@dataclass(frozen=True, slots=True)
class RetrievalOptions:
    """Per-request retrieval configuration."""

    limit: int | None = None
    expand_graph: bool = True
    graph_limit: int = 5
    note_types: frozenset[str] | None = None
    statuses: frozenset[str] | None = None
    tags: frozenset[str] | None = None

    def __post_init__(self) -> None:
        if self.limit is not None:
            if (
                isinstance(self.limit, bool)
                or not isinstance(self.limit, int)
                or self.limit < 1
            ):
                raise ValueError("limit must be a positive integer or None")

        if not isinstance(self.expand_graph, bool):
            raise TypeError("expand_graph must be a boolean")

        if (
            isinstance(self.graph_limit, bool)
            or not isinstance(self.graph_limit, int)
            or self.graph_limit < 0
        ):
            raise ValueError("graph_limit must be a non-negative integer")

        for field_name in ("note_types", "statuses", "tags"):
            value = getattr(self, field_name)

            if value is None:
                continue

            if not isinstance(value, (set, frozenset)):
                raise TypeError(
                    f"{field_name} must be a set, frozenset, or None"
                )

            if not all(
                isinstance(item, str) and item.strip()
                for item in value
            ):
                raise ValueError(
                    f"{field_name} must contain non-empty strings"
                )

            object.__setattr__(
                self,
                field_name,
                frozenset(item.strip() for item in value),
            )


@dataclass(frozen=True, slots=True)
class AgentRequest:
    """Input contract for one agent execution."""

    query: str
    retrieval: RetrievalOptions | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.query, str):
            raise TypeError("query must be a string")

        if not self.query.strip():
            raise ValueError("query cannot be empty")

        if self.retrieval is not None and not isinstance(
            self.retrieval,
            RetrievalOptions,
        ):
            raise TypeError(
                "retrieval must be a RetrievalOptions instance or None"
            )


@dataclass(frozen=True, slots=True)
class AgentResponse:
    """Output contract returned by the agent orchestrator."""

    answer: str
    sources: tuple[ContextItem, ...]
    usage: LLMUsage | None
    provider: str
    model: str

    def __post_init__(self) -> None:
        if not isinstance(self.answer, str):
            raise TypeError("answer must be a string")

        if not self.answer.strip():
            raise ValueError("answer cannot be empty")

        if not isinstance(self.sources, tuple):
            raise TypeError("sources must be a tuple")

        if not all(
            isinstance(source, ContextItem)
            for source in self.sources
        ):
            raise TypeError("sources must contain ContextItem instances")

        if self.usage is not None and not isinstance(
            self.usage,
            LLMUsage,
        ):
            raise TypeError("usage must be an LLMUsage instance or None")

        if not isinstance(self.provider, str) or not self.provider.strip():
            raise ValueError("provider cannot be empty")

        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model cannot be empty")
