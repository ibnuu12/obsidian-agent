from __future__ import annotations

from dataclasses import dataclass

from agent_obsidian.application.agent_orchestrator import (
    AgentOrchestrator,
)
from agent_obsidian.application.context_builder import ContextBuilder
from agent_obsidian.application.hybrid_retrieval import (
    HybridRetrievalService,
)
from agent_obsidian.infrastructure.llm.factory import (
    create_llm_provider,
)
from agent_obsidian.infrastructure.retrieval.sqlite_index import (
    SQLiteIndex,
)
from agent_obsidian.shared.config import Settings


@dataclass(slots=True)
class AgentRuntime:
    """Own the lifecycle of infrastructure used by the agent."""

    orchestrator: AgentOrchestrator
    index: SQLiteIndex

    def close(self) -> None:
        self.index.close()

    def __enter__(self) -> AgentRuntime:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> None:
        self.close()


def create_agent_runtime(settings: Settings) -> AgentRuntime:
    """
    Compose infrastructure and application dependencies.

    The SQLite index connection is owned by the returned runtime.
    The caller is responsible for closing the runtime.
    """

    index = SQLiteIndex(settings.database_path)

    try:
        retrieval_service = HybridRetrievalService(index)
        context_builder = ContextBuilder()
        provider = create_llm_provider(settings)

        orchestrator = AgentOrchestrator(
            retrieval_service=retrieval_service,
            context_builder=context_builder,
            provider=provider,
            settings=settings,
        )

        return AgentRuntime(
            orchestrator=orchestrator,
            index=index,
        )

    except Exception:
        index.close()
        raise