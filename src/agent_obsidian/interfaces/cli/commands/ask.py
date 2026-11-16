
from __future__ import annotations

import asyncio

from agent_obsidian.application.agent_models import (
    AgentRequest,
    RetrievalOptions,
)
from agent_obsidian.infrastructure.agent.factory import (
    create_agent_runtime,
)
from agent_obsidian.shared.config import load_settings
from agent_obsidian.interfaces.cli.presenters.console import (
    present_agent_response,
)


def execute_ask(
    query: str,
    retrieval: RetrievalOptions,
) -> int:
    """
    Execute an ask command through the application orchestrator.
    """

    settings = load_settings()

    with create_agent_runtime(settings) as runtime:
        response = asyncio.run(
            runtime.orchestrator.run(
                AgentRequest(
                    query=query,
                    retrieval=retrieval,
                )
            )
        )

    present_agent_response(query, response)

    return 0
