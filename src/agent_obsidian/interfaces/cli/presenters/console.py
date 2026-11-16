
from __future__ import annotations

from agent_obsidian.application.agent_models import AgentResponse


SEPARATOR = "─" * 60


def present_agent_response(
    query: str,
    response: AgentResponse,
) -> None:
    """Render an agent response in the terminal."""

    print()
    print("Agent Obsidian")
    print(SEPARATOR)

    print("\nQuestion:")
    print(query)

    print("\nAnswer:")
    print(response.answer)

    print("\nSources:")

    if not response.sources:
        print("No relevant sources found.")
    else:
        for index, source in enumerate(response.sources, start=1):
            print(f"[{index}] {source.title}")
            print(f"    Path: {source.path}")
            print(f"    Type: {source.note_type or 'unknown'}")
            print(f"    Source: {source.source}")

    print()
    print(SEPARATOR)
    print(f"Provider: {response.provider}")
    print(f"Model: {response.model}")

    if response.usage is not None:
        print(f"Input tokens: {response.usage.input_tokens}")
        print(f"Output tokens: {response.usage.output_tokens}")
        print(f"Total tokens: {response.usage.total_tokens}")

    print()
