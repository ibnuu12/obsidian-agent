
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

    metrics = response.performance

    if metrics is not None:
        print("\nPerformance:")
        print(f"Retrieval: {metrics.retrieval_ms:.2f} ms")
        print(f"Context Build: {metrics.context_build_ms:.2f} ms")
        print(f"Prompt Build: {metrics.prompt_build_ms:.2f} ms")
        print(f"LLM Inference: {metrics.llm_inference_ms:.2f} ms")
        print(f"Total: {metrics.total_ms:.2f} ms")

        print(f"Retrieved Notes: {metrics.retrieved_notes}")
        print(f"Context Items: {metrics.context_items}")
        print(
            "Estimated Context Tokens: "
            f"{metrics.estimated_context_tokens}"
        )
        print(
            "Estimated Prompt Tokens: "
            f"{metrics.estimated_prompt_tokens}"
        )

    runtime = response.llm_runtime_metrics

    if runtime is not None:
        print("\nLLM Runtime Diagnostics:")

        print(
            "Total Duration: "
            f"{runtime.total_duration_ms:.2f} ms"
            if runtime.total_duration_ms is not None
            else "Total Duration: N/A"
        )

        print(
            "Model Load: "
            f"{runtime.load_duration_ms:.2f} ms"
            if runtime.load_duration_ms is not None
            else "Model Load: N/A"
        )

        print(
            "Prompt Evaluation: "
            f"{runtime.prompt_eval_duration_ms:.2f} ms"
            if runtime.prompt_eval_duration_ms is not None
            else "Prompt Evaluation: N/A"
        )

        print(
            "Token Generation: "
            f"{runtime.eval_duration_ms:.2f} ms"
            if runtime.eval_duration_ms is not None
            else "Token Generation: N/A"
        )

        print(
            "Generation Speed: "
            f"{runtime.tokens_per_second:.2f} tokens/s"
            if runtime.tokens_per_second is not None
            else "Generation Speed: N/A"
        )

    print()