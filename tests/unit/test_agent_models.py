
import pytest

from agent_obsidian.application.agent_models import (
    AgentRequest,
    AgentResponse,
    RetrievalOptions,
)
from agent_obsidian.application.context_models import ContextItem
from agent_obsidian.application.llm.models import LLMUsage


def test_retrieval_options_defaults():
    options = RetrievalOptions()

    assert options.limit is None
    assert options.expand_graph is True
    assert options.graph_limit == 5
    assert options.note_types is None
    assert options.statuses is None
    assert options.tags is None


def test_retrieval_options_normalizes_filters():
    options = RetrievalOptions(
        note_types={"permanent"},
        statuses={"evergreen"},
        tags={"knowledge", "zettel/permanent"},
    )

    assert options.note_types == frozenset({"permanent"})
    assert options.statuses == frozenset({"evergreen"})
    assert options.tags == frozenset({
        "knowledge",
        "zettel/permanent",
    })


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_retrieval_options_rejects_invalid_limit(limit):
    with pytest.raises(ValueError):
        RetrievalOptions(limit=limit)


@pytest.mark.parametrize("graph_limit", [-1, True, 1.5])
def test_retrieval_options_rejects_invalid_graph_limit(graph_limit):
    with pytest.raises(ValueError):
        RetrievalOptions(graph_limit=graph_limit)


def test_retrieval_options_rejects_invalid_filter():
    with pytest.raises(ValueError):
        RetrievalOptions(note_types={"permanent", ""})


def test_agent_request_rejects_empty_query():
    with pytest.raises(ValueError):
        AgentRequest(query="   ")


def test_agent_request_rejects_invalid_retrieval():
    with pytest.raises(TypeError):
        AgentRequest(query="knowledge", retrieval="invalid")


def test_agent_request_accepts_retrieval_options():
    options = RetrievalOptions(limit=5)

    request = AgentRequest(
        query="knowledge graph",
        retrieval=options,
    )

    assert request.query == "knowledge graph"
    assert request.retrieval is options


def test_agent_response_preserves_sources_and_usage():
    source = ContextItem(
        path="30 Notes/Knowledge Graph.md",
        title="Knowledge Graph",
        note_type="permanent",
        status="evergreen",
        tags=("knowledge",),
        source="fts",
        score=-1.5,
        excerpt="A graph of connected knowledge.",
        estimated_tokens=10,
    )

    usage = LLMUsage(
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
    )

    response = AgentResponse(
        answer="A knowledge graph connects related notes.",
        sources=(source,),
        usage=usage,
        provider="mock",
        model="mock-model",
    )

    assert response.sources == (source,)
    assert response.usage == usage
    assert response.provider == "mock"
    assert response.model == "mock-model"


def test_agent_response_rejects_empty_answer():
    with pytest.raises(ValueError):
        AgentResponse(
            answer=" ",
            sources=(),
            usage=None,
            provider="mock",
            model="mock-model",
        )
