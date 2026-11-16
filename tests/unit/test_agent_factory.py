
from dataclasses import replace

import pytest

from agent_obsidian.application.agent_orchestrator import (
    AgentOrchestrator,
)
from agent_obsidian.infrastructure.agent import factory
from agent_obsidian.infrastructure.agent.factory import (
    AgentRuntime,
    create_agent_runtime,
)
from agent_obsidian.shared.config import load_settings


class FakeSQLiteIndex:
    def __init__(self, database_path):
        self.database_path = database_path
        self.close_count = 0

    def close(self):
        self.close_count += 1


def make_settings(tmp_path):
    return replace(
        load_settings(),
        database_path=tmp_path / "agent-test.db",
        llm_provider="none",
        llm_model="",
    )


def test_create_agent_runtime(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    index = FakeSQLiteIndex(settings.database_path)

    monkeypatch.setattr(
        factory,
        "SQLiteIndex",
        lambda path: index,
    )

    runtime = create_agent_runtime(settings)

    assert isinstance(runtime, AgentRuntime)
    assert isinstance(runtime.orchestrator, AgentOrchestrator)
    assert runtime.index is index
    assert runtime.orchestrator.provider is None
    assert index.close_count == 0

    runtime.close()

    assert index.close_count == 1

def test_agent_runtime_context_manager_closes_index(
    tmp_path,
    monkeypatch,
):
    settings = make_settings(tmp_path)
    index = FakeSQLiteIndex(settings.database_path)

    monkeypatch.setattr(
        factory,
        "SQLiteIndex",
        lambda path: index,
    )

    with pytest.raises(RuntimeError, match="test failure"):
        with create_agent_runtime(settings) as runtime:
            assert runtime.index is index
            raise RuntimeError("test failure")

    assert index.close_count == 1


def test_factory_closes_index_when_provider_creation_fails(
    tmp_path,
    monkeypatch,
):
    settings = make_settings(tmp_path)
    index = FakeSQLiteIndex(settings.database_path)

    monkeypatch.setattr(
        factory,
        "SQLiteIndex",
        lambda path: index,
    )

    def raise_provider_error(settings):
        raise ValueError("provider initialization failed")

    monkeypatch.setattr(
        factory,
        "create_llm_provider",
        raise_provider_error,
    )

    with pytest.raises(
        ValueError,
        match="provider initialization failed",
    ):
        create_agent_runtime(settings)

    assert index.close_count == 1
