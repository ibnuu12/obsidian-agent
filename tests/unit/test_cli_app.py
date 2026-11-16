from __future__ import annotations

import pytest

from agent_obsidian.application.agent_models import RetrievalOptions
from agent_obsidian.application.llm.exceptions import (
    LLMAuthenticationError,
    LLMError,
    LLMInvalidResponseError,
    LLMNotConfiguredError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from agent_obsidian.interfaces.cli import app


def test_run_cli_without_command_displays_help(capsys):
    result = app.run_cli([])

    output = capsys.readouterr().out

    assert result == 0
    assert "usage:" in output
    assert "commands:" in output


def test_run_cli_version(capsys):
    with pytest.raises(SystemExit) as exc_info:
        app.run_cli(["--version"])

    output = capsys.readouterr().out

    assert exc_info.value.code == 0
    assert "agent-obsidian 0.1.0" in output


@pytest.mark.parametrize(
    ("arguments", "expected_message"),
    [
        (
            ["ask", "test query", "--limit", "0"],
            "value must be greater than zero",
        ),
        (
            ["ask", "test query", "--limit", "-1"],
            "value must be greater than zero",
        ),
        (
            ["ask", "test query", "--limit", "invalid"],
            "value must be an integer",
        ),
        (
            ["ask", "test query", "--graph-limit", "-1"],
            "value must be zero or greater",
        ),
        (
            ["ask", "test query", "--graph-limit", "invalid"],
            "value must be an integer",
        ),
    ],
)
def test_run_cli_rejects_invalid_numeric_arguments(
    arguments,
    expected_message,
):
    with pytest.raises(SystemExit) as exc_info:
        app.run_cli(arguments)

    assert exc_info.value.code == 2


def test_run_cli_dispatches_ask_with_retrieval_options(monkeypatch):
    received = {}

    def fake_execute_ask(query, retrieval):
        received["query"] = query
        received["retrieval"] = retrieval
        return 0

    monkeypatch.setattr(
        app,
        "execute_ask",
        fake_execute_ask,
    )

    result = app.run_cli([
        "ask",
        "Explain knowledge graphs",
        "--limit",
        "7",
        "--graph-limit",
        "3",
        "--no-graph",
        "--tag",
        "knowledge",
        "--tag",
        "zettel",
        "--note-type",
        "permanent",
        "--status",
        "evergreen",
    ])

    assert result == 0
    assert received["query"] == "Explain knowledge graphs"

    retrieval = received["retrieval"]

    assert isinstance(retrieval, RetrievalOptions)
    assert retrieval.limit == 7
    assert retrieval.expand_graph is False
    assert retrieval.graph_limit == 3
    assert retrieval.tags == frozenset({"knowledge", "zettel"})
    assert retrieval.note_types == frozenset({"permanent"})
    assert retrieval.statuses == frozenset({"evergreen"})


@pytest.mark.parametrize(
    "error",
    [
        LLMError("base LLM error"),
        LLMAuthenticationError("authentication failed"),
        LLMRateLimitError("rate limit exceeded"),
        LLMTimeoutError("request timed out"),
        LLMProviderError("provider failed"),
        LLMInvalidResponseError("invalid response"),
        LLMNotConfiguredError("provider not configured"),
        OSError("database unavailable"),
        ValueError("invalid configuration"),
    ],
)
def test_run_cli_handles_expected_ask_errors(
    monkeypatch,
    capsys,
    error,
):
    def fake_execute_ask(query, retrieval):
        raise error

    monkeypatch.setattr(
        app,
        "execute_ask",
        fake_execute_ask,
    )

    result = app.run_cli([
        "ask",
        "test query",
    ])

    output = capsys.readouterr().out

    assert result == 1

    if isinstance(error, LLMError):
        assert f"LLM error: {error}" in output
    else:
        assert f"Error: {error}" in output


def test_run_cli_does_not_silently_swallow_unexpected_error(
    monkeypatch,
):
    def fake_execute_ask(query, retrieval):
        raise RuntimeError("unexpected failure")

    monkeypatch.setattr(
        app,
        "execute_ask",
        fake_execute_ask,
    )

    with pytest.raises(
        RuntimeError,
        match="unexpected failure",
    ):
        app.run_cli([
            "ask",
            "test query",
        ])


def test_run_cli_status_is_not_implemented():
    with pytest.raises(SystemExit) as exc_info:
        app.run_cli(["status"])

    assert exc_info.value.code == 2
