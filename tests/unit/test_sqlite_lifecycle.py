
import sqlite3
from unittest.mock import Mock

import pytest

from agent_obsidian.infrastructure.retrieval.sqlite_index import (
    SQLiteIndex,
)


@pytest.mark.parametrize(
    "failure_stage",
    ["execute", "executescript"],
)
def test_constructor_closes_connection_on_failure(
    tmp_path,
    monkeypatch,
    failure_stage,
):
    connection = Mock()

    error = sqlite3.OperationalError(
        "Database initialization failed"
    )

    if failure_stage == "execute":
        connection.execute.side_effect = error
    else:
        connection.executescript.side_effect = error

    monkeypatch.setattr(
        sqlite3,
        "connect",
        lambda *args, **kwargs: connection,
    )

    with pytest.raises(
        sqlite3.OperationalError,
        match="Database initialization failed",
    ):
        SQLiteIndex(tmp_path / "failure.db")

    connection.close.assert_called_once()


@pytest.mark.parametrize(
    "operation",
    ["commit", "rollback"],
)
def test_context_manager_closes_connection_on_failure(
    operation,
):
    connection = Mock()

    error = sqlite3.OperationalError(
        f"{operation} failed"
    )

    getattr(connection, operation).side_effect = error

    index = SQLiteIndex.__new__(SQLiteIndex)
    index.connection = connection

    if operation == "commit":
        exit_args = (None, None, None)
    else:
        original_error = RuntimeError("Original error")
        exit_args = (
            RuntimeError,
            original_error,
            None,
        )

    with pytest.raises(
        sqlite3.OperationalError,
        match=f"{operation} failed",
    ):
        index.__exit__(*exit_args)

    connection.close.assert_called_once()


def test_context_manager_commits_and_closes():
    connection = Mock()

    index = SQLiteIndex.__new__(SQLiteIndex)
    index.connection = connection

    index.__exit__(None, None, None)

    connection.commit.assert_called_once()
    connection.rollback.assert_not_called()
    connection.close.assert_called_once()


def test_context_manager_rolls_back_and_closes():
    connection = Mock()

    index = SQLiteIndex.__new__(SQLiteIndex)
    index.connection = connection

    original_error = RuntimeError("Operation failed")

    index.__exit__(
        RuntimeError,
        original_error,
        None,
    )

    connection.rollback.assert_called_once()
    connection.commit.assert_not_called()
    connection.close.assert_called_once()


def test_cleanup_error_does_not_hide_commit_error():
    connection = Mock()

    commit_error = sqlite3.OperationalError(
        "Commit failed"
    )

    connection.commit.side_effect = commit_error
    connection.close.side_effect = RuntimeError(
        "Close failed"
    )

    index = SQLiteIndex.__new__(SQLiteIndex)
    index.connection = connection

    with pytest.raises(sqlite3.OperationalError) as exc:
        index.__exit__(None, None, None)

    assert exc.value is commit_error
    connection.close.assert_called_once()
