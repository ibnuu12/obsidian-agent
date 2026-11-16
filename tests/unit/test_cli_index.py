from __future__ import annotations

from agent_obsidian.application.indexing.vault_indexer import (
    IndexingResult,
)
from agent_obsidian.interfaces.cli.commands import index as index_command


def test_execute_index_rejects_unsupported_root(capsys):
    result = index_command.execute_index(
        root="outside_vault"
    )

    output = capsys.readouterr().out

    assert result == 1
    assert "unsupported vault root" in output


def test_present_indexing_result(capsys):
    result = IndexingResult(
        discovered=10,
        indexed=8,
        skipped=2,
        failed=0,
        relationship_resolution={
            "resolved": 5,
            "unresolved": 3,
        },
    )

    index_command._present_indexing_result(
        root="K003_Personal",
        result=result,
    )

    output = capsys.readouterr().out

    assert "K003_Personal" in output
    assert "Discovered : 10" in output
    assert "Indexed    : 8" in output
    assert "Skipped    : 2" in output
    assert "Failed     : 0" in output
    assert "resolved: 5" in output
    assert "unresolved: 3" in output


def test_present_indexing_result_displays_errors(capsys):
    result = IndexingResult(
        discovered=2,
        indexed=1,
        failed=1,
        errors=[
            "K003_Personal/example.md: RuntimeError: read failed"
        ],
    )

    index_command._present_indexing_result(
        root="K003_Personal",
        result=result,
    )

    output = capsys.readouterr().out

    assert "Errors" in output
    assert "RuntimeError: read failed" in output

def test_run_cli_dispatches_index_command(monkeypatch):
    from agent_obsidian.interfaces.cli import app

    received = {}

    def fake_execute_index(root):
        received["root"] = root
        return 0

    monkeypatch.setattr(
        app,
        "execute_index",
        fake_execute_index,
    )

    result = app.run_cli([
        "index",
        "--root",
        "K003_Personal",
    ])

    assert result == 0
    assert received["root"] == "K003_Personal"


def test_run_cli_rejects_invalid_index_root():
    from agent_obsidian.interfaces.cli.app import run_cli

    import pytest

    with pytest.raises(SystemExit) as exc_info:
        run_cli([
            "index",
            "--root",
            "outside_vault",
        ])

    assert exc_info.value.code == 2