
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from agent_obsidian.infrastructure.mcp.obsidian_client import (
    ObsidianMCPClient,
)


def test_exit_resets_state_when_stack_close_fails():
    client = ObsidianMCPClient.__new__(ObsidianMCPClient)

    error = RuntimeError("Stack close failed")

    stack = SimpleNamespace(
        __aexit__=AsyncMock(side_effect=error)
    )

    client._stack = stack
    client._client = Mock()
    client._available_tools = {"vault_list", "vault_read"}

    with pytest.raises(
        RuntimeError,
        match="Stack close failed",
    ):
        asyncio.run(
            client.__aexit__(None, None, None)
        )

    assert client._stack is None
    assert client._client is None
    assert client._available_tools == set()


def test_exit_resets_state_without_stack():
    client = ObsidianMCPClient.__new__(ObsidianMCPClient)

    client._stack = None
    client._client = Mock()
    client._available_tools = {"vault_list"}

    asyncio.run(
        client.__aexit__(None, None, None)
    )

    assert client._stack is None
    assert client._client is None
    assert client._available_tools == set()


def test_exit_passes_exception_to_stack():
    client = ObsidianMCPClient.__new__(ObsidianMCPClient)

    stack = SimpleNamespace(
        __aexit__=AsyncMock(return_value=False)
    )

    client._stack = stack
    client._client = Mock()
    client._available_tools = {"vault_list"}

    original_error = RuntimeError("Operation failed")

    asyncio.run(
        client.__aexit__(
            RuntimeError,
            original_error,
            None,
        )
    )

    stack.__aexit__.assert_awaited_once_with(
        RuntimeError,
        original_error,
        None,
    )

    assert client._stack is None
    assert client._client is None
    assert client._available_tools == set()

class FakeAsyncContext:
    def __init__(self, value):
        self.value = value
        self.entered = False
        self.exited = False

    async def __aenter__(self):
        self.entered = True
        return self.value

    async def __aexit__(self, exc_type, exc_value, traceback):
        self.exited = True
        return False


class FakeAsyncExitStack:
    def __init__(self, cleanup_error=None):
        self.contexts = []
        self.cleanup_error = cleanup_error

    async def __aenter__(self):
        return self

    async def enter_async_context(self, context):
        value = await context.__aenter__()
        self.contexts.append(context)
        return value

    async def aclose(self):
        for context in reversed(self.contexts):
            await context.__aexit__(None, None, None)

        if self.cleanup_error is not None:
            raise self.cleanup_error


class FakeMCPClient:
    def __init__(self, tools=None, list_tools_error=None):
        self.tools = tools or []
        self.list_tools_error = list_tools_error

    async def list_tools(self):
        if self.list_tools_error is not None:
            raise self.list_tools_error

        return SimpleNamespace(tools=self.tools)


def prepare_startup_test(monkeypatch, mcp_client, cleanup_error=None):
    import agent_obsidian.infrastructure.mcp.obsidian_client as module

    http_context = FakeAsyncContext(object())
    mcp_context = FakeAsyncContext(mcp_client)

    stack = FakeAsyncExitStack(cleanup_error)

    monkeypatch.setattr(module, "AsyncExitStack", lambda: stack)
    monkeypatch.setattr(
        module.httpx2,
        "AsyncClient",
        lambda **kwargs: http_context,
    )
    monkeypatch.setattr(
        module,
        "streamable_http_client",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        module,
        "Client",
        lambda transport: mcp_context,
    )
    monkeypatch.setattr(
        module.ssl,
        "create_default_context",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        module.httpx2,
        "Timeout",
        lambda *args, **kwargs: object(),
    )

    settings = SimpleNamespace(
        obsidian_api_key="test-api-key",
        ca_cert=SimpleNamespace(is_file=lambda: True),
        request_timeout=30,
        mcp_endpoint="https://localhost:27124/mcp/",
    )

    client = ObsidianMCPClient(settings)

    return client, stack, http_context, mcp_context


def test_startup_failure_closes_opened_resources(monkeypatch):
    original_error = RuntimeError("MCP tool discovery failed")

    mcp = FakeMCPClient(
        list_tools_error=original_error,
    )

    client, stack, http_context, mcp_context = (
        prepare_startup_test(monkeypatch, mcp)
    )

    with pytest.raises(RuntimeError) as exc:
        asyncio.run(client.__aenter__())

    assert exc.value is original_error
    assert http_context.entered
    assert http_context.exited
    assert mcp_context.entered
    assert mcp_context.exited
    assert client._stack is None
    assert client._client is None
    assert client._available_tools == set()


def test_startup_failure_when_required_tools_are_missing(monkeypatch):
    mcp = FakeMCPClient(
        tools=[SimpleNamespace(name="vault_list")]
    )

    client, stack, http_context, mcp_context = (
        prepare_startup_test(monkeypatch, mcp)
    )

    with pytest.raises(
        RuntimeError,
        match="Required MCP tools missing",
    ):
        asyncio.run(client.__aenter__())

    assert http_context.exited
    assert mcp_context.exited
    assert client._stack is None
    assert client._client is None
    assert client._available_tools == set()


def test_startup_error_is_preserved_when_cleanup_fails(monkeypatch):
    original_error = RuntimeError("MCP startup failed")
    cleanup_error = RuntimeError("Cleanup failed")

    mcp = FakeMCPClient(
        list_tools_error=original_error,
    )

    client, stack, http_context, mcp_context = (
        prepare_startup_test(
            monkeypatch,
            mcp,
            cleanup_error=cleanup_error,
        )
    )

    with pytest.raises(RuntimeError) as exc:
        asyncio.run(client.__aenter__())

    assert exc.value is original_error
    assert http_context.exited
    assert mcp_context.exited
    assert client._stack is None
    assert client._client is None
    assert client._available_tools == set()