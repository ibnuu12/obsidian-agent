
from contextlib import AsyncExitStack
from pathlib import PurePosixPath
import re
import ssl

import httpx2
from mcp import Client
from mcp.client.streamable_http import streamable_http_client

from agent_obsidian.shared.config import Settings


ALLOWED_READ_TOOLS = {
    "vault_list",
    "vault_read",
    "search_simple",
    "search_query",
    "tag_list",
    "vault_get_document_map",
    "active_file_get_path",
}

REQUIRED_TOOLS = {
    "vault_list",
    "vault_read",
}


class ObsidianMCPClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._stack = None
        self._client = None
        self._available_tools = set()


    async def __aenter__(self):
        if not self.settings.obsidian_api_key:
            raise RuntimeError("Obsidian API key is missing")

        if not self.settings.ca_cert or not self.settings.ca_cert.is_file():
            raise RuntimeError("Obsidian CA certificate not found")

        ssl_context = ssl.create_default_context(
            cafile=str(self.settings.ca_cert)
        )

        timeout = httpx2.Timeout(
            self.settings.request_timeout,
            read=300.0,
        )

        self._available_tools.clear()
        self._client = None

        stack = AsyncExitStack()
        self._stack = stack

        try:
            await stack.__aenter__()

            http_client = await stack.enter_async_context(
                httpx2.AsyncClient(
                    headers={
                        "Authorization": (
                            f"Bearer {self.settings.obsidian_api_key}"
                        )
                    },
                    verify=ssl_context,
                    timeout=timeout,
                )
            )

            transport = streamable_http_client(
                self.settings.mcp_endpoint,
                http_client=http_client,
            )

            self._client = await stack.enter_async_context(
                Client(transport)
            )

            tools_result = await self._client.list_tools()

            self._available_tools = {
                tool.name for tool in tools_result.tools
            }

            missing = REQUIRED_TOOLS - self._available_tools

            if missing:
                raise RuntimeError(
                    f"Required MCP tools missing: {sorted(missing)}"
                )

            return self

        except BaseException:
            try:
                await stack.aclose()
            except BaseException:
                # Preserve the original startup exception.
                pass
            finally:
                self._stack = None
                self._client = None
                self._available_tools.clear()

            raise

    async def __aexit__(self, exc_type, exc_value, traceback):
        stack = self._stack

        try:
            if stack is not None:
                await stack.__aexit__(
                    exc_type,
                    exc_value,
                    traceback,
                )
        finally:
            self._stack = None
            self._client = None
            self._available_tools.clear()

    @staticmethod
    def _validate_path(path: str, allow_empty: bool = False) -> str:
        if not isinstance(path, str):
            raise ValueError("Vault path must be a string")

        path = path.strip()

        if not path:
            if allow_empty:
                return ""
            raise ValueError("Vault file path cannot be empty")

        normalized = path.replace("\\", "/")

        if (
            normalized.startswith("/")
            or re.match(r"^[A-Za-z]:", normalized)
            or "\x00" in normalized
        ):
            raise ValueError("Path must be relative to vault root")

        parsed = PurePosixPath(normalized)

        if ".." in parsed.parts:
            raise ValueError("Path traversal is not allowed")

        return parsed.as_posix()

    async def _call_read_tool(
        self,
        name: str,
        arguments: dict | None = None,
    ):
        if self._client is None:
            raise RuntimeError("MCP client is not connected")

        if name not in ALLOWED_READ_TOOLS:
            raise PermissionError(
                f"MCP tool is blocked by read-only policy: {name}"
            )

        if name not in self._available_tools:
            raise RuntimeError(
                f"MCP tool is not available: {name}"
            )

        result = await self._client.call_tool(
            name,
            arguments or {},
        )

        if getattr(result, "is_error", False):
            messages = [
                getattr(item, "text", "")
                for item in result.content
            ]
            raise RuntimeError(
                f"MCP tool failed ({name}): {' '.join(messages)}"
            )

        return result

    async def list_tools(self):
        if self._client is None:
            raise RuntimeError("MCP client is not connected")

        return await self._client.list_tools()

    async def list_vault(self, path: str = ""):
        path = self._validate_path(path, allow_empty=True)

        arguments = {"path": path} if path else {}

        return await self._call_read_tool(
            "vault_list",
            arguments,
        )

    async def read_note(self, path: str):
        path = self._validate_path(path)

        return await self._call_read_tool(
            "vault_read",
            {"path": path},
        )

    async def search_notes(
        self,
        query: str,
        max_results: int = 10,
        context_length: int = 200,
    ):
        if not query.strip():
            raise ValueError("Search query cannot be empty")

        if max_results < 1:
            raise ValueError("max_results must be positive")

        if not 0 <= context_length <= 500:
            raise ValueError("context_length must be between 0 and 500")

        return await self._call_read_tool(
            "search_simple",
            {
                "query": query,
                "maxResults": max_results,
                "contextLength": context_length,
            },
        )

    async def get_document_map(self, path: str):
        path = self._validate_path(path)

        return await self._call_read_tool(
            "vault_get_document_map",
            {"path": path},
        )

    async def list_tags(self):
        return await self._call_read_tool("tag_list")

    async def get_active_file_path(self):
        return await self._call_read_tool("active_file_get_path")
