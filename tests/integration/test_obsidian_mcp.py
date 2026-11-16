
import asyncio
import ssl
import json
import httpx2

from mcp import Client
from mcp.client.streamable_http import streamable_http_client

from agent_obsidian.shared.config import load_settings


async def main():
    settings = load_settings()

    if not settings.obsidian_api_key:
        raise RuntimeError("OBSIDIAN_API_KEY is not configured")

    if not settings.ca_cert or not settings.ca_cert.is_file():
        raise RuntimeError("Obsidian CA certificate not found")

    ssl_context = ssl.create_default_context(
        cafile=str(settings.ca_cert)
    )

    timeout = httpx2.Timeout(
        settings.request_timeout,
        read=300.0,
    )

    async with httpx2.AsyncClient(
        headers={
            "Authorization": f"Bearer {settings.obsidian_api_key}",
        },
        verify=ssl_context,
        timeout=timeout,
    ) as http_client:

        transport = streamable_http_client(
            settings.mcp_endpoint,
            http_client=http_client,
        )

        async with Client(transport) as client:
            print("MCP connection: OK")

            result = await client.list_tools()

            print("\nAvailable MCP tools:")

            for tool in result.tools:
                print(f"\nName: {tool.name}")
                print(f"Description: {tool.description or '-'}")
                print("Input schema:")
                print(json.dumps(
                    tool.input_schema,
                    indent=2,
                    ensure_ascii=False,
                ))

            print("\nTool discovery: OK")

if __name__ == "__main__":
    asyncio.run(main())
