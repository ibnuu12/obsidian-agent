import anyio
import getpass
import ssl
import httpx2
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from agent_obsidian.shared.config import load_settings

async def main():
    settings = load_settings()
    MCP_URL = settings.mcp_endpoint
    CERT_PATH = settings.ca_cert

    if CERT_PATH is None or not CERT_PATH.is_file():
        raise FileNotFoundError(
            f"CA certificate not found: {CERT_PATH}"
        )

    api_key = getpass.getpass("Obsidian API Key: ")

    if not api_key.strip():
        raise ValueError("API Key cannot be empty")

    ssl_context = ssl.create_default_context(
        cafile=str(CERT_PATH)
    )

    async with httpx2.AsyncClient(
        headers={
            "Authorization": f"Bearer {api_key.strip()}"
        },
        verify=ssl_context,
        timeout=httpx2.Timeout(30.0, read=300.0),
    ) as http_client:

        transport = streamable_http_client(
            MCP_URL,
            http_client=http_client,
        )

        async with Client(transport) as client:
            print("\n=== MCP CONNECTION ===")
            print("Server:", client.server_info)
            print("Protocol:", client.protocol_version)
            print("Capabilities:", client.server_capabilities)

            print("\n=== AVAILABLE TOOLS ===")
            result = await client.list_tools()

            for tool in result.tools:
                print(f"\nName: {tool.name}")
                print(f"Description: {tool.description}")
                print(f"Input schema: {tool.input_schema}")

            print(f"\nTotal tools: {len(result.tools)}")


if __name__ == "__main__":
    anyio.run(main)
