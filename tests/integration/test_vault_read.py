
import asyncio

from agent_obsidian.shared.config import load_settings
from agent_obsidian.infrastructure.mcp.obsidian_client import (
    ObsidianMCPClient,
)

from test_helpers import print_mcp_result


async def main():
    settings = load_settings()

    async with ObsidianMCPClient(settings) as obsidian:
        print("MCP connection: OK")

        print("\n=== VAULT LIST ===")

        listing = await obsidian.list_vault()
        print_mcp_result(listing)

        print("\n=== NOTE READ ===")

        path = input(
            "\nEnter an existing note path from the list "
            "(leave empty to skip): "
        ).strip()

        if path:
            note = await obsidian.read_note(path)
            print_mcp_result(note)
            print("\nNote read: OK")
        else:
            print("\nNote read: SKIPPED")

        print("\nVault read test: COMPLETED")


if __name__ == "__main__":
    asyncio.run(main())
