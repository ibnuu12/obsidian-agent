
import asyncio
import os
from agent_obsidian.shared.config import load_settings
from agent_obsidian.infrastructure.mcp.obsidian_client import ObsidianMCPClient


async def main():
    settings = load_settings()

    async with ObsidianMCPClient(settings) as obsidian:
        vault_root = os.getenv("OBSIDIAN_TEST_VAULT_ROOT", "")
        if not vault_root:
            raise RuntimeError(
                "Set OBSIDIAN_TEST_VAULT_ROOT to a vault folder for integration testing"
            )

        result = await obsidian.list_vault(vault_root)

        print("TYPE:", type(result))
        print("STRUCTURED CONTENT:")
        print(repr(getattr(result, "structuredContent", None)))

        print("\nCONTENT:")
        for item in getattr(result, "content", []):
            print("Type:", type(item))
            print("Text:", repr(getattr(item, "text", None)))

        print("\nIS ERROR:", getattr(result, "is_error", None))


if __name__ == "__main__":
    asyncio.run(main())
