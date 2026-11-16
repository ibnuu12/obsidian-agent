
import asyncio
import os

from agent_obsidian.shared.config import load_settings
from agent_obsidian.infrastructure.mcp.obsidian_client import ObsidianMCPClient
from test_helpers import print_mcp_result


async def main():
    settings = load_settings()

    vault_root = os.getenv("OBSIDIAN_TEST_VAULT_ROOT", "").strip()

    if not vault_root:
        raise RuntimeError(
            "Set OBSIDIAN_TEST_VAULT_ROOT to a vault folder "
            "before running this integration test."
        )

    moc_path = os.getenv(
        "OBSIDIAN_TEST_MOC_PATH",
        f"{vault_root}/20 MOCs",
    )

    async with ObsidianMCPClient(settings) as obsidian:
        for path in (vault_root, moc_path):
            print(f"\n=== LIST: {path} ===")
            result = await obsidian.list_vault(path)
            print("Result type:", type(result))
            print_mcp_result(result)


if __name__ == "__main__":
    asyncio.run(main())
