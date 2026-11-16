
from agent_obsidian.shared.config import (
    load_settings,
    validate_settings,
)

settings = load_settings()

print("Environment :", settings.environment)
print("Vault       :", settings.vault_path)
print("MCP endpoint:", settings.mcp_endpoint)
print("LLM provider:", settings.llm_provider)
print("Read only   :", settings.read_only)
print("API key     :", "CONFIGURED" if settings.obsidian_api_key else "MISSING")

errors = validate_settings(settings)

if errors:
    print("\nConfiguration errors:")
    for error in errors:
        print("-", error)
else:
    print("\nConfiguration: OK")
