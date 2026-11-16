
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
import os
import tomllib

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    project_root: Path
    environment: str
    log_level: str

    vault_path: Path
    obsidian_api_url: str
    mcp_endpoint: str
    obsidian_api_key: str | None
    ca_cert: Path | None
    request_timeout: int

    read_only: bool
    require_write_approval: bool
    allow_delete: bool
    allow_bulk_operations: bool

    llm_provider: str
    llm_model: str
    llm_api_key: str | None
    max_context_tokens: int
    max_output_tokens: int

    retrieval_backend: str
    database_path: Path
    max_results: int

    audit_enabled: bool
    audit_log_path: Path

    # LLM runtime configuration
    llm_timeout_seconds: int = 30
    llm_temperature: float = 0.2
    llm_monthly_budget_usd: float = 5.0


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()

    if not path.is_absolute():
        path = root / path

    return path.resolve()



def load_settings(config_path: Path | None = None) -> Settings:
    project_root = Path(__file__).resolve().parents[3]

    if config_path is None:
        local_config = project_root / "config" / "settings.toml"
        example_config = project_root / "config" / "settings.example.toml"

        if local_config.is_file():
            config_path = local_config
        elif example_config.is_file():
            config_path = example_config
        else:
            raise FileNotFoundError(
                "Configuration file not found. "
                "Expected config/settings.toml or "
                "config/settings.example.toml."
            )
    else:
        config_path = config_path.resolve()

    load_dotenv(project_root / ".env", override=False)

    with config_path.open("rb") as file:
        config = tomllib.load(file)

    app = config.get("app", {})
    obsidian = config.get("obsidian", {})
    security = config.get("security", {})
    llm = config.get("llm", {})
    retrieval = config.get("retrieval", {})
    audit = config.get("audit", {})

    ca_cert_value = (
        os.getenv("OBSIDIAN_CA_CERT")
        or obsidian.get("ca_cert")
        or ""
    )

    ca_cert = (
        _resolve_path(project_root, ca_cert_value)
        if ca_cert_value
        else None
    )

    return Settings(
        project_root=project_root,
        environment=os.getenv(
            "AGENT_ENV",
            app.get("environment", "development"),
        ),
        log_level=os.getenv(
            "AGENT_LOG_LEVEL",
            app.get("log_level", "INFO"),
        ).upper(),

        vault_path=_resolve_path(
            project_root,
            os.getenv("OBSIDIAN_VAULT_PATH")
            or obsidian.get("vault_path", ""),
        ),
        obsidian_api_url=obsidian.get(
            "api_url",
            "https://127.0.0.1:27124",
        ),
        mcp_endpoint=obsidian.get(
            "mcp_endpoint",
            "https://127.0.0.1:27124/mcp/",
        ),
        obsidian_api_key=os.getenv("OBSIDIAN_API_KEY") or None,
        ca_cert=ca_cert,
        request_timeout=int(
            obsidian.get("request_timeout", 30)
        ),

        read_only=bool(security.get("read_only", True)),
        require_write_approval=bool(
            security.get("require_write_approval", True)
        ),
        allow_delete=bool(security.get("allow_delete", False)),
        allow_bulk_operations=bool(
            security.get("allow_bulk_operations", False)
        ),

        llm_provider=os.getenv(
            "LLM_PROVIDER",
            llm.get("provider", "none"),
        ).strip().lower(),
        llm_model=os.getenv(
            "LLM_MODEL",
            llm.get("model", ""),
        ).strip(),
        llm_api_key=os.getenv("LLM_API_KEY") or None,
        llm_timeout_seconds=int(
            llm.get("timeout_seconds", 30)
        ),
        llm_temperature=float(
            llm.get("temperature", 0.2)
        ),
        llm_monthly_budget_usd=float(
            llm.get("monthly_budget_usd", 5.0)
        ),

        max_output_tokens=int(
            llm.get("max_output_tokens", 1024)
        ),
        max_context_tokens=int(
            llm.get("max_context_tokens", 8000)
        ),

        retrieval_backend=retrieval.get(
            "backend", "sqlite_fts5"
        ),
        database_path=_resolve_path(
            project_root,
            retrieval.get(
                "database_path",
                "data/index/agent.db",
            ),
        ),
        max_results=int(retrieval.get("max_results", 10)),

        audit_enabled=bool(audit.get("enabled", True)),
        audit_log_path=_resolve_path(
            project_root,
            audit.get("log_path", "data/logs/agent.log"),
        ),
    )


def validate_settings(settings: Settings) -> list[str]:
    errors = []

    if not settings.vault_path.is_dir():
        errors.append(
            f"Vault directory not found: {settings.vault_path}"
        )

    if not settings.obsidian_api_key:
        errors.append("OBSIDIAN_API_KEY is not configured")

    for endpoint in (
        settings.obsidian_api_url,
        settings.mcp_endpoint,
    ):
        parsed = urlparse(endpoint)

        if parsed.scheme != "https":
            errors.append(
                f"Endpoint must use HTTPS: {endpoint}"
            )
        elif parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
            errors.append(
                f"Endpoint must use localhost: {endpoint}"
            )

    if settings.ca_cert is None or not settings.ca_cert.is_file():
        errors.append("Obsidian TLS CA certificate not found")

    if not settings.read_only:
        errors.append(
            "MVP security policy requires read_only=true"
        )

    if not settings.require_write_approval:
        errors.append(
            "MVP security policy requires write approval"
        )

    if settings.allow_delete:
        errors.append("Delete operations must remain disabled")

    if settings.allow_bulk_operations:
        errors.append(
            "Bulk operations must remain disabled"
        )

    # LLM configuration validation
    supported_providers = {"none", "mock"}

    if settings.llm_provider not in supported_providers:
        errors.append(
            f"Unsupported LLM provider: {settings.llm_provider}"
        )

    if settings.llm_timeout_seconds <= 0:
        errors.append("LLM timeout must be greater than zero")

    if not 0.0 <= settings.llm_temperature <= 2.0:
        errors.append(
            "LLM temperature must be between 0.0 and 2.0"
        )

    if settings.max_context_tokens < 1:
        errors.append(
            "LLM max_context_tokens must be greater than zero"
        )

    if settings.max_output_tokens < 1:
        errors.append(
            "LLM max_output_tokens must be greater than zero"
        )

    if settings.llm_monthly_budget_usd < 0:
        errors.append(
            "LLM monthly budget cannot be negative"
        )

    return errors
