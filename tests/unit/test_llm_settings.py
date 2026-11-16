
from dataclasses import replace
import agent_obsidian.shared.config as config_module

from agent_obsidian.shared.config import (
    load_settings,
    validate_settings,
)


def test_load_llm_settings_from_toml(tmp_path, monkeypatch):
    config_file = tmp_path / "settings.toml"
    config_file.write_text(
        """
[llm]
provider = "mock"
model = "test-model"
timeout_seconds = 45
max_context_tokens = 6000
max_output_tokens = 1200
temperature = 0.5
monthly_budget_usd = 3.5
""",
        encoding="utf-8",
    )

    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    # Isolasi test dari konfigurasi .env lokal.
    monkeypatch.setattr(config_module, "load_dotenv", lambda *args, **kwargs: None)

    settings = load_settings(config_file)

    assert settings.llm_provider == "mock"
    assert settings.llm_model == "test-model"
    assert settings.llm_timeout_seconds == 45
    assert settings.max_context_tokens == 6000
    assert settings.max_output_tokens == 1200
    assert settings.llm_temperature == 0.5
    assert settings.llm_monthly_budget_usd == 3.5
    assert settings.llm_base_url == "http://127.0.0.1:11434"

def test_environment_overrides_toml_provider(
    tmp_path,
    monkeypatch,
):
    config_file = tmp_path / "settings.toml"
    config_file.write_text(
        """
[llm]
provider = "none"
""",
        encoding="utf-8",
    )

    monkeypatch.setenv("LLM_PROVIDER", " MOCK ")

    settings = load_settings(config_file)

    assert settings.llm_provider == "mock"


def test_validation_rejects_unsupported_provider():
    settings = load_settings()

    invalid_settings = replace(
        settings,
        llm_provider="unsupported",
    )

    errors = validate_settings(invalid_settings)

    assert any(
        "Unsupported LLM provider" in error
        for error in errors
    )


def test_validation_rejects_negative_budget():
    settings = load_settings()

    invalid_settings = replace(
        settings,
        llm_monthly_budget_usd=-1.0,
    )

    errors = validate_settings(invalid_settings)

    assert any(
        "monthly budget" in error.lower()
        for error in errors
    )


def test_validation_rejects_invalid_temperature():
    settings = load_settings()

    invalid_settings = replace(
        settings,
        llm_temperature=3.0,
    )

    errors = validate_settings(invalid_settings)

    assert any(
        "temperature" in error.lower()
        for error in errors
    )

def test_environment_overrides_toml_base_url(
    tmp_path,
    monkeypatch,
):
    config_file = tmp_path / "settings.toml"
    config_file.write_text(
        """
[llm]
provider = "ollama"
model = "qwen2.5:3b"
base_url = "http://127.0.0.1:11434"
""",
        encoding="utf-8",
    )

    monkeypatch.setenv(
        "LLM_BASE_URL",
        "http://localhost:11434/",
    )

    settings = load_settings(config_file)

    assert settings.llm_base_url == "http://localhost:11434"


def test_validation_accepts_ollama_localhost():
    settings = load_settings()

    ollama_settings = replace(
        settings,
        llm_provider="ollama",
        llm_model="qwen2.5:3b",
        llm_base_url="http://127.0.0.1:11434",
    )

    errors = validate_settings(ollama_settings)

    assert not any(
        "Unsupported LLM provider" in error
        or "Ollama base URL" in error
        or "LLM model must be configured" in error
        for error in errors
    )


def test_validation_rejects_remote_ollama_url():
    settings = load_settings()

    invalid_settings = replace(
        settings,
        llm_provider="ollama",
        llm_model="qwen2.5:3b",
        llm_base_url="http://192.168.1.10:11434",
    )

    errors = validate_settings(invalid_settings)

    assert any(
        "Ollama base URL" in error
        for error in errors
    )


def test_validation_requires_ollama_model():
    settings = load_settings()

    invalid_settings = replace(
        settings,
        llm_provider="ollama",
        llm_model="",
    )

    errors = validate_settings(invalid_settings)

    assert any(
        "LLM model must be configured" in error
        for error in errors
    )