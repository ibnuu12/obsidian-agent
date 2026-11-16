
from pathlib import Path

import agent_obsidian.shared.config as config_module
from agent_obsidian.shared.config import load_settings


def prepare_project(tmp_path, monkeypatch):
    project_root = tmp_path

    fake_module_path = (
        project_root
        / "src"
        / "agent_obsidian"
        / "shared"
        / "config.py"
    )

    monkeypatch.setattr(
        config_module,
        "__file__",
        str(fake_module_path),
    )
    monkeypatch.setattr(
        config_module,
        "load_dotenv",
        lambda *args, **kwargs: None,
    )

    monkeypatch.delenv("AGENT_ENV", raising=False)
    monkeypatch.delenv("OBSIDIAN_VAULT_PATH", raising=False)

    config_dir = project_root / "config"
    config_dir.mkdir(parents=True)

    return project_root, config_dir


def test_fallback_to_example_when_local_config_missing(
    tmp_path,
    monkeypatch,
):
    project_root, config_dir = prepare_project(
        tmp_path,
        monkeypatch,
    )

    example_config = config_dir / "settings.example.toml"
    example_config.write_text(
        """
[app]
environment = "example"

[obsidian]
vault_path = "example-vault"
""",
        encoding="utf-8",
    )

    settings = load_settings()

    assert settings.environment == "example"
    assert settings.vault_path == (
        project_root / "example-vault"
    ).resolve()


def test_local_config_has_priority_over_example(
    tmp_path,
    monkeypatch,
):
    _, config_dir = prepare_project(
        tmp_path,
        monkeypatch,
    )

    (config_dir / "settings.example.toml").write_text(
        """
[app]
environment = "example"
""",
        encoding="utf-8",
    )

    (config_dir / "settings.toml").write_text(
        """
[app]
environment = "local"
""",
        encoding="utf-8",
    )

    settings = load_settings()

    assert settings.environment == "local"
