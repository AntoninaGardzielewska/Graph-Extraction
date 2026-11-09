from pathlib import Path

import yaml


def load_config_yaml(config_file: str | Path) -> dict:
    """Load a YAML config from the repository config folder or an explicit path."""
    path = Path(config_file)
    project_root = Path(__file__).resolve().parents[2]
    if not path.is_absolute():
        configured_path = project_root / "resources" / path
        path = configured_path if configured_path.exists() else project_root / path

    with path.open(encoding="utf-8") as config_stream:
        config = yaml.safe_load(config_stream)
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a YAML mapping: {path}")
    return config
