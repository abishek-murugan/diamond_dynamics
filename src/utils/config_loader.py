"""Configuration loader utility."""

import yaml
from pathlib import Path
from typing import Any, Dict
from src.utils.exceptions import ConfigurationError


def load_yaml(file_path: str) -> Dict[str, Any]:
    """Load a YAML file and return its contents as a dictionary.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        Dict[str, Any]: The loaded YAML data.

    Raises:
        ConfigurationError: If the file is not found or invalid YAML.
    """
    path = Path(file_path)
    if not path.exists():
        raise ConfigurationError(f"Configuration file not found: {file_path}")

    try:
        with open(path, "rt", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if data is None:
                return {}
            return data
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Error parsing YAML file {file_path}: {e}")
    except Exception as e:
        raise ConfigurationError(f"Unexpected error loading config {file_path}: {e}")


class ConfigManager:
    """Manager to hold all configurations."""

    def __init__(self, config_dir: str = "config/"):
        """Initialize the configuration manager."""
        self.config_dir = Path(config_dir)
        self.paths = load_yaml(self.config_dir / "paths.yaml")
        self.config = load_yaml(self.config_dir / "config.yaml")
        self.model_config = load_yaml(self.config_dir / "model.yaml")

    def get_path(self, key: str) -> str:
        """Get a path by key."""
        if key not in self.paths:
            raise ConfigurationError(f"Path key '{key}' not found in paths.yaml")
        return self.paths[key]

    def get_model_config(self, key: str) -> Any:
        """Get a model configuration by key."""
        if key not in self.model_config:
            raise ConfigurationError(f"Model config key '{key}' not found in model.yaml")
        return self.model_config[key]
