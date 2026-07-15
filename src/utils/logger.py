"""Logging configuration and setup."""

import logging
import logging.config
import os
import yaml
from pathlib import Path


def setup_logging(
    default_path: str = "config/logging.yaml",
    default_level: int = logging.INFO,
    env_key: str = "LOG_CFG",
) -> None:
    """Setup logging configuration."""
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value

    if os.path.exists(path):
        with open(path, "rt", encoding="utf-8") as f:
            try:
                config = yaml.safe_load(f.read())
                # Ensure logs directory exists if a file handler is configured
                for handler in config.get("handlers", {}).values():
                    if "filename" in handler:
                        log_dir = Path(handler["filename"]).parent
                        log_dir.mkdir(parents=True, exist_ok=True)
                logging.config.dictConfig(config)
            except Exception as e:
                print(f"Error in Logging Configuration. Using default configs. Error: {e}")
                logging.basicConfig(level=default_level)
    else:
        logging.basicConfig(level=default_level)
        print(f"Failed to load configuration file {path}. Using basic configs.")


def get_logger(name: str) -> logging.Logger:
    """Get logger with the specified name."""
    return logging.getLogger(name)
