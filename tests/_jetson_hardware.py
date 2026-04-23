"""Shared helpers for Jetson-only test suites."""

from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import TYPE_CHECKING

from mousedroid.validation.runtime import load_runtime_settings

if TYPE_CHECKING:
    from mousedroid.config.schema import Settings


JETSON_PROD_CONFIG = os.getenv("MOUSEDROID_JETSON_CONFIG", "config/jetson_production.yaml")


def is_jetson_host() -> bool:
    """Return True when running on a Jetson host."""
    return platform.system() == "Linux" and Path("/etc/nv_tegra_release").exists()


def load_jetson_runtime_settings() -> Settings:
    """Load runtime settings for Jetson-targeted hardware validation."""
    raw_configs = os.getenv("MOUSEDROID_JETSON_CONFIGS", "").strip()
    if raw_configs:
        config_paths = [Path(part.strip()) for part in raw_configs.split(",") if part.strip()]
    else:
        config_path = Path(JETSON_PROD_CONFIG)
        config_paths = [config_path] if config_path.exists() else [Path("config/default.yaml")]

    settings = load_runtime_settings(config_paths)
    return settings.model_copy(update={"mock_hardware": False})
