"""YAML overlay loader for MouseDroid configuration.

Loads base defaults from config/default.yaml, then merges environment-specific
overlays. Environment variables with MOUSEDROID_ prefix override all.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from mousedroid.config.schema import Settings

_DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent.parent / "config"


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge overlay dict into base dict.

    Args:
        base: Base configuration dictionary.
        overlay: Overlay values to merge on top.

    Returns:
        Merged dictionary (base is modified in-place and returned).
    """
    for key, value in overlay.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a single YAML file and return its contents as a dict.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed YAML contents.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    with path.open() as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}
    return data


def load_settings(
    *overlay_paths: Path,
    config_dir: Path | None = None,
) -> Settings:
    """Load Settings by merging default.yaml with optional overlay files.

    Merge order:
        1. config/default.yaml (base)
        2. Each overlay_path in order
        3. Environment variables (MOUSEDROID_* prefix, handled by pydantic-settings)

    Args:
        overlay_paths: Additional YAML files to merge on top of defaults.
        config_dir: Directory containing default.yaml. Defaults to project config/.

    Returns:
        Fully resolved Settings instance.
    """
    base_dir = config_dir or _DEFAULT_CONFIG_DIR
    default_path = base_dir / "default.yaml"

    merged: dict[str, Any] = {}
    if default_path.exists():
        merged = load_yaml(default_path)

    for overlay_path in overlay_paths:
        overlay_data = load_yaml(overlay_path)
        _deep_merge(merged, overlay_data)

    return Settings(**merged)
