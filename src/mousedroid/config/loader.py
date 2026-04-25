"""YAML overlay loader for MouseDroid configuration.

Loads base defaults from config/default.yaml, then merges environment-specific
overlays. Environment variables with MOUSEDROID_ prefix override all.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from mousedroid.config.schema import Settings
from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)

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
        _log.debug("config_base_loaded", path=str(default_path))
    else:
        _log.debug("config_no_default_yaml", path=str(default_path))

    for overlay_path in overlay_paths:
        overlay_data = load_yaml(overlay_path)
        _deep_merge(merged, overlay_data)
        _log.debug("config_overlay_applied", path=str(overlay_path))

    _log.info("config_settings_resolved", n_overlays=len(overlay_paths))
    # pydantic-settings v2 gives init kwargs HIGHER priority than env vars.
    # Remove top-level keys from merged that are already overridden by a
    # MOUSEDROID_<KEY> env var so the env var source wins for those fields.
    env_prefix = "MOUSEDROID_"
    env_overridden = {
        k[len(env_prefix) :].lower()
        for k in os.environ
        if k.upper().startswith(env_prefix) and "__" not in k[len(env_prefix) :]
    }
    for key in env_overridden:
        merged.pop(key, None)

    # Sanitize nested env vars whose value is empty/whitespace. pydantic-settings
    # v2 interprets MOUSEDROID_SECTION__FIELD="" as {"section": {"field": ""}},
    # which then materializes an Optional nested config (e.g. GCPConfig) with
    # an empty required field and fails validation. Empty env values always
    # mean "unset" here, so drop them before Settings() construction.
    empty_nested = [
        k
        for k, v in os.environ.items()
        if k.upper().startswith(env_prefix) and "__" in k[len(env_prefix) :] and not v.strip()
    ]
    with _ScopedEnvUnset(empty_nested):
        return Settings(**merged)


class _ScopedEnvUnset:
    """Context manager that temporarily removes the given env vars."""

    def __init__(self, names: list[str]) -> None:
        self._names = names
        self._saved: dict[str, str] = {}

    def __enter__(self) -> _ScopedEnvUnset:
        for name in self._names:
            if name in os.environ:
                self._saved[name] = os.environ.pop(name)
                _log.debug("config_env_empty_nested_dropped", name=name)
        return self

    def __exit__(self, *_exc: object) -> None:
        for name, value in self._saved.items():
            os.environ[name] = value
