"""Regression: every overlay under ``config/`` loads against the schema.

This catches any future YAML/schema drift before it reaches CI's heavier
stages. Mirrors :mod:`scripts.validate_configs` so the same set is checked
both locally (via the script) and from pytest.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from mousedroid.config.loader import load_settings
from mousedroid.config.schema import Settings

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _REPO_ROOT / "config"
_SCRIPT = _REPO_ROOT / "scripts" / "validate_configs.py"


def _discover_overlays() -> list[Path]:
    """Reuse the script's discovery so test + CLI stay in lockstep."""
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))
    try:
        from validate_configs import discover_overlays
    finally:
        sys.path.pop(0)
    overlays: list[Path] = list(discover_overlays(_CONFIG_DIR, include_default=False))
    return overlays


_OVERLAYS = _discover_overlays()
_VALID_PLATFORMS = {"mouse_droid", "robot_arm"}


@pytest.mark.parametrize(
    "overlay_path",
    _OVERLAYS,
    ids=[p.name for p in _OVERLAYS],
)
def test_overlay_loads_against_schema(overlay_path: Path) -> None:
    """Every YAML overlay must produce a valid ``Settings`` object."""
    settings = load_settings(overlay_path, config_dir=_CONFIG_DIR)
    assert isinstance(settings, Settings)
    assert (
        settings.platform in _VALID_PLATFORMS
    ), f"{overlay_path.name}: unexpected platform {settings.platform!r}"


def test_default_yaml_loads_standalone() -> None:
    """``default.yaml`` must validate as the base config."""
    settings = load_settings(config_dir=_CONFIG_DIR)
    assert isinstance(settings, Settings)
    assert settings.platform in _VALID_PLATFORMS


def test_overlay_set_is_non_empty() -> None:
    """Guard against accidental glob misconfiguration."""
    assert _OVERLAYS, "No overlays discovered under config/ — regression net is empty."


def test_validate_configs_cli_exits_zero() -> None:
    """End-to-end: the CLI must succeed against the current repo state."""
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--include-default"],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )
    assert result.returncode == 0, (
        f"validate_configs.py exited {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
