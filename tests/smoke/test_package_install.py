"""Smoke test: verify mousedroid package installs and imports correctly."""

from __future__ import annotations

import re

import pytest

pytestmark = pytest.mark.smoke

# Semver regex — validates major.minor.patch with optional pre-release/build metadata
_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+")


def test_version_is_string() -> None:
    """Verify __version__ is a non-empty string from importlib.metadata."""
    import mousedroid

    assert hasattr(mousedroid, "__version__")
    assert isinstance(mousedroid.__version__, str)
    assert len(mousedroid.__version__) > 0


def test_version_is_valid_semver() -> None:
    """Verify __version__ matches semver pattern (no hardcoded version check)."""
    import mousedroid

    assert _SEMVER_RE.match(
        mousedroid.__version__
    ), f"Version {mousedroid.__version__!r} is not a valid semver string"


def test_version_matches_metadata() -> None:
    """Verify __version__ matches importlib.metadata (single-source coherence)."""
    from importlib.metadata import version as _pkg_version

    import mousedroid

    assert mousedroid.__version__ == _pkg_version("mousedroid")


def test_settings_instantiation() -> None:
    """Verify Settings can be instantiated with mock_hardware=True."""
    from mousedroid.config.schema import Settings

    settings = Settings(mock_hardware=True)
    assert settings.mock_hardware is True


def test_public_modules_importable() -> None:
    """Verify key public modules can be imported without errors."""
    import importlib

    modules = [
        "mousedroid.config.schema",
        "mousedroid.factory",
    ]
    for module_name in modules:
        mod = importlib.import_module(module_name)
        assert mod is not None, f"Failed to import {module_name}"


def test_build_world_model_callable() -> None:
    """Verify build_world_model is importable and callable (do not invoke)."""
    from mousedroid.factory import build_world_model

    assert callable(build_world_model)
