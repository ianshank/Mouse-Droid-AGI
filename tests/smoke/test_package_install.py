"""Smoke test: verify mousedroid package installs and imports correctly.

Validates version consistency, semver format, and that all key public
modules are importable without hardware dependencies (GPIO, serial, etc.).
"""

from __future__ import annotations

import importlib
import re

import pytest

EXPECTED_VERSION = "0.2.0"

# Semver pattern: MAJOR.MINOR.PATCH with optional pre-release / build metadata
_SEMVER_RE = re.compile(
    r"^(?P<major>0|[1-9]\d*)"
    r"\.(?P<minor>0|[1-9]\d*)"
    r"\.(?P<patch>0|[1-9]\d*)"
    r"(?:-(?P<pre>[0-9A-Za-z\-]+(?:\.[0-9A-Za-z\-]+)*))?"
    r"(?:\+(?P<build>[0-9A-Za-z\-]+(?:\.[0-9A-Za-z\-]+)*))?$"
)

# Public modules that must be importable without hardware (no GPIO, serial, camera).
_PUBLIC_MODULES = [
    "mousedroid.config.schema",
    "mousedroid.factory",
    "mousedroid.orchestrator",
    "mousedroid.telemetry",
    "mousedroid.resilience",
    "mousedroid.safety",
    "mousedroid.reward",
    "mousedroid.memory",
    "mousedroid.curiosity",
    "mousedroid.learning",
    "mousedroid.cognitive",
    "mousedroid.world_model",
    "mousedroid.logging.setup",
    "mousedroid.common",
    "mousedroid.utils",
]


@pytest.mark.smoke
def test_version_matches_expected() -> None:
    """Verify __version__ equals the expected release version."""
    import mousedroid

    assert mousedroid.__version__ == EXPECTED_VERSION, (
        f"Expected version {EXPECTED_VERSION}, got {mousedroid.__version__}"
    )


@pytest.mark.smoke
def test_version_is_valid_semver() -> None:
    """Verify __version__ is a valid semantic version string."""
    import mousedroid

    version = mousedroid.__version__
    assert isinstance(version, str)
    assert len(version) > 0
    assert _SEMVER_RE.match(version), (
        f"Version '{version}' does not match semver pattern"
    )


@pytest.mark.smoke
def test_version_components_are_integers() -> None:
    """Verify major.minor.patch components parse as non-negative integers."""
    import mousedroid

    match = _SEMVER_RE.match(mousedroid.__version__)
    assert match is not None

    major = int(match.group("major"))
    minor = int(match.group("minor"))
    patch = int(match.group("patch"))

    assert major >= 0
    assert minor >= 0
    assert patch >= 0


@pytest.mark.smoke
def test_settings_instantiation() -> None:
    """Verify Settings can be instantiated with mock_hardware=True."""
    from mousedroid.config.schema import Settings

    settings = Settings(mock_hardware=True)
    assert settings.mock_hardware is True


@pytest.mark.smoke
@pytest.mark.parametrize("module_name", _PUBLIC_MODULES)
def test_public_module_importable(module_name: str) -> None:
    """Verify each key public module imports without errors.

    Args:
        module_name: Fully qualified module path to import.
    """
    mod = importlib.import_module(module_name)
    assert mod is not None, f"Failed to import {module_name}"


@pytest.mark.smoke
def test_build_world_model_callable() -> None:
    """Verify build_world_model is importable and callable (do not invoke)."""
    from mousedroid.factory import build_world_model

    assert callable(build_world_model)


@pytest.mark.smoke
def test_package_has_py_typed() -> None:
    """Verify py.typed marker exists for PEP 561 compliance."""
    from pathlib import Path

    import mousedroid

    package_dir = Path(mousedroid.__file__).parent
    py_typed = package_dir / "py.typed"
    assert py_typed.exists(), "py.typed marker missing -- PEP 561 compliance requires it"
