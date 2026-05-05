"""Regression: OpenClaw schema additions must not break existing YAML.

Three guarantees:

1. Every YAML file under ``config/`` loads cleanly into the new
   :class:`Settings` schema (overlay files are merged onto
   ``default.yaml`` via :func:`load_settings` like in production).
2. ``Settings.openclaw`` defaults to ``None`` for every file (no
   pre-existing YAML opts in to OpenClaw).
3. ``Settings.mcp.bind_external`` defaults to ``False`` for every file
   that defines an ``mcp:`` block.

Failures here mean a schema change accidentally became opt-out instead
of opt-in.

Files marked with ``# config-validator: skip`` (e.g. deploy descriptors
that intentionally reuse the ``jetson:`` namespace with deploy-only
keys) are excluded by the same convention the existing
``test_config_overlays_load`` test uses.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mousedroid.config.loader import load_settings


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _validator_skipped(path: Path) -> bool:
    """Return True when the file declares ``# config-validator: skip``."""
    try:
        head = path.read_text(encoding="utf-8")[:512]
    except OSError:
        return False
    return "config-validator: skip" in head


def _yaml_files() -> list[Path]:
    config_dir = _project_root() / "config"
    return sorted(p for p in config_dir.glob("*.yaml") if p.is_file() and not _validator_skipped(p))


@pytest.mark.parametrize("yaml_path", _yaml_files(), ids=lambda p: p.name)
def test_yaml_loads_with_openclaw_schema(yaml_path: Path) -> None:
    config_dir = _project_root() / "config"
    # Use the production loader so overlays merge on top of default.yaml
    # — exactly how the runtime loads them. Setting the overlay path
    # explicitly skips the auto-discovery that would pull in every
    # overlay in config/.
    s = load_settings(yaml_path, config_dir=config_dir)
    assert s.openclaw is None, (
        f"{yaml_path.name} unexpectedly opts in to OpenClaw; "
        "the schema change must keep openclaw=None as the default."
    )
    if s.mcp is not None:
        assert s.mcp.bind_external is False, (
            f"{yaml_path.name} unexpectedly enables mcp.bind_external; "
            "the schema change must keep bind_external=False as the default."
        )


def test_at_least_one_yaml_file_present() -> None:
    """Sanity: parametrize doesn't silently skip every test."""
    assert len(_yaml_files()) > 0, "no config/*.yaml files found"
