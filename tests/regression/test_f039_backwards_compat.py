"""F-039 backwards-compat: existing YAML still loads; queue default is additive."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mousedroid.config.schema.gcp_cloud import GCPConfig, GCPLoggingConfig
from mousedroid.config.schema.root import Settings

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_queue_maxsize_default_keeps_legacy_logging_block_valid() -> None:
    """A pre-F-039 logging block (enabled + min_level only) still constructs."""
    cfg = GCPLoggingConfig(enabled=False, min_level="INFO")
    assert cfg.queue_maxsize == 256


def test_settings_without_gcp_still_loads() -> None:
    """Stock Settings have no gcp block; F-039 must not require one."""
    settings = Settings(mock_hardware=True)
    assert settings.gcp is None
    assert settings.logging.level == "INFO"


def test_existing_gcp_yaml_overlay_loads() -> None:
    """Committed gcp overlay still validates after the new field."""
    overlay = _REPO_ROOT / "config" / "gcp_digital_twin.yaml"
    raw = yaml.safe_load(overlay.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    settings = Settings.model_validate({**raw, "mock_hardware": True})
    assert settings.gcp is not None
    assert settings.gcp.logging.queue_maxsize == 256
    assert settings.gcp.logging.min_level == "INFO"


def test_gcp_config_partial_block_still_defaults_logging_off() -> None:
    cfg = GCPConfig(project_id="some-project")
    assert cfg.logging.enabled is False
    assert cfg.logging.queue_maxsize == 256


@pytest.mark.parametrize(
    "overlay",
    sorted(p.name for p in (_REPO_ROOT / "config").glob("*.yaml")),
)
def test_shipped_overlays_still_load(overlay: str) -> None:
    path = _REPO_ROOT / "config" / overlay
    text = path.read_text(encoding="utf-8")
    if "# config-validator: skip" in text:
        pytest.skip(f"{overlay} carries skip marker (not a Settings overlay)")
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        return
    cfg = Settings.model_validate({**raw, "mock_hardware": True})
    if cfg.gcp is None:
        return
    assert cfg.gcp.logging.queue_maxsize == 256
