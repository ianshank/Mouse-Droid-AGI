"""Backwards-compat pins for F-029 — flipping the GCP egress defaults is inert.

The change moves ``GCPLoggingConfig.enabled`` and ``GCPMonitoringConfig.enabled``
from ``True`` to ``False``. That is a default change, so it needs proof it alters
behaviour for **no shipped config**: ``config/gcp_digital_twin.yaml`` is the only
overlay in the tree declaring a ``gcp:`` block, and it sets all three flags
explicitly rather than relying on the defaults.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from mousedroid.config.schema.gcp_cloud import GCPConfig
from mousedroid.config.schema.root import Settings

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TWIN_OVERLAY = _REPO_ROOT / "config" / "gcp_digital_twin.yaml"


def test_absent_gcp_block_still_yields_none() -> None:
    """No gcp: block means no GCP config at all — unchanged by this feature."""
    assert Settings(mock_hardware=True).gcp is None


def test_shipped_twin_overlay_resolves_to_the_same_effective_state() -> None:
    """The one overlay declaring gcp: sets the flags explicitly, so it is unaffected.

    This is the assertion that makes the default flip safe. If a future edit
    deletes those explicit lines and leans on the defaults, this test turns red
    rather than the rover quietly ceasing to export.
    """
    data = yaml.safe_load(_TWIN_OVERLAY.read_text(encoding="utf-8"))
    gcp = GCPConfig.model_validate(data["gcp"])
    assert gcp.logging.enabled is True
    assert gcp.monitoring.enabled is True
    assert gcp.firestore.enabled is False


def test_twin_overlay_sets_the_flags_explicitly_rather_than_by_default() -> None:
    """Belt-and-braces: the YAML text itself carries the opt-in."""
    data = yaml.safe_load(_TWIN_OVERLAY.read_text(encoding="utf-8"))
    assert data["gcp"]["logging"]["enabled"] is True
    assert data["gcp"]["monitoring"]["enabled"] is True


def test_shipped_default_yaml_still_parses() -> None:
    """The committed default config loads unchanged and declares no gcp block."""
    data = yaml.safe_load((_REPO_ROOT / "config" / "default.yaml").read_text(encoding="utf-8"))
    settings = Settings.model_validate(data)
    assert settings.gcp is None


def test_unrelated_fields_survive_a_partial_gcp_block() -> None:
    """The block is additive — it does not reach into anything it should not."""
    cfg = GCPConfig(project_id="some-project", robot_id="droid-002")
    assert cfg.project_id == "some-project"
    assert cfg.robot_id == "droid-002"
    # Sub-blocks unrelated to egress keep their own defaults.
    assert cfg.storage.bucket == GCPConfig(project_id="x").storage.bucket
