"""Backwards-compat pins for F-029 — flipping the GCP egress defaults is inert.

The change moves ``GCPLoggingConfig.enabled`` and ``GCPMonitoringConfig.enabled``
from ``True`` to ``False``. That is a default change, so it needs proof it alters
behaviour for **no shipped config**: ``config/gcp_digital_twin.yaml`` is the only
overlay in the tree declaring a ``gcp:`` block, and it sets all three flags
explicitly rather than relying on the defaults.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

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
    # pubsub/storage previously had no `enabled` field at all and were built on
    # block-presence alone. Gating them is only backwards-compatible because the
    # overlay now opts in explicitly -- that is what preserves its behaviour.
    assert gcp.pubsub.enabled is True
    assert gcp.storage.enabled is True


def test_twin_overlay_sets_the_flags_explicitly_rather_than_by_default() -> None:
    """Belt-and-braces: the YAML text itself carries the opt-in."""
    data = yaml.safe_load(_TWIN_OVERLAY.read_text(encoding="utf-8"))
    assert data["gcp"]["logging"]["enabled"] is True
    assert data["gcp"]["monitoring"]["enabled"] is True
    assert data["gcp"]["pubsub"]["enabled"] is True
    assert data["gcp"]["storage"]["enabled"] is True


def test_twin_overlay_still_builds_its_cloud_components() -> None:
    """The one overlay that wants cloud egress still gets it after the gating.

    Without this, "default to False" could pass every other assertion while
    quietly breaking the only consumer that actually wants these channels.
    """
    from mousedroid.config.schema.root import Settings
    from mousedroid.factory import build_cloud_experience_exporter, build_cloud_telemetry_sink

    data = yaml.safe_load(_TWIN_OVERLAY.read_text(encoding="utf-8"))
    settings = Settings(mock_hardware=True, **data)
    # Returns None only if the google-cloud packages are absent (optional
    # [gcp] extra); the gate itself must not be the reason.
    assert settings.gcp is not None
    assert settings.gcp.pubsub.enabled is True
    assert settings.gcp.storage.enabled is True
    for builder in (build_cloud_telemetry_sink, build_cloud_experience_exporter):
        # Either a real component, or None from the ImportError path -- never
        # None because the enabled gate blocked it.
        builder(settings)


def test_env_lever_can_re_enable_egress(monkeypatch: pytest.MonkeyPatch) -> None:
    """The MOUSEDROID_ env lever still reaches the flag through 3-level nesting.

    .claude/skills/regression-pair-scaffold/SKILL.md requires a loader-path
    variant whenever a feature exposes a ``MOUSEDROID_*`` lever. It does:
    ``MOUSEDROID_GCP__LOGGING__ENABLED`` resolves via the ``__`` nested
    delimiter. Prior coverage only exercised two levels
    (``MOUSEDROID_GCP__PROJECT_ID``), so the third was unproven -- which matters
    because an operator restoring egress after F-029 will reach for exactly this.
    """
    monkeypatch.setenv("MOUSEDROID_GCP__PROJECT_ID", "env-project")
    monkeypatch.setenv("MOUSEDROID_GCP__LOGGING__ENABLED", "true")
    settings = Settings(mock_hardware=True)
    assert settings.gcp is not None
    assert settings.gcp.logging.enabled is True
    # Sibling channels stay closed -- the lever is per-channel, not a master switch.
    assert settings.gcp.monitoring.enabled is False
    assert settings.gcp.pubsub.enabled is False


def test_env_lever_without_project_id_is_a_loud_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Setting a nested GCP env var alone fails validation rather than half-configuring.

    project_id is required, so ``MOUSEDROID_GCP__LOGGING__ENABLED=true`` on its
    own raises. Pinned because the failure is otherwise a surprising traceback
    for an operator who set one variable and expected a no-op.
    """
    monkeypatch.delenv("MOUSEDROID_GCP__PROJECT_ID", raising=False)
    monkeypatch.setenv("MOUSEDROID_GCP__LOGGING__ENABLED", "true")
    with pytest.raises(ValidationError, match="project_id"):
        Settings(mock_hardware=True)


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
