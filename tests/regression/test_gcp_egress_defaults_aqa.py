"""AQA pins for F-029 — GCP egress sub-configs default OFF.

``Settings.gcp`` is ``Optional`` and defaults ``None``, so nothing egresses on
a stock config. The hazard this pins is narrower and easier to miss: an
operator who adds a minimal ``gcp:`` block for one reason (Firestore, storage)
used to silently enable **two** egress channels they never named, because
``GCPLoggingConfig.enabled`` and ``GCPMonitoringConfig.enabled`` defaulted
``True`` while ``GCPFirestoreConfig.enabled`` defaulted ``False``.

Assertions read the ``FieldInfo`` off ``model_fields`` rather than round-tripping
through ``model_validate``, per .claude/skills/test-tier-mirror/SKILL.md: a
refactor that swapped ``Field(...)`` for a property override would slip past a
value-only check.
"""

from __future__ import annotations

import pytest

from mousedroid.config.schema.gcp_cloud import (
    GCPConfig,
    GCPFirestoreConfig,
    GCPLoggingConfig,
    GCPMonitoringConfig,
)

# The repo convention for a description that actually explains itself; mirrors
# _MIN_DESCRIPTION_CHARS in tests/regression/test_f025_aqa.py.
_MIN_DESCRIPTION_CHARS = 20

_EGRESS_MODELS = (
    (GCPLoggingConfig, "logging"),
    (GCPMonitoringConfig, "monitoring"),
    (GCPFirestoreConfig, "firestore"),
)


@pytest.mark.parametrize(("model", "label"), _EGRESS_MODELS)
def test_egress_enabled_defaults_false(model: type, label: str) -> None:
    """Every off-device egress channel is opt-in, never opt-out."""
    info = model.model_fields["enabled"]
    assert info.default is False, (
        f"GCP {label} egress must default OFF — a default-ON egress channel is "
        "a silent behaviour change for anyone who adds a gcp: block (CHARTER §6: "
        "new capabilities are additive and opt-in)"
    )


@pytest.mark.parametrize(("model", "label"), _EGRESS_MODELS)
def test_egress_enabled_description_is_substantive(model: type, label: str) -> None:
    """The description carries the rationale, not just a restatement."""
    info = model.model_fields["enabled"]
    assert info.description, f"GCP {label} enabled field has no description"
    assert len(info.description) > _MIN_DESCRIPTION_CHARS


def test_partial_gcp_block_opens_no_egress_channel() -> None:
    """The scenario that actually bit: a minimal gcp: block added for one reason.

    Asserting "nothing egresses by default" would be vacuously true, since
    ``Settings.gcp`` is None by default. This asserts the real hazard.
    """
    cfg = GCPConfig(project_id="some-project")
    assert cfg.logging.enabled is False
    assert cfg.monitoring.enabled is False
    assert cfg.firestore.enabled is False


def test_explicit_opt_in_still_works() -> None:
    """Proves the flag is not always-False — the default is a default, not a pin."""
    cfg = GCPConfig.model_validate(
        {
            "project_id": "some-project",
            "logging": {"enabled": True},
            "monitoring": {"enabled": True},
        }
    )
    assert cfg.logging.enabled is True
    assert cfg.monitoring.enabled is True
