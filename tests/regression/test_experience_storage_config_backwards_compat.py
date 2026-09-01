"""Backwards-compat pins for ExperienceConfig's storage-diagnostics fields.

``nvme_device``, ``nvme_partition``, ``diagnostics_subprocess_timeout_s``, and
``ssd_mount_override_env_var`` (``config/schema/misc.py``) already carry
schema descriptions and behavioral unit-test coverage (they're read by
``validation/runtime/_storage.py`` and ``_shared.py``), but — per an audit —
had no regression pin proving a YAML file that predates them, or one that
doesn't set them, still loads unchanged. That's the actual backwards-compat
guarantee ``ExperienceConfig`` (and CLAUDE.md invariant #6) promises: every
new field carries a default, so old config never breaks on a `git pull`.
"""

from __future__ import annotations

from mousedroid.config.schema.misc import ExperienceConfig
from mousedroid.config.schema.root import Settings


def test_experience_config_storage_fields_have_documented_defaults() -> None:
    """Absent overrides, the four fields resolve to their documented defaults."""
    cfg = ExperienceConfig()
    assert cfg.nvme_device == "/dev/nvme0n1"
    assert cfg.nvme_partition == "/dev/nvme0n1p1"
    assert cfg.diagnostics_subprocess_timeout_s == 10.0
    assert cfg.ssd_mount_override_env_var == "MOUSEDROID_SSD_MOUNT"


def test_settings_without_experience_overrides_still_loads() -> None:
    """A minimal Settings (no `experience:` block at all) is unaffected."""
    s = Settings.model_validate({"mock_hardware": True})
    assert s.experience is not None
    assert s.experience.nvme_device == "/dev/nvme0n1"
    assert s.experience.nvme_partition == "/dev/nvme0n1p1"
    assert s.experience.diagnostics_subprocess_timeout_s == 10.0


def test_partial_experience_block_only_overrides_what_it_names() -> None:
    """A YAML file overriding one field keeps the schema defaults for the rest."""
    s = Settings.model_validate(
        {"mock_hardware": True, "experience": {"nvme_device": "/dev/nvme1n1"}}
    )
    assert s.experience.nvme_device == "/dev/nvme1n1"
    assert s.experience.nvme_partition == "/dev/nvme0n1p1"
    assert s.experience.diagnostics_subprocess_timeout_s == 10.0
    assert s.experience.ssd_mount_override_env_var == "MOUSEDROID_SSD_MOUNT"
