"""Automated Quality Assurance (AQA) — schema + protocol hygiene for PR #106.

PR #106 (Jetson + USB-C rover smoke validation) is documented in CLAUDE.md
with explicit "non-negotiable contracts" language — the same weight given to
every other feature that gets a regression AQA + backwards-compat pair (F-025,
PR104, PR115, growth, on-device-learning, alayaworld, dashboard, portfolio-
reframe). This file was the one gap in that convention; it pins the three
contracts CLAUDE.md calls out for PR #106:

1. ``USBCDiscoveryConfig.enabled`` is a documented, default-``False`` gate.
2. ``_require_endpoints_when_enabled`` rejects ``enabled=True`` with an empty
   ``required_endpoints`` list at YAML-load time — a misconfigured gate never
   silently passes.
3. The boot-race guard: ``enumerate_usbc_devices``/``resolve_endpoint`` never
   raise when ``by_id_root`` doesn't exist yet (pre-udev container startup) —
   they surface a structured MISSING/None result instead of an uncaught
   ``FileNotFoundError`` crashing the smoke harness.

Naming + behaviour drift here is the most insidious failure mode: tests still
pass, but a refactor weakens the contract silently. AQA catches that.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError
from pydantic.fields import FieldInfo

from mousedroid.config.schema import ESP32Config, USBCDiscoveryConfig, USBCEndpointSpec
from mousedroid.diagnostics.usbc import enumerate_usbc_devices, resolve_endpoint

# ---------------------------------------------------------------------------
# Schema hygiene — USBCDiscoveryConfig.enabled is documented + defaults False
# ---------------------------------------------------------------------------


def test_usbc_discovery_enabled_has_description() -> None:
    """``USBCDiscoveryConfig.enabled`` carries a non-empty Pydantic description."""
    info: FieldInfo = USBCDiscoveryConfig.model_fields["enabled"]
    assert info.description
    assert len(info.description) > 10, info.description


def test_usbc_discovery_enabled_default_is_false() -> None:
    """Defaults ``False`` so pre-existing overlays without a ``usbc_discovery:``
    block load byte-identically — the master switch that "keeps default YAML
    inert" per the field's own docstring."""
    info: FieldInfo = USBCDiscoveryConfig.model_fields["enabled"]
    assert info.default is False


# ---------------------------------------------------------------------------
# _require_endpoints_when_enabled — a misconfigured gate never silently passes
# ---------------------------------------------------------------------------


def test_enabled_with_empty_required_endpoints_raises_at_load() -> None:
    """``enabled=True`` with an empty ``required_endpoints`` list is rejected
    at YAML-load time, not silently accepted as a no-op gate."""
    with pytest.raises(ValidationError, match=r"usbc_discovery\.enabled=true"):
        USBCDiscoveryConfig(enabled=True, required_endpoints=[])


def test_enabled_with_at_least_one_endpoint_loads_cleanly() -> None:
    """The same gate, satisfied — proves the validator isn't just always-raise."""
    cfg = USBCDiscoveryConfig(
        enabled=True,
        required_endpoints=[
            USBCEndpointSpec(name="rover_esp32", by_id_glob="*CP2102N*-if00-port0")
        ],
    )
    assert cfg.enabled is True
    assert len(cfg.required_endpoints) == 1


def test_disabled_with_empty_required_endpoints_loads_cleanly() -> None:
    """The validator only fires when ``enabled=True`` — the default-disabled,
    empty-list state (every pre-discovery overlay) must load unchanged."""
    cfg = USBCDiscoveryConfig()
    assert cfg.enabled is False
    assert cfg.required_endpoints == []


# ---------------------------------------------------------------------------
# ESP32Config.smoke_test_velocity_mps — ge=0, not gt=0 (permits a safe-bench 0)
# ---------------------------------------------------------------------------


def test_smoke_test_velocity_mps_accepts_zero() -> None:
    """``ge=0`` (not ``gt=0``) lets operators express a permanent zero-motion
    safe-bench config — the runtime ``allow_motion`` gate stays authoritative
    regardless of this setpoint, but the field itself must accept 0.0."""
    cfg = ESP32Config(smoke_test_velocity_mps=0.0)
    assert cfg.smoke_test_velocity_mps == 0.0


def test_smoke_test_velocity_mps_rejects_negative() -> None:
    """Still bounded below — a negative setpoint is a config typo, not a
    valid "reverse at smoke-test time" request."""
    with pytest.raises(ValidationError, match="smoke_test_velocity_mps"):
        ESP32Config(smoke_test_velocity_mps=-0.01)


# ---------------------------------------------------------------------------
# Boot-race guard — enumerate_usbc_devices / resolve_endpoint never raise
# ---------------------------------------------------------------------------


def _enabled_cfg(by_id_root: Path) -> USBCDiscoveryConfig:
    return USBCDiscoveryConfig(
        enabled=True,
        by_id_root=by_id_root,
        required_endpoints=[
            USBCEndpointSpec(name="rover_esp32", by_id_glob="*CP2102N*-if00-port0")
        ],
    )


def test_enumerate_usbc_devices_missing_by_id_root_returns_missing_not_raise(
    tmp_path: Path,
) -> None:
    """A pre-udev boot race (``by_id_root`` doesn't exist yet) surfaces every
    required endpoint as MISSING — never an uncaught ``FileNotFoundError``
    from ``Path.glob`` crashing the smoke harness."""
    missing_root = tmp_path / "does-not-exist-yet"
    assert not missing_root.exists()

    results = enumerate_usbc_devices(_enabled_cfg(missing_root))

    assert "rover_esp32" in results
    assert results["rover_esp32"].resolved_path is None
    assert results["rover_esp32"].status.value in {"missing", "warn"}


def test_resolve_endpoint_missing_by_id_root_returns_none_not_raise(tmp_path: Path) -> None:
    """Same boot-race guard for the single-endpoint resolver used by the
    factory override chain."""
    missing_root = tmp_path / "also-does-not-exist"
    assert not missing_root.exists()

    resolved = resolve_endpoint(_enabled_cfg(missing_root), "rover_esp32")

    assert resolved is None
