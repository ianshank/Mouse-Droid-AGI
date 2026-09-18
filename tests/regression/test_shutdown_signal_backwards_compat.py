"""S-1 backwards-compatibility regression tests.

Pins CLAUDE.md invariant 6: new config fields MUST carry defaults, and
existing YAML must load unchanged after a ``git pull``. Every shipped
overlay in ``config/`` predates ``loop.shutdown_grace_s``, so none of them
mention it — they must all still parse, and pick up the default.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mousedroid.config.loader import load_settings
from mousedroid.config.schema import Settings

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _REPO_ROOT / "config"

#: Every shipped overlay, exercised through the real loader.
#:
#: These are *fragments* merged on top of ``default.yaml`` by
#: ``load_settings``, not standalone Settings documents — several omit
#: required blocks and only validate once the base is underneath them. So
#: this list is fed to the loader rather than to ``model_validate``.
#: ``baselines.yaml`` is excluded: it is a metrics baseline document, not a
#: Settings overlay. ``*.example`` files are templates and are not globbed.
_SHIPPED_OVERLAYS = sorted(
    p for p in _CONFIG_DIR.glob("*.yaml") if p.name not in {"baselines.yaml", "default.yaml"}
)


def test_shutdown_grace_s_defaults_when_absent() -> None:
    """A pre-S-1 config has no ``shutdown_grace_s`` key at all."""
    cfg = Settings.model_validate({"mock_hardware": True})
    assert cfg.loop.shutdown_grace_s == 2.0


def test_legacy_yaml_without_shutdown_grace_loads_unchanged() -> None:
    """A minimal legacy YAML loads cleanly and unrelated fields survive.

    Proves the new field is additive, not entangled with the loop timing
    values that already shipped.
    """
    legacy_yaml = """
    mock_hardware: true
    platform: mouse_droid
    loop:
      control_hz: 25.0
      tick_timeout_s: 0.5
    """
    data = yaml.safe_load(legacy_yaml)
    cfg = Settings.model_validate(data)

    assert cfg.loop.shutdown_grace_s == 2.0
    assert cfg.loop.control_hz == 25.0
    assert cfg.loop.tick_timeout_s == 0.5


def test_shutdown_grace_s_round_trips_when_present() -> None:
    """An operator override reaches the field rather than being dropped."""
    cfg = Settings.model_validate({"mock_hardware": True, "loop": {"shutdown_grace_s": 6.0}})
    assert cfg.loop.shutdown_grace_s == 6.0


def test_default_yaml_still_loads_clean() -> None:
    """The committed base config loads with the new field absent."""
    cfg = load_settings()
    assert cfg.loop.shutdown_grace_s == 2.0


@pytest.mark.parametrize("overlay", _SHIPPED_OVERLAYS, ids=lambda p: p.name)
def test_shipped_overlay_still_loads(overlay: Path) -> None:
    """Every committed overlay still merges cleanly with the new field absent."""
    cfg = load_settings(overlay)
    assert cfg.loop.shutdown_grace_s > 0


def test_no_shipped_overlay_mentions_shutdown_grace() -> None:
    """Guards the premise of the tests above: they all predate the field.

    If an overlay later sets it, this fails and forces the parametrised
    test above to be re-read rather than quietly proving nothing.
    """
    setters = [
        p.name for p in _SHIPPED_OVERLAYS if "shutdown_grace_s" in p.read_text(encoding="utf-8")
    ]
    assert setters == []


def test_production_overlay_grace_fits_inside_docker_stop_window() -> None:
    """The rover's own overlay must still halt before SIGKILL lands.

    ``docker stop`` sends SIGTERM then SIGKILL after its grace period.
    A ``shutdown_grace_s`` at or above that window would let the loop be
    killed before the escalation ever fired.
    """
    cfg = load_settings(_CONFIG_DIR / "jetson_production.yaml")

    docker_default_stop_window_s = 10.0
    assert cfg.loop.shutdown_grace_s < docker_default_stop_window_s


def test_esp32_command_set_default_is_unchanged() -> None:
    """S-1 did NOT migrate the wire protocol — that stays a separate decision.

    Pinned so the new ``esp32_heartbeat_unavailable`` warning cannot be
    "fixed" by silently flipping the default command set, which would
    change every legacy deployment's firmware protocol on upgrade.
    """
    cfg = Settings.model_validate({"mock_hardware": True})
    assert cfg.esp32.command_set == "legacy"
    assert cfg.esp32.heartbeat_enabled is True
