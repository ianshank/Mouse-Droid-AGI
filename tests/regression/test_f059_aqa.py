# tests/regression/test_f059_aqa.py
"""AQA pins for F-059 (rover gate closeout).

Each test names the finding it pins: a round-3 R-id, or the PR review that raised it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from mousedroid.config.loader import load_settings
from mousedroid.config.schema.hardware import WAVESHARE_STOCK_BAUD
from mousedroid.validation import runtime
from mousedroid.validation.preflight import PreflightStatus, _check_esp32
from mousedroid.validation.runtime import Esp32ProbeResult

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _REPO_ROOT / "config"
_PROD = _CONFIG_DIR / "jetson_production.yaml"

#: One ``MOUSEDROID_ESP32__NAME=value`` assignment, as an operator would copy it.
_ENV_ASSIGNMENT = re.compile(r"(MOUSEDROID_ESP32__[A-Z_]+)=(\w+)")


def _features() -> dict[str, dict]:
    raw = yaml.safe_load((_REPO_ROOT / "features.yaml").read_text(encoding="utf-8"))
    return {f["id"]: f for f in raw["features"]}


def test_f059_is_reserved_and_critical() -> None:
    f = _features()["F-059"]
    assert f["priority"] == "critical"
    assert f["tier"] == "fast"
    assert f["validation_command"] == "bash scripts/validations/F-059.sh"


def test_f059_edge_points_the_right_way() -> None:
    """R3-1: F-059 makes F-008 passable, so F-008 waits on F-059 — never the reverse."""
    feats = _features()
    assert "F-008" not in feats["F-059"]["depends_on"]
    assert "F-059" in feats["F-008"]["depends_on"]


def test_select_next_returns_f059_while_it_is_in_progress() -> None:
    """F-059 wins the in-progress pick by priority (critical beats F-050 high / F-051 medium);
    the F-050/F-051 deferral is catalog honesty (EX-3), not the lever.
    """
    catalog = _features()
    if catalog["F-059"]["status"] != "in_progress":
        pytest.skip("F-059 is no longer in progress; the edge pin is the durable guarantee")
    if any(
        f["priority"] == "critical" and f["status"] == "in_progress"
        for fid, f in catalog.items()
        if fid != "F-059"
    ):
        pytest.skip("another critical feature is in progress; the selector tie-breaks on id")
    from mousedroid.harness import spec

    feats = spec.load_features(str(_REPO_ROOT / "features.yaml"))
    sel = spec.select_next(feats)
    assert sel.feature is not None, sel
    assert sel.feature["id"] == "F-059", sel


async def test_preflight_esp32_warns_rather_than_ok_when_the_board_is_mocked() -> None:
    """R-4 / R3-2 (S-4): the shipped production overlay mocks the ESP32.

    ``config/jetson_production.yaml`` sets ``esp32.enabled: false``, so the factory
    hands back ``MockESP32Driver``; preflight used to report ``OK driver=...`` for
    that. It must be WARN (aggregate DEGRADED) -- there is no SKIP member (R-4).
    """
    cfg = load_settings(_PROD, config_dir=_CONFIG_DIR)
    cfg.mock_hardware = False
    assert (await _check_esp32(cfg)).status is PreflightStatus.WARN


async def test_preflight_esp32_disabled_remedy_works_when_followed_literally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Copilot review on #256: the WARN's own env remedy, applied to the shipped overlay.

    The remedy is operator-facing text, so it is executed rather than string-matched: every
    ``MOUSEDROID_ESP32__*=value`` it prints is set in the environment, the production overlay
    is reloaded, and the check must now reach the probe (stubbed -- no wire) and report OK.
    The stock baud is asserted too: a stock board read at the schema's legacy 1 Mbaud looks
    dead (F-025 vendor audit R2), so the remedy is only complete if the config lands on the
    stock rate -- whether the overlay pins it or the command-set coupling derives it.
    """
    cfg = load_settings(_PROD, config_dir=_CONFIG_DIR)
    cfg.mock_hardware = False
    warned = await _check_esp32(cfg)
    assert warned.status is PreflightStatus.WARN

    assignments = _ENV_ASSIGNMENT.findall(warned.detail)
    assert {name for name, _ in assignments} == {
        "MOUSEDROID_ESP32__ENABLED",
        "MOUSEDROID_ESP32__COMMAND_SET",
    }, warned.detail
    for name, value in assignments:
        monkeypatch.setenv(name, value)
    followed = load_settings(_PROD, config_dir=_CONFIG_DIR)
    followed.mock_hardware = False
    assert followed.esp32.enabled is True
    assert followed.esp32.command_set == "waveshare_stock"
    assert followed.esp32.serial_baud == WAVESHARE_STOCK_BAUD

    probed: list[object] = []

    async def alive(probe_cfg: object) -> Esp32ProbeResult:
        probed.append(probe_cfg)
        return Esp32ProbeResult(True, 1001, 11.9, 0.1, "FEEDBACK_BASE_INFO after 1 poll(s)")

    monkeypatch.setattr(runtime, "collect_esp32_probe", alive)
    result = await _check_esp32(followed)
    assert result.status is PreflightStatus.OK
    assert probed == [followed]  # it reached the probe, not a config-level WARN


def _explicit_cfg(command_set: str):
    """Real-hardware settings with ``enabled`` lifted and ``command_set`` stated outright.

    Nothing is read from ``config/*.yaml``: the two pins below are about the levers the
    remedy names, and which command set the shipped overlay carries is another PR's to change.
    """
    from mousedroid.config.schema import Settings

    return Settings.model_validate(
        {
            "mock_hardware": False,
            "ultrasonic": {"trigger_pin": 17, "echo_pin": 27},
            "esp32": {"enabled": True, "command_set": command_set},
        }
    )


async def test_preflight_esp32_enabling_alone_only_swaps_the_warn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Why the remedy names two variables (Copilot review on #256).

    ``esp32.enabled=true`` on the ``legacy`` command set lands on the "no probe" WARN, so
    lifting ``enabled`` alone only swaps one WARN for another and never reaches the board.
    """

    async def never(probe_cfg: object) -> Esp32ProbeResult:
        msg = "probe must not run under the legacy command set"
        raise RuntimeError(msg)

    monkeypatch.setattr(runtime, "collect_esp32_probe", never)
    result = await _check_esp32(_explicit_cfg("legacy"))
    assert result.status is PreflightStatus.WARN
    assert "no probe" in result.detail
    assert "MOUSEDROID_ESP32__COMMAND_SET=waveshare_stock" in result.detail  # the missing lever


async def test_preflight_esp32_enabled_on_the_stock_command_set_reaches_the_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other half of the remedy: the pair is enough (Copilot review on #256).

    ``enabled=true`` plus ``command_set='waveshare_stock'`` is what the remedy's two
    assignments produce. The check must reach the probe (stubbed -- no wire) and report its
    verdict instead of another config-level WARN.
    """
    cfg = _explicit_cfg("waveshare_stock")
    probed: list[object] = []

    async def alive(probe_cfg: object) -> Esp32ProbeResult:
        probed.append(probe_cfg)
        return Esp32ProbeResult(True, 1001, 11.9, 0.1, "FEEDBACK_BASE_INFO after 1 poll(s)")

    monkeypatch.setattr(runtime, "collect_esp32_probe", alive)
    result = await _check_esp32(cfg)
    assert result.status is PreflightStatus.OK
    assert probed == [cfg]  # it reached the probe, not a config-level WARN


def test_trend_unit_never_runs_exclusive_device_checks() -> None:
    """R-11: the on-timer preflight must never open the serial port or the camera.

    The orchestrator container owns those devices while it runs. systemd honours the LAST
    ``Environment=`` assignment of a variable, so every such line is checked, not the first.
    """
    text = (_REPO_ROOT / "scripts" / "mousedroid-trend.service").read_text(encoding="utf-8")
    declared = re.findall(r"^Environment=MOUSEDROID_TREND_CHECKS=(\S+)$", text, re.M)
    assert declared, "default check list must be an Environment= line"
    listed = {name for value in declared for name in value.split(",")}
    assert not {"esp32", "lidar", "camera"} & listed
