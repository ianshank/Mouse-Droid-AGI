# tests/regression/test_f059_aqa.py
"""AQA pins for F-059 (rover gate closeout). Each test names the round-3 finding it pins."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from mousedroid.config.loader import load_settings
from mousedroid.validation.preflight import PreflightStatus, _check_esp32

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _REPO_ROOT / "config"
_PROD = _CONFIG_DIR / "jetson_production.yaml"


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


def test_trend_unit_never_runs_exclusive_device_checks() -> None:
    """R-11: the on-timer preflight must never open the serial port or the camera.

    The orchestrator container owns those devices while it runs.
    """
    text = (_REPO_ROOT / "scripts" / "mousedroid-trend.service").read_text(encoding="utf-8")
    match = re.search(r"^Environment=MOUSEDROID_TREND_CHECKS=(\S+)$", text, re.M)
    assert match, "default check list must be an Environment= line"
    assert not {"esp32", "lidar", "camera"} & set(match.group(1).split(","))
