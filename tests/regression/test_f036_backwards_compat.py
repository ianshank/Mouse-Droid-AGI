"""F-036 backwards-compatibility regression tests.

Pins CLAUDE.md invariant 9 for a change that adds *no* schema field:
existing YAML still loads, the legacy codec still zeros IMU fields, and
the default ``command_set="legacy"`` path still reports odometry heading.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mousedroid.comms._utils import parse_encoder_reading
from mousedroid.comms.command_set import LEGACY_CODEC, WAVESHARE_STOCK_CODEC
from mousedroid.config.schema import Settings

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_command_set_default_still_legacy() -> None:
    cfg = Settings.model_validate({"mock_hardware": True})
    assert cfg.esp32.command_set == "legacy"


def test_legacy_codec_parse_zeros_imu_even_when_rpy_keys_are_present() -> None:
    """Legacy wire JSON uses ``lv``/``rv``/``h``, not stock ``r``/``p``/``y``.

    A confused frame that *also* carries stock IMU keys must not flip
    ``imu_valid`` — that would steal heading from the odometry slot.
    """
    reading = LEGACY_CODEC.parse_encoders(
        {"lv": 0.4, "rv": 0.5, "h": 0.9, "r": 0.1, "p": 0.2, "y": 1.5}
    )
    assert reading.heading_rad == pytest.approx(0.9)
    assert reading.imu_valid is False
    assert reading.yaw_rad == 0.0
    assert reading.heading_for_motor() == pytest.approx(0.9)


def test_legacy_parse_encoder_reading_helper_stays_imu_inert() -> None:
    reading = parse_encoder_reading({"lv": 0.1, "h": 0.4})
    assert reading.imu_valid is False
    assert reading.heading_for_motor() == pytest.approx(0.4)


def test_stock_wrong_frame_type_does_not_mark_imu_valid() -> None:
    reading = WAVESHARE_STOCK_CODEC.parse_encoders({"T": 1002, "y": 1.57})
    assert reading.imu_valid is False
    assert reading.heading_for_motor() == 0.0


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
    assert cfg.esp32.command_set == "legacy"
