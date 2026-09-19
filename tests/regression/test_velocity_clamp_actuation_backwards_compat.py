"""Backwards-compat pin for the actuation-path finite guards (D-17..D-19).

Exercises the real codec entry points end to end, not just the helper, so
the pin covers what actually goes on the wire.

Unchanged:
  * Every finite velocity produces the identical PWM / physical-unit frame.
  * Frame shape, keys and command-type constants are untouched.
  * ``clamp``'s signature is unchanged, and its five call sites still bound
    out-of-range values to ``lo``/``hi`` exactly as before.

Deliberately changed (this is the defect):
  * A non-finite velocity used to resolve to the *upper bound* and be
    transmitted as maximum commanded speed. It now resolves to ``0.0`` on
    every path, so the failure direction is no motion.

Both codecs are covered because ``comms/_utils.py::clamp`` is shared by
them, and ``NEXT_STEPS.md`` item 3 is about to flip the shipped
``command_set`` from ``legacy`` to ``waveshare_stock``.
"""

from __future__ import annotations

import pytest

from mousedroid.comms._utils import (
    ESP32_CMD_TYPE_VELOCITY,
    MAX_PWM,
    build_velocity_cmd,
)
from mousedroid.comms.command_set import WaveshareStockCodec
from mousedroid.config.schema import Settings

_NON_FINITE = [float("nan"), float("inf"), float("-inf")]


@pytest.fixture
def esp32_cfg():  # type: ignore[no-untyped-def]
    return Settings(mock_hardware=True).esp32


# --------------------------------------------------------------------------
# Legacy codec -> PWM ints
# --------------------------------------------------------------------------


def test_legacy_finite_frame_is_unchanged(esp32_cfg) -> None:  # type: ignore[no-untyped-def]
    """A benign command produces the same frame it always did."""
    cmd = build_velocity_cmd(esp32_cfg.max_velocity_mps, 0.0, 0.0, esp32_cfg)
    assert cmd["T"] == ESP32_CMD_TYPE_VELOCITY
    assert cmd["vx"] == MAX_PWM
    assert cmd["vy"] == 0
    assert cmd["omega"] == 0


def test_legacy_out_of_range_still_saturates(esp32_cfg) -> None:  # type: ignore[no-untyped-def]
    """Out-of-range input still clamps to full scale — that part was correct."""
    cmd = build_velocity_cmd(esp32_cfg.max_velocity_mps * 99, 0.0, 0.0, esp32_cfg)
    assert cmd["vx"] == MAX_PWM
    cmd = build_velocity_cmd(-esp32_cfg.max_velocity_mps * 99, 0.0, 0.0, esp32_cfg)
    assert cmd["vx"] == -MAX_PWM


@pytest.mark.parametrize("bad", _NON_FINITE)
def test_legacy_non_finite_vx_is_now_zero_pwm(bad: float, esp32_cfg) -> None:  # type: ignore[no-untyped-def]
    """Was ``MAX_PWM`` (full-scale PWM on the wire, no exception)."""
    cmd = build_velocity_cmd(bad, 0.0, 0.0, esp32_cfg)
    assert cmd["vx"] == 0
    assert cmd["vx"] != MAX_PWM


@pytest.mark.parametrize("bad", _NON_FINITE)
def test_legacy_non_finite_omega_is_now_zero_pwm(bad: float, esp32_cfg) -> None:  # type: ignore[no-untyped-def]
    cmd = build_velocity_cmd(0.0, 0.0, bad, esp32_cfg)
    assert cmd["omega"] == 0


def test_legacy_non_finite_does_not_raise(esp32_cfg) -> None:  # type: ignore[no-untyped-def]
    """It never raised before either — it silently sent full scale.

    Worth pinning: a reader might assume ``int(nan)`` would have raised
    ValueError and failed safe. It could not, because the clamp resolved
    the NaN to ``1.0`` before ``int()`` ever saw it.
    """
    cmd = build_velocity_cmd(float("nan"), float("nan"), float("nan"), esp32_cfg)
    assert cmd == {"T": ESP32_CMD_TYPE_VELOCITY, "vx": 0, "vy": 0, "omega": 0}


# --------------------------------------------------------------------------
# Stock codec -> physical units
# --------------------------------------------------------------------------


def test_stock_finite_frame_is_unchanged(esp32_cfg) -> None:  # type: ignore[no-untyped-def]
    codec = WaveshareStockCodec()
    cmd = codec.build_velocity(0.1, 0.0, 0.2, esp32_cfg)
    assert cmd["X"] == 0.1
    assert cmd["Z"] == 0.2
    assert "Y" not in cmd  # no lateral axis in CMD_ROS_CTRL


def test_stock_out_of_range_still_saturates(esp32_cfg) -> None:  # type: ignore[no-untyped-def]
    codec = WaveshareStockCodec()
    cmd = codec.build_velocity(999.0, 0.0, 999.0, esp32_cfg)
    assert cmd["X"] == esp32_cfg.max_velocity_mps
    assert cmd["Z"] == esp32_cfg.max_omega_rads


@pytest.mark.parametrize("bad", _NON_FINITE)
def test_stock_non_finite_is_now_zero(bad: float, esp32_cfg) -> None:  # type: ignore[no-untyped-def]
    """Was ``max_velocity_mps`` / ``max_omega_rads`` in the frame."""
    codec = WaveshareStockCodec()
    cmd = codec.build_velocity(bad, 0.0, bad, esp32_cfg)
    assert cmd["X"] == 0.0
    assert cmd["Z"] == 0.0
    assert cmd["X"] != esp32_cfg.max_velocity_mps


def test_stock_stop_frame_is_unchanged(esp32_cfg) -> None:  # type: ignore[no-untyped-def]
    codec = WaveshareStockCodec()
    assert codec.build_stop()["X"] == 0.0
    assert codec.build_stop()["Z"] == 0.0
