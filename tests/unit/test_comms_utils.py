"""Tests for ``comms/_utils`` helper functions.

Covers ``build_velocity_cmd`` and ``parse_encoder_reading``, which are the
single-source-of-truth implementations used by both serial and WiFi drivers
via ``BaseESP32Driver``.
"""

from __future__ import annotations

import pytest

from mousedroid.comms._utils import (
    ESP32_CMD_TYPE_BATTERY,
    ESP32_CMD_TYPE_STOP,
    ESP32_CMD_TYPE_VELOCITY,
    MAX_PWM,
    build_velocity_cmd,
    clamp,
    parse_encoder_reading,
)
from mousedroid.config.schema import ESP32Config

# ---------------------------------------------------------------------------
# clamp (existing, keep for regression)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "lo", "hi", "expected"),
    [
        (0.5, 0.0, 1.0, 0.5),  # within range
        (-0.5, 0.0, 1.0, 0.0),  # below lower bound
        (1.5, 0.0, 1.0, 1.0),  # above upper bound
        (-1.5, -1.0, 1.0, -1.0),  # negative lower bound
        (3.0, -1.0, 1.0, 1.0),  # positive upper bound
        (0.0, 0.0, 0.0, 0.0),  # degenerate range
    ],
)
def test_clamp(value: float, lo: float, hi: float, expected: float) -> None:
    assert clamp(value, lo, hi) == expected


# ---------------------------------------------------------------------------
# build_velocity_cmd
# ---------------------------------------------------------------------------


def _cfg(max_vel: float = 1.0, max_omega: float = 1.0) -> ESP32Config:
    return ESP32Config(max_velocity_mps=max_vel, max_omega_rads=max_omega)


def test_build_velocity_cmd_type_is_velocity():
    cmd = build_velocity_cmd(0.0, 0.0, 0.0, _cfg())
    assert cmd["T"] == ESP32_CMD_TYPE_VELOCITY


def test_build_velocity_cmd_zero_velocity():
    cmd = build_velocity_cmd(0.0, 0.0, 0.0, _cfg())
    assert cmd["vx"] == 0
    assert cmd["vy"] == 0
    assert cmd["omega"] == 0


def test_build_velocity_cmd_full_forward():
    cmd = build_velocity_cmd(1.0, 0.0, 0.0, _cfg(max_vel=1.0))
    assert cmd["vx"] == MAX_PWM
    assert cmd["vy"] == 0
    assert cmd["omega"] == 0


def test_build_velocity_cmd_full_reverse():
    cmd = build_velocity_cmd(-1.0, 0.0, 0.0, _cfg(max_vel=1.0))
    assert cmd["vx"] == -MAX_PWM


def test_build_velocity_cmd_half_speed():
    cmd = build_velocity_cmd(0.5, 0.0, 0.0, _cfg(max_vel=1.0))
    assert cmd["vx"] == 127  # int(0.5 * 255) = 127


def test_build_velocity_cmd_clamps_above_max():
    cmd = build_velocity_cmd(100.0, 0.0, 0.0, _cfg(max_vel=1.0))
    assert cmd["vx"] == MAX_PWM


def test_build_velocity_cmd_clamps_below_neg_max():
    cmd = build_velocity_cmd(-100.0, 0.0, 0.0, _cfg(max_vel=1.0))
    assert cmd["vx"] == -MAX_PWM


def test_build_velocity_cmd_omega_scaling():
    cmd = build_velocity_cmd(
        0.0,
        0.0,
        0.5,
        _cfg(max_vel=1.0, max_omega=1.0),
    )
    assert cmd["omega"] == 127


def test_build_velocity_cmd_custom_max_vel():
    # At 1.0 m/s with max_vel = 2.0 → ratio = 0.5 → 127 PWM
    cmd = build_velocity_cmd(1.0, 0.0, 0.0, _cfg(max_vel=2.0))
    assert cmd["vx"] == 127


def test_build_velocity_cmd_returns_int_values():
    cmd = build_velocity_cmd(0.3, -0.3, 0.6, _cfg())
    assert isinstance(cmd["vx"], int)
    assert isinstance(cmd["vy"], int)
    assert isinstance(cmd["omega"], int)


def test_build_velocity_cmd_all_axes():
    cmd = build_velocity_cmd(0.5, -0.5, 1.0, _cfg(max_vel=1.0, max_omega=1.0))
    assert cmd["T"] == ESP32_CMD_TYPE_VELOCITY
    assert cmd["vx"] == 127
    assert cmd["vy"] == -127
    assert cmd["omega"] == MAX_PWM


# ---------------------------------------------------------------------------
# parse_encoder_reading
# ---------------------------------------------------------------------------


def test_parse_encoder_reading_full_data():
    from mousedroid.comms.protocol import EncoderReading

    data = {"lv": 0.5, "rv": 0.3, "ox": 1.0, "oy": 2.0, "h": 0.1, "ts": 100.0}
    reading = parse_encoder_reading(data)
    assert isinstance(reading, EncoderReading)
    assert reading.left_velocity_mps == pytest.approx(0.5)
    assert reading.right_velocity_mps == pytest.approx(0.3)
    assert reading.odometry_x_m == pytest.approx(1.0)
    assert reading.odometry_y_m == pytest.approx(2.0)
    assert reading.heading_rad == pytest.approx(0.1)
    assert reading.timestamp == pytest.approx(100.0)


def test_parse_encoder_reading_empty_dict():
    reading = parse_encoder_reading({})
    assert reading.left_velocity_mps == 0.0
    assert reading.right_velocity_mps == 0.0
    assert reading.odometry_x_m == 0.0
    assert reading.odometry_y_m == 0.0
    assert reading.heading_rad == 0.0
    assert reading.timestamp == 0.0


def test_parse_encoder_reading_partial_data():
    data = {"lv": 1.2, "ts": 42.0}
    reading = parse_encoder_reading(data)
    assert reading.left_velocity_mps == pytest.approx(1.2)
    assert reading.right_velocity_mps == 0.0
    assert reading.timestamp == pytest.approx(42.0)


def test_parse_encoder_reading_returns_floats():
    data = {"lv": 1, "rv": 2, "ox": 3, "oy": 4, "h": 5, "ts": 6}
    reading = parse_encoder_reading(data)
    assert isinstance(reading.left_velocity_mps, float)
    assert isinstance(reading.right_velocity_mps, float)


# ---------------------------------------------------------------------------
# Protocol constants sanity checks
# ---------------------------------------------------------------------------


def test_constants_are_distinct():
    constants = {
        ESP32_CMD_TYPE_STOP,
        ESP32_CMD_TYPE_VELOCITY,
        ESP32_CMD_TYPE_BATTERY,
    }
    assert len(constants) == 3


def test_max_pwm_is_positive():
    assert MAX_PWM > 0


# ---------------------------------------------------------------------------
# F-025 — command-set codecs (mousedroid.comms.command_set)
#
# These pins are named by features.yaml F-025's verification bullets and MUST
# stay in this file (the feature's validation_command runs it): the stock
# T=13 velocity / T=136 heartbeat / FEEDBACK_BASE_INFO voltage-parse pins,
# and the legacy codec's full-dict byte-identity with the historical builder.
# ---------------------------------------------------------------------------


def _stock_cfg(**overrides: object) -> ESP32Config:
    return ESP32Config.model_validate({"command_set": "waveshare_stock", **overrides})


class TestCommandCodecResolution:
    """resolve_command_codec dispatches on the selector, legacy fall-through."""

    def test_default_resolves_legacy_singleton(self) -> None:
        """A default config resolves the legacy codec — pre-F-025 behaviour.

        Asserts the *default value* inline first so a silent selector-default
        drift fails here, not three layers up in a driver test.
        """
        from mousedroid.comms.command_set import LEGACY_CODEC, resolve_command_codec

        cfg = ESP32Config()
        assert cfg.command_set == "legacy"
        assert resolve_command_codec(cfg) is LEGACY_CODEC

    def test_stock_resolves_stock_singleton(self) -> None:
        from mousedroid.comms.command_set import (
            WAVESHARE_STOCK_CODEC,
            resolve_command_codec,
        )

        assert resolve_command_codec(_stock_cfg()) is WAVESHARE_STOCK_CODEC

    def test_both_codecs_satisfy_protocol(self) -> None:
        from mousedroid.comms.command_set import (
            LEGACY_CODEC,
            WAVESHARE_STOCK_CODEC,
            ESP32CommandCodec,
        )

        assert isinstance(LEGACY_CODEC, ESP32CommandCodec)
        assert isinstance(WAVESHARE_STOCK_CODEC, ESP32CommandCodec)


class TestLegacyCodecByteIdentity:
    """The legacy codec is pure delegation — full-dict equality, key for key."""

    def test_velocity_identical_to_historical_builder(self) -> None:
        from mousedroid.comms.command_set import LEGACY_CODEC

        cfg = _cfg()
        assert LEGACY_CODEC.build_velocity(0.5, -0.5, 1.0, cfg) == build_velocity_cmd(
            0.5, -0.5, 1.0, cfg
        )
        assert LEGACY_CODEC.build_velocity(0.5, -0.5, 1.0, cfg) == {
            "T": ESP32_CMD_TYPE_VELOCITY,
            "vx": 127,
            "vy": -127,
            "omega": MAX_PWM,
        }

    def test_stop_and_battery_dicts_identical(self) -> None:
        from mousedroid.comms.command_set import LEGACY_CODEC

        assert LEGACY_CODEC.build_stop() == {"T": ESP32_CMD_TYPE_STOP}
        assert LEGACY_CODEC.battery_query() == {"T": ESP32_CMD_TYPE_BATTERY}

    def test_encoder_query_sends_nothing(self) -> None:
        """Legacy reads never write — firmware pushes frames unsolicited."""
        from mousedroid.comms.command_set import LEGACY_CODEC

        assert LEGACY_CODEC.encoder_query() is None

    def test_battery_parse_reads_v_key(self) -> None:
        from mousedroid.comms.command_set import LEGACY_CODEC

        assert LEGACY_CODEC.parse_battery({"v": 11.7}) == pytest.approx(11.7)

    def test_battery_parse_returns_none_when_absent(self) -> None:
        """A frame with no ``v`` is a non-answer, not a zero-volt rover.

        The historical contract fabricated ``0.0``, which the safety monitor
        then read as a critically flat pack — a comms fault latching a
        permanent emergency stop. ``None`` keeps the two cases distinct.
        """
        from mousedroid.comms.command_set import LEGACY_CODEC

        assert LEGACY_CODEC.parse_battery({}) is None
        assert LEGACY_CODEC.parse_battery({"other": 1}) is None
        assert LEGACY_CODEC.parse_battery({"v": "junk"}) is None

    def test_encoder_parse_delegates_to_utils(self) -> None:
        from mousedroid.comms.command_set import LEGACY_CODEC

        reading = LEGACY_CODEC.parse_encoders({"lv": 0.25, "rv": -0.25, "ts": 3.0})
        assert reading == parse_encoder_reading({"lv": 0.25, "rv": -0.25, "ts": 3.0})

    def test_no_connect_commands(self) -> None:
        """Legacy connect sends zero extra writes — pinned so the
        ``_arm_command_set`` hook can never regress the historical
        connect sequence."""
        from mousedroid.comms.command_set import LEGACY_CODEC

        assert LEGACY_CODEC.connect_commands(_cfg()) == []

    def test_supports_lateral(self) -> None:
        from mousedroid.comms.command_set import LEGACY_CODEC

        assert LEGACY_CODEC.supports_lateral is True


class TestWaveshareStockVelocity:
    """CMD_ROS_CTRL: physical units, clamped, no lateral axis, no PWM."""

    def test_velocity_is_t13_physical_units(self) -> None:
        from mousedroid.comms.command_set import (
            WAVESHARE_CMD_ROS_CTRL,
            WAVESHARE_STOCK_CODEC,
        )

        cfg = _stock_cfg(max_velocity_mps=0.5, max_omega_rads=2.0)
        cmd = WAVESHARE_STOCK_CODEC.build_velocity(0.25, 0.0, 1.0, cfg)
        assert cmd == {"T": WAVESHARE_CMD_ROS_CTRL, "X": 0.25, "Z": 1.0}
        assert WAVESHARE_CMD_ROS_CTRL == 13

    def test_velocity_clamps_to_config_maxima(self) -> None:
        from mousedroid.comms.command_set import WAVESHARE_STOCK_CODEC

        cfg = _stock_cfg(max_velocity_mps=0.5, max_omega_rads=2.0)
        cmd = WAVESHARE_STOCK_CODEC.build_velocity(9.0, 0.0, -9.0, cfg)
        assert cmd["X"] == cfg.max_velocity_mps
        assert cmd["Z"] == -cfg.max_omega_rads

    def test_velocity_has_no_lateral_key_and_flag_says_so(self) -> None:
        """Skid-steer chassis: vy is physically inexpressible in T=13."""
        from mousedroid.comms.command_set import WAVESHARE_STOCK_CODEC

        cmd = WAVESHARE_STOCK_CODEC.build_velocity(0.1, 0.4, 0.0, _stock_cfg())
        assert set(cmd) == {"T", "X", "Z"}
        assert WAVESHARE_STOCK_CODEC.supports_lateral is False

    def test_velocity_no_quantisation(self) -> None:
        """Stock X passes through exactly (floats) — never PWM-truncated."""
        from mousedroid.comms.command_set import WAVESHARE_STOCK_CODEC

        cmd = WAVESHARE_STOCK_CODEC.build_velocity(0.123, 0.0, 0.456, _stock_cfg())
        assert cmd["X"] == pytest.approx(0.123)
        assert cmd["Z"] == pytest.approx(0.456)

    def test_stop_is_zero_velocity_ros_ctrl(self) -> None:
        """Stock defines no e-stop command; stop = zero-velocity T=13."""
        from mousedroid.comms.command_set import (
            WAVESHARE_CMD_ROS_CTRL,
            WAVESHARE_STOCK_CODEC,
        )

        assert WAVESHARE_STOCK_CODEC.build_stop() == {
            "T": WAVESHARE_CMD_ROS_CTRL,
            "X": 0.0,
            "Z": 0.0,
        }


class TestWaveshareStockTelemetry:
    """CMD_BASE_FEEDBACK poll + T-gated FEEDBACK_BASE_INFO parsing."""

    def test_battery_query_is_base_feedback_read(self) -> None:
        """The battery step polls T=130 — a READ. The legacy {"T":2} was
        stock CMD_SET_MOTOR_PID, a motor-controller WRITE fired immediately
        before commanding motion (audit R1/R5)."""
        from mousedroid.comms.command_set import (
            WAVESHARE_CMD_BASE_FEEDBACK,
            WAVESHARE_STOCK_CODEC,
        )

        assert WAVESHARE_STOCK_CODEC.battery_query() == {"T": WAVESHARE_CMD_BASE_FEEDBACK}
        assert WAVESHARE_CMD_BASE_FEEDBACK == 130
        assert WAVESHARE_STOCK_CODEC.battery_query() != {"T": ESP32_CMD_TYPE_BATTERY}

    def test_encoder_query_polls_base_feedback(self) -> None:
        """Stock frames must be polled — an un-polled read returns nothing
        forever (serial `_query_data` only writes when given a command)."""
        from mousedroid.comms.command_set import (
            WAVESHARE_CMD_BASE_FEEDBACK,
            WAVESHARE_STOCK_CODEC,
        )

        assert WAVESHARE_STOCK_CODEC.encoder_query() == {"T": WAVESHARE_CMD_BASE_FEEDBACK}

    def test_battery_parse_reads_v_from_1001_frame(self) -> None:
        from mousedroid.comms.command_set import (
            WAVESHARE_FEEDBACK_BASE_INFO,
            WAVESHARE_STOCK_CODEC,
        )

        frame = {"T": WAVESHARE_FEEDBACK_BASE_INFO, "L": 0.1, "R": 0.1, "v": 11.4}
        assert WAVESHARE_STOCK_CODEC.parse_battery(frame) == pytest.approx(11.4)
        assert WAVESHARE_FEEDBACK_BASE_INFO == 1001

    def test_battery_parse_rejects_wrong_frame_type(self) -> None:
        """A stale / non-1001 frame yields ``None`` — no reading, not 0 V.

        Stock firmware streams several frame types with no per-command ACKs,
        so reading a wrong-typed frame is routine; reporting it as zero volts
        would trip ``battery_critical`` and latch an emergency stop.
        """
        from mousedroid.comms.command_set import WAVESHARE_STOCK_CODEC

        assert WAVESHARE_STOCK_CODEC.parse_battery({"T": 1003, "v": 11.4}) is None
        assert WAVESHARE_STOCK_CODEC.parse_battery({}) is None
        assert WAVESHARE_STOCK_CODEC.parse_battery({"T": "junk", "v": 9.9}) is None
        # A well-formed 1001 frame missing the voltage key is also "no reading".
        assert WAVESHARE_STOCK_CODEC.parse_battery({"T": 1001, "L": 0.0}) is None

    def test_encoder_parse_maps_l_r_wheel_speeds(self) -> None:
        from mousedroid.comms.command_set import (
            WAVESHARE_FEEDBACK_BASE_INFO,
            WAVESHARE_STOCK_CODEC,
        )

        frame = {"T": WAVESHARE_FEEDBACK_BASE_INFO, "L": 0.22, "R": -0.11, "v": 12.0}
        reading = WAVESHARE_STOCK_CODEC.parse_encoders(frame)
        assert reading.left_velocity_mps == pytest.approx(0.22)
        assert reading.right_velocity_mps == pytest.approx(-0.11)
        # Encoder-less chassis: odometry/heading stay structurally zero.
        assert reading.odometry_x_m == 0.0
        assert reading.heading_rad == 0.0

    def test_encoder_parse_rejects_wrong_frame_type(self) -> None:
        from mousedroid.comms.command_set import WAVESHARE_STOCK_CODEC

        reading = WAVESHARE_STOCK_CODEC.parse_encoders({"T": 1002, "L": 5.0})
        assert reading.left_velocity_mps == 0.0
        assert reading.right_velocity_mps == 0.0


class TestWaveshareStockHeartbeat:
    """CMD_HEART_BEAT_SET armed at connect; window derived from the driver's
    own worst-case command gap (not ``keepalive_hz`` alone)."""

    def test_connect_arms_t136_with_derived_window(self) -> None:
        from mousedroid.comms.command_set import (
            WAVESHARE_CMD_HEART_BEAT_SET,
            WAVESHARE_STOCK_CODEC,
            heartbeat_window_ms,
        )

        cfg = _stock_cfg()
        cmds = WAVESHARE_STOCK_CODEC.connect_commands(cfg)
        # Shipped defaults: worst-case gap is degraded_poll_interval_s (1.0 s)
        # x multiple 3.0 = 3000 ms.
        assert cmds == [{"T": WAVESHARE_CMD_HEART_BEAT_SET, "cmd": 3000}]
        assert WAVESHARE_CMD_HEART_BEAT_SET == 136
        assert heartbeat_window_ms(cfg) == 3000

    def test_window_derives_from_worst_case_gap_not_keepalive_alone(self) -> None:
        """The window must out-wait every blocking budget the driver has.

        Deriving from ``keepalive_hz`` alone yielded 300 ms — shorter than
        both ``command_timeout_s`` (500 ms) and ``degraded_poll_interval_s``
        (1000 ms), so the chassis failsafe would halt the wheels during
        conditions the host considers normal-but-slow.
        """
        from mousedroid.comms.command_set import heartbeat_window_ms, worst_case_command_gap_s

        cfg = _stock_cfg()
        assert worst_case_command_gap_s(cfg) == cfg.degraded_poll_interval_s
        window_ms = heartbeat_window_ms(cfg)
        assert window_ms > cfg.command_timeout_s * 1000
        assert window_ms > cfg.degraded_poll_interval_s * 1000

    def test_window_tracks_whichever_budget_dominates(self) -> None:
        """Each budget takes over the derivation when it becomes the largest."""
        from mousedroid.comms.command_set import heartbeat_window_ms, worst_case_command_gap_s

        # command_timeout dominates
        a = _stock_cfg(command_timeout_s=2.0, degraded_poll_interval_s=0.5, keepalive_hz=10.0)
        assert worst_case_command_gap_s(a) == 2.0
        assert heartbeat_window_ms(a) == 6000
        # keepalive period dominates (very slow cadence)
        b = _stock_cfg(keepalive_hz=0.2, command_timeout_s=0.1, degraded_poll_interval_s=0.1)
        assert worst_case_command_gap_s(b) == 5.0
        assert heartbeat_window_ms(b) == 15000

    def test_tightening_timeouts_tightens_the_window(self) -> None:
        """Dynamic coupling: no second number for the operator to remember."""
        from mousedroid.comms.command_set import heartbeat_window_ms

        loose = heartbeat_window_ms(_stock_cfg())
        tight = heartbeat_window_ms(_stock_cfg(command_timeout_s=0.1, degraded_poll_interval_s=0.2))
        assert tight < loose
        assert tight == 600

    def test_window_never_rounds_to_zero(self) -> None:
        """A zero window means *disabled* on stock firmware, not *tightest*.

        ``keepalive_hz`` has no upper bound, so an extreme value (or a tiny
        multiple) could round the derivation to 0 and silently disarm the
        failsafe while ``heartbeat_enabled`` still reads True. The floor
        keeps "enabled" honest.
        """
        from mousedroid.comms.command_set import MIN_HEARTBEAT_WINDOW_MS, heartbeat_window_ms

        extreme = _stock_cfg(
            keepalive_hz=100000.0,
            command_timeout_s=1e-6,
            degraded_poll_interval_s=1e-6,
            heartbeat_window_multiple=1e-6,
        )
        assert heartbeat_window_ms(extreme) == MIN_HEARTBEAT_WINDOW_MS
        assert MIN_HEARTBEAT_WINDOW_MS > 0

    def test_window_rounds_up_never_down(self) -> None:
        """``ceil``, not banker's rounding — rounding up never disarms."""
        from mousedroid.comms.command_set import heartbeat_window_ms

        # gap 0.1 s * 0.0205 -> 2.05 ms; round() would give 2, ceil gives 3.
        cfg = _stock_cfg(
            keepalive_hz=10.0,
            command_timeout_s=0.001,
            degraded_poll_interval_s=0.001,
            heartbeat_window_multiple=0.0205,
        )
        assert heartbeat_window_ms(cfg) == 3

    def test_heartbeat_disabled_sends_nothing(self) -> None:
        from mousedroid.comms.command_set import WAVESHARE_STOCK_CODEC

        assert WAVESHARE_STOCK_CODEC.connect_commands(_stock_cfg(heartbeat_enabled=False)) == []
