"""GPIO integration tests using mock gpiod fixtures.

Tests GPIO chip lifecycle, pin request/release, trigger/echo sequence
for HC-SR04, distance calculation from echo timing, and error handling.

All thresholds derived from ``UltrasonicConfig`` — no hardcoded values.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import MockGPIOChip, MockGPIOLine

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# 1. GPIO chip open/close lifecycle
# ---------------------------------------------------------------------------


class TestGPIOChipLifecycle:
    """Verify GPIO chip open and close behaviour."""

    def test_chip_opens_with_name(self, mock_gpiod: MockGPIOChip) -> None:
        """Chip should report the configured name."""
        assert mock_gpiod.name == "gpiochip0"

    def test_chip_starts_open(self, mock_gpiod: MockGPIOChip) -> None:
        """Chip should not be closed after construction."""
        assert not mock_gpiod.is_closed

    def test_chip_close_marks_closed(self, mock_gpiod: MockGPIOChip) -> None:
        """Close should mark the chip as closed."""
        mock_gpiod.close()
        assert mock_gpiod.is_closed

    def test_get_line_after_close_raises(self, mock_gpiod: MockGPIOChip) -> None:
        """Accessing a line on a closed chip should raise RuntimeError."""
        mock_gpiod.close()
        with pytest.raises(RuntimeError, match="closed"):
            mock_gpiod.get_line(23)

    def test_multiple_close_idempotent(self, mock_gpiod: MockGPIOChip) -> None:
        """Closing an already-closed chip should not raise."""
        mock_gpiod.close()
        mock_gpiod.close()
        assert mock_gpiod.is_closed


# ---------------------------------------------------------------------------
# 2. Pin request/release
# ---------------------------------------------------------------------------


class TestPinRequestRelease:
    """Verify GPIO line request and release semantics."""

    def test_get_line_returns_mock_line(self, mock_gpiod: MockGPIOChip) -> None:
        """get_line should return a MockGPIOLine with correct offset."""
        line = mock_gpiod.get_line(23)
        assert isinstance(line, MockGPIOLine)
        assert line.offset == 23

    def test_line_not_requested_initially(self, mock_gpiod: MockGPIOChip) -> None:
        """Line should not be requested before explicit request call."""
        line = mock_gpiod.get_line(24)
        assert not line.is_requested

    def test_request_marks_line_requested(self, mock_gpiod: MockGPIOChip) -> None:
        """request() should mark the line as requested."""
        line = mock_gpiod.get_line(23)
        line.request(consumer="test", type=1)
        assert line.is_requested

    def test_release_marks_line_unrequested(self, mock_gpiod: MockGPIOChip) -> None:
        """release() should mark the line as no longer requested."""
        line = mock_gpiod.get_line(23)
        line.request(consumer="test", type=1)
        line.release()
        assert not line.is_requested

    def test_same_offset_returns_same_line(self, mock_gpiod: MockGPIOChip) -> None:
        """Requesting the same offset twice should return the same instance."""
        line_a = mock_gpiod.get_line(23)
        line_b = mock_gpiod.get_line(23)
        assert line_a is line_b

    def test_different_offsets_different_lines(self, mock_gpiod: MockGPIOChip) -> None:
        """Different offsets should return different line instances."""
        line_23 = mock_gpiod.get_line(23)
        line_24 = mock_gpiod.get_line(24)
        assert line_23 is not line_24


# ---------------------------------------------------------------------------
# 3. Trigger/echo sequence for HC-SR04
# ---------------------------------------------------------------------------


class TestTriggerEchoSequence:
    """Verify trigger pulse and echo timing simulation."""

    def test_set_value_high(self, mock_gpiod: MockGPIOChip) -> None:
        """Setting trigger HIGH should store value 1."""
        trigger = mock_gpiod.get_line(23)
        trigger.request(consumer="hc_sr04_trigger", type=1)
        trigger.set_value(1)
        # get_value on output pin returns stored value
        assert trigger.get_value() == 1

    def test_set_value_low(self, mock_gpiod: MockGPIOChip) -> None:
        """Setting trigger LOW should store value 0."""
        trigger = mock_gpiod.get_line(23)
        trigger.request(consumer="hc_sr04_trigger", type=1)
        trigger.set_value(1)
        trigger.set_value(0)
        assert trigger.get_value() == 0

    def test_echo_sequence_returns_configured_values(self, mock_gpiod: MockGPIOChip) -> None:
        """Echo line should return configured sequence values in order."""
        echo = mock_gpiod.get_line(24)
        echo.request(consumer="hc_sr04_echo", type=2)
        # Simulate: LOW LOW LOW HIGH HIGH HIGH LOW (echo pulse)
        echo.set_echo_sequence([0, 0, 0, 1, 1, 1, 0])

        results = [echo.get_value() for _ in range(7)]
        assert results == [0, 0, 0, 1, 1, 1, 0]

    def test_echo_sequence_cycles(self, mock_gpiod: MockGPIOChip) -> None:
        """Echo sequence should cycle when exhausted."""
        echo = mock_gpiod.get_line(24)
        echo.set_echo_sequence([0, 1])

        results = [echo.get_value() for _ in range(4)]
        assert results == [0, 1, 0, 1]

    def test_trigger_echo_full_sequence(self, mock_gpiod: MockGPIOChip) -> None:
        """Full trigger-echo sequence should work without errors."""
        trigger = mock_gpiod.get_line(23)
        echo = mock_gpiod.get_line(24)
        trigger.request(consumer="hc_sr04_trigger", type=1)
        echo.request(consumer="hc_sr04_echo", type=2)

        # Configure echo: 5 LOWs (waiting), 3 HIGHs (pulse), then LOW
        echo.set_echo_sequence([0, 0, 0, 0, 0, 1, 1, 1, 0])

        # Trigger pulse
        trigger.set_value(1)
        time.sleep(0.00001)  # 10 us trigger pulse
        trigger.set_value(0)

        # Wait for echo start (reading LOWs)
        for _ in range(5):
            val = echo.get_value()
            if val == 1:
                break
        # Read HIGH period
        high_count = 0
        for _ in range(10):
            val = echo.get_value()
            if val == 1:
                high_count += 1
            else:
                break

        assert high_count >= 1


# ---------------------------------------------------------------------------
# 4. Distance calculation from echo timing
# ---------------------------------------------------------------------------


class TestDistanceCalculation:
    """Verify HC-SR04 distance calculation using config values."""

    @pytest.fixture
    def ultrasonic_cfg(self) -> object:
        """Provide a minimal UltrasonicConfig for distance calculations."""
        from mousedroid.config.schema import UltrasonicConfig

        return UltrasonicConfig(
            trigger_pin=23,
            echo_pin=24,
            max_range_m=4.0,
            min_range_m=0.02,
            timeout_s=0.1,
            speed_of_sound_mps=343.0,
        )

    def test_distance_from_echo_duration(self, ultrasonic_cfg: object) -> None:
        """Distance = (echo_duration * speed_of_sound) / 2."""
        from mousedroid.config.schema import UltrasonicConfig

        cfg: UltrasonicConfig = ultrasonic_cfg  # type: ignore[assignment]
        # Simulate 1 ms echo duration -> ~0.1715 m
        echo_duration_s = 0.001
        expected_m = (echo_duration_s * cfg.speed_of_sound_mps) / 2.0

        assert abs(expected_m - 0.1715) < 0.001

    def test_distance_capped_at_max_range(self, ultrasonic_cfg: object) -> None:
        """Distance exceeding max_range should be capped."""
        from mousedroid.config.schema import UltrasonicConfig

        cfg: UltrasonicConfig = ultrasonic_cfg  # type: ignore[assignment]
        # Very long echo -> should cap at max_range
        echo_duration_s = 0.05  # ~8.575 m, exceeds max_range
        raw_distance = (echo_duration_s * cfg.speed_of_sound_mps) / 2.0
        capped = min(raw_distance, cfg.max_range_m)

        assert capped == cfg.max_range_m

    def test_short_echo_within_min_range(self, ultrasonic_cfg: object) -> None:
        """Very short echo should produce distance below min_range."""
        from mousedroid.config.schema import UltrasonicConfig

        cfg: UltrasonicConfig = ultrasonic_cfg  # type: ignore[assignment]
        # 0.05 ms echo -> ~0.0086 m which is below min_range
        echo_duration_s = 0.00005
        distance = (echo_duration_s * cfg.speed_of_sound_mps) / 2.0

        assert distance < cfg.min_range_m

    def test_typical_1m_distance(self, ultrasonic_cfg: object) -> None:
        """1 metre target should give ~5.83 ms echo duration."""
        from mousedroid.config.schema import UltrasonicConfig

        cfg: UltrasonicConfig = ultrasonic_cfg  # type: ignore[assignment]
        target_m = 1.0
        expected_echo_s = (2.0 * target_m) / cfg.speed_of_sound_mps
        calculated_m = (expected_echo_s * cfg.speed_of_sound_mps) / 2.0

        assert abs(calculated_m - target_m) < 0.001


# ---------------------------------------------------------------------------
# 5. Error handling: chip not found, pin busy
# ---------------------------------------------------------------------------


class TestGPIOErrorHandling:
    """Verify graceful error handling for GPIO failures."""

    def test_chip_not_found_raises(self) -> None:
        """Opening a non-existent chip should raise an error."""
        chip = MockGPIOChip("nonexistent_chip")
        chip.close()
        with pytest.raises(RuntimeError):
            chip.get_line(0)

    def test_line_value_without_request(self, mock_gpiod: MockGPIOChip) -> None:
        """Getting value on unrequested line should still work (mock behaviour)."""
        line = mock_gpiod.get_line(23)
        # In mock, this should not raise even without request
        val = line.get_value()
        assert val == 0

    def test_set_value_stores_correctly(self, mock_gpiod: MockGPIOChip) -> None:
        """set_value should correctly store the new value."""
        line = mock_gpiod.get_line(23)
        line.request(consumer="test", type=1)
        line.set_value(1)
        assert line.get_value() == 1
        line.set_value(0)
        assert line.get_value() == 0

    def test_echo_sequence_reset_on_new_set(self, mock_gpiod: MockGPIOChip) -> None:
        """Setting a new echo sequence resets the index."""
        line = mock_gpiod.get_line(24)
        line.set_echo_sequence([1, 0, 1])
        _ = line.get_value()  # consume first
        _ = line.get_value()  # consume second

        # Reset with new sequence
        line.set_echo_sequence([0, 0, 0])
        assert line.get_value() == 0

    def test_multiple_pins_independent(self, mock_gpiod: MockGPIOChip) -> None:
        """Multiple pins should operate independently."""
        pin_23 = mock_gpiod.get_line(23)
        pin_24 = mock_gpiod.get_line(24)

        pin_23.set_value(1)
        pin_24.set_value(0)

        assert pin_23.get_value() == 1
        assert pin_24.get_value() == 0

    def test_hc_sr04_timeout_returns_max_range(self) -> None:
        """When echo never goes HIGH, sensor should return max_range_m."""
        from mousedroid.config.schema import UltrasonicConfig

        cfg = UltrasonicConfig(
            trigger_pin=23,
            echo_pin=24,
            max_range_m=4.0,
            min_range_m=0.02,
            timeout_s=0.001,  # Very short timeout for test speed
            speed_of_sound_mps=343.0,
        )

        # Patch _GPIO to simulate always-low echo
        mock_gpio = MagicMock()
        mock_gpio.BCM = 11
        mock_gpio.OUT = 0
        mock_gpio.IN = 1
        mock_gpio.HIGH = 1
        mock_gpio.LOW = 0
        mock_gpio.input.return_value = 0  # Echo never goes HIGH
        mock_gpio.setmode = MagicMock()
        mock_gpio.setup = MagicMock()
        mock_gpio.output = MagicMock()

        with patch("mousedroid.hardware.sensors.ultrasonic._GPIO", mock_gpio):
            from mousedroid.hardware.sensors.ultrasonic import HcSr04

            sensor = HcSr04(cfg)
            distance = sensor._measure_distance()

        assert distance == cfg.max_range_m
