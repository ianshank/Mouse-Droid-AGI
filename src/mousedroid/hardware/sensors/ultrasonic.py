"""HC-SR04 ultrasonic distance sensor driver for Jetson Nano GPIO.

Implements ``DistanceSensorProtocol`` using trigger/echo GPIO timing.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import UltrasonicConfig

_GPIO: Any = None
try:
    import Jetson.GPIO as _GPIO_MOD

    _GPIO = _GPIO_MOD
except ImportError:  # pragma: no cover
    pass

_log = get_logger(__name__)


class HcSr04:
    """HC-SR04 ultrasonic sensor implementing ``DistanceSensorProtocol``.

    Uses Jetson.GPIO for trigger/echo timing. All real GPIO access is
    delegated to ``asyncio.to_thread``.
    """

    def __init__(self, cfg: UltrasonicConfig) -> None:
        """Initialise sensor from config.

        Args:
            cfg: Ultrasonic sensor configuration with GPIO pins and ranges.
        """
        self._cfg = cfg
        self._trigger_pin: int = cfg.trigger_pin
        self._echo_pin: int = cfg.echo_pin
        self._speed_of_sound_mps: float = cfg.speed_of_sound_mps
        self._echo_timeout_s: float = cfg.timeout_s
        self._setup_gpio()

    def _setup_gpio(self) -> None:  # pragma: no cover
        """Configure GPIO pins for trigger and echo."""
        if _GPIO is None:
            _log.warning("jetson_gpio_unavailable")
            return
        _GPIO.setmode(_GPIO.BCM)
        _GPIO.setup(self._trigger_pin, _GPIO.OUT, initial=_GPIO.LOW)
        _GPIO.setup(self._echo_pin, _GPIO.IN)
        _log.info(
            "hc_sr04_gpio_setup",
            trigger=self._trigger_pin,
            echo=self._echo_pin,
        )

    async def read_distance_m(self) -> float:
        """Read distance via GPIO trigger/echo timing.

        Returns:
            Distance in metres. Returns ``max_range_m`` on timeout.
        """
        if _GPIO is None:
            return self._cfg.max_range_m
        return await asyncio.to_thread(self._measure_distance)

    def _measure_distance(self) -> float:  # pragma: no cover
        """Perform blocking trigger/echo measurement.

        Returns:
            Distance in metres.
        """
        _GPIO.output(self._trigger_pin, _GPIO.HIGH)
        time.sleep(0.00001)
        _GPIO.output(self._trigger_pin, _GPIO.LOW)

        start = time.monotonic()
        deadline = start + self._echo_timeout_s

        # Wait for echo start
        while _GPIO.input(self._echo_pin) == 0:
            start = time.monotonic()
            if start > deadline:
                return self._cfg.max_range_m

        # Wait for echo end
        end = start
        while _GPIO.input(self._echo_pin) == 1:
            end = time.monotonic()
            if end > deadline:
                return self._cfg.max_range_m

        elapsed = end - start
        distance = (elapsed * self._speed_of_sound_mps) / 2.0
        return min(distance, self._cfg.max_range_m)

    @property
    def max_range_m(self) -> float:
        """Maximum detection range in metres."""
        return self._cfg.max_range_m

    @property
    def min_range_m(self) -> float:
        """Minimum detection range in metres."""
        return self._cfg.min_range_m
