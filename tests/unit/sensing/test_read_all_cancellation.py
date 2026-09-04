"""``SensorManager.read_all`` must not leave sensor reads running detached.

``read_all`` fans five reads out concurrently. It used to join them with five
*sequential* awaits, which is not equivalent to ``gather`` under cancellation:
``Task.cancel()`` propagates only to the task currently parked in the
coroutine's ``_fut_waiter``, so cancelling ``read_all`` cancelled exactly one
child and orphaned the rest.

That matters because ``read_all`` runs inside
``asyncio.wait_for(self.tick(), tick_timeout_s)``. On a tick timeout the
orphans kept running into the *next* tick, where they mutate shared
degraded-mode state (``_motor_consecutive_failures``, ``_motor_degraded``,
``_cached_motor_state``) out of order, and — with a real serial driver — kept
the ESP32 handle busy past the point where ``run()`` issues its emergency-stop
frame.

These tests exercise the join itself by substituting the five ``_safe_*_read``
helpers, so they pin ``read_all``'s cancellation contract without depending on
any sensor protocol's method names.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import numpy as np
import pytest

from mousedroid.config.schema import Settings
from mousedroid.sensing.manager import SensorManager

_SAFE_READS = (
    "_safe_vision_read",
    "_safe_distance_read",
    "_safe_motor_read",
    "_safe_audio_read",
    "_safe_lidar_read",
)


def _make_manager(cfg: Settings) -> SensorManager:
    """Build a manager whose sensors are inert; the reads are patched per test."""
    vision = AsyncMock()
    vision.capture_features.return_value = np.zeros(cfg.camera.feature_dim, dtype=np.float32)
    distance = AsyncMock()
    distance.read_distance_m.return_value = 1.5
    distance.max_range_m = 4.0
    esp32 = AsyncMock()
    return SensorManager(vision, distance, esp32, cfg)


class _BlockingRead:
    """Stands in for one ``_safe_*_read``: parks on a gate, records its fate."""

    def __init__(self, name: str, gate: asyncio.Event, result: Any) -> None:
        self.name = name
        self.completed = False
        self.cancelled = False
        self._gate = gate
        self._result = result

    async def __call__(self) -> Any:
        try:
            await self._gate.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        self.completed = True
        return self._result


@pytest.mark.asyncio
async def test_cancelling_read_all_cancels_every_sensor_read() -> None:
    """No sensor read may survive cancellation of ``read_all``.

    Red against the pre-fix sequential-await join: the reads that had not yet
    been reached kept running after cancellation was delivered.
    """
    cfg = Settings(mock_hardware=True)
    mgr = _make_manager(cfg)
    gate = asyncio.Event()

    results: dict[str, Any] = {
        "_safe_vision_read": (np.zeros(cfg.camera.feature_dim, dtype=np.float32), True),
        "_safe_distance_read": (1.5, True),
        "_safe_motor_read": (np.zeros(4, dtype=np.float32), True),
        "_safe_audio_read": (np.zeros(1024, dtype=np.float32), True),
        "_safe_lidar_read": (None, False),
    }
    reads = {name: _BlockingRead(name, gate, results[name]) for name in _SAFE_READS}
    for name, stub in reads.items():
        setattr(mgr, name, stub)

    task = asyncio.create_task(mgr.read_all())
    # Let every child reach its await point before cancelling.
    for _ in range(4):
        await asyncio.sleep(0)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Release the gate and give any orphan scheduling slots it should not use.
    gate.set()
    for _ in range(4):
        await asyncio.sleep(0)

    survivors = sorted(name for name, stub in reads.items() if stub.completed)
    assert not survivors, (
        f"sensor reads {survivors} outlived the cancellation of read_all(). "
        "Every child must be cancelled with the parent, or an orphan from "
        "tick N mutates SensorManager state during tick N+1 and holds the "
        "ESP32 serial handle across the emergency-stop write."
    )
    assert all(stub.cancelled for stub in reads.values())


@pytest.mark.asyncio
async def test_read_all_preserves_positional_result_ordering() -> None:
    """Backwards-compatibility half: results still map to the right modalities.

    ``gather`` returns results in argument order, not completion order. This
    pins that the vision / distance / motor / audio / lidar unpacking still
    lines up, by making each read resolve out of order.
    """
    cfg = Settings(mock_hardware=True)
    mgr = _make_manager(cfg)

    async def _vision() -> tuple[Any, bool]:
        await asyncio.sleep(0.03)  # resolves last
        return np.full(cfg.camera.feature_dim, 7.0, dtype=np.float32), True

    async def _distance() -> tuple[float, bool]:
        return 2.25, True

    async def _motor() -> tuple[Any, bool]:
        await asyncio.sleep(0.01)
        return np.array([1.0, 2.0, 3.0, 11.5], dtype=np.float32), True

    async def _audio() -> tuple[Any, bool]:
        return np.zeros(1024, dtype=np.float32), False

    async def _lidar() -> tuple[Any, bool]:
        return None, False

    mgr._safe_vision_read = _vision
    mgr._safe_distance_read = _distance
    mgr._safe_motor_read = _motor
    mgr._safe_audio_read = _audio
    mgr._safe_lidar_read = _lidar

    bundle = await mgr.read_all()

    assert bundle.vision_features[0] == pytest.approx(7.0)
    assert bundle.distance_m == pytest.approx(2.25)
    assert bundle.motor_state[3] == pytest.approx(11.5)
    # vision ok, distance ok, motor ok, audio not ok
    np.testing.assert_array_equal(bundle.valid_mask[:4], [1.0, 1.0, 1.0, 0.0])
