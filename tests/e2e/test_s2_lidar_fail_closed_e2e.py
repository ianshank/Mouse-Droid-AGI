"""E2E: a dead LiDAR reaches the ESP32 emergency stop, not the motors.

Finding S-2. The tiers below this one prove the sensing layer stops fabricating
an all-clear feature vector and that the safety monitor honours
``SafetyConfig.lidar_unavailable_policy``. This one runs the real
factory-built ``MouseDroidOrchestrator`` tick and asserts the consequence an
operator actually cares about: with the interlock armed,
``ESP32CommProtocol.emergency_stop`` is what the rover receives.

Pre-fix, this scenario drove normally. The LiDAR read raised, the manager
substituted ``np.ones(feature_dim)``, and because features are normalised
range fractions the monitor read 12 m of clearance in every sector.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from mousedroid.config.schema import Settings
from mousedroid.factory import build_orchestrator


@pytest.fixture(autouse=True)
def _set_mock_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOUSEDROID_MOCK_HARDWARE", "true")


def _settings(policy: str) -> Settings:
    return Settings.model_validate(
        {
            "mock_hardware": True,
            "lidar": {"enabled": True, "n_sectors": 36, "feature_dim": 36},
            "safety": {
                "lidar_unavailable_policy": policy,
                # Keep the tick's recovery detour short; the LiDAR stays dead
                # through it either way, which is the point.
                "sensor_recovery_attempts": 1,
                "sensor_recovery_delay_s": 0.01,
                # Park the generic staleness sweep so any emergency stop
                # observed below is attributable to the LiDAR policy.
                "sensor_stale_s": 1000.0,
            },
        }
    )


async def _stops_during_one_tick(policy: str, *, kill_lidar: bool) -> int:
    """Count ``emergency_stop`` calls made by ONE tick.

    Counted before ``stop()``, which issues its own shutdown emergency stop --
    folding that in would make every case here look like a trip.
    """
    orch = build_orchestrator(_settings(policy))
    if kill_lidar:
        assert orch._sensor_manager._lidar is not None, "LiDAR must be wired"
        orch._sensor_manager._lidar.read_scan = AsyncMock(
            side_effect=RuntimeError("LD19 serial gone")
        )
    stop_spy = AsyncMock()
    orch._esp32.emergency_stop = stop_spy  # type: ignore[method-assign]

    await orch.start()
    try:
        await orch.tick()
        during_tick = stop_spy.await_count
    finally:
        await orch.stop()
    return during_tick


async def test_dead_lidar_emergency_stops_the_rover() -> None:
    """Armed interlock: the tick ends in an emergency stop."""
    assert await _stops_during_one_tick("emergency", kill_lidar=True) >= 1


async def test_dead_lidar_under_default_policy_keeps_driving() -> None:
    """Unarmed (shipped default): behaviour is unchanged, as invariant 6 requires.

    This is the uncomfortable half of the fix, asserted deliberately rather
    than left implicit: until an operator sets the policy in YAML, a dead
    LiDAR still does not stop the rover. What the unconditional half of S-2
    removes is the *fabricated distance*, not this posture.
    """
    assert await _stops_during_one_tick("ignore", kill_lidar=True) == 0


async def test_healthy_lidar_does_not_trip_the_armed_interlock() -> None:
    """A working mock LiDAR ticks cleanly with the interlock armed."""
    assert await _stops_during_one_tick("emergency", kill_lidar=False) == 0
