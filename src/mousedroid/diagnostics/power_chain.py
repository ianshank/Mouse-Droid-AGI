"""Power-chain smoke helper — battery + zero-vel + e-stop within budget."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from mousedroid.config.schema import ESP32Config
from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)


@runtime_checkable
class _PowerCapableDriver(Protocol):
    """Minimal slice of the ESP32 driver interface we depend on."""

    async def send_velocity(self, vx: float, vy: float, omega: float) -> None: ...
    async def emergency_stop(self) -> None: ...
    async def get_battery_voltage(self) -> float: ...


@dataclass(frozen=True)
class PowerChainResult:
    """Structured outcome of an ``assert_power_chain`` run."""

    battery_voltage_v: float
    commanded_velocity_mps: float
    estop_latency_ms: float
    notes: str


async def assert_power_chain(
    *,
    driver: _PowerCapableDriver,
    esp32_cfg: ESP32Config,
    allow_motion: bool,
) -> PowerChainResult:
    """Probe battery, dispatch a (possibly zero) velocity, then time the e-stop.

    Args:
        driver: Anything satisfying ``_PowerCapableDriver`` (mock or real).
        esp32_cfg: Source of the smoke velocity setpoint + e-stop budget.
        allow_motion: When False, ``commanded_velocity_mps`` is forced to
            zero so the rover does not roll while running unattended.

    Returns:
        Structured ``PowerChainResult`` for the smoke harness to assert on.
    """
    voltage = await driver.get_battery_voltage()
    target = esp32_cfg.smoke_test_velocity_mps if allow_motion else 0.0
    _log.info(
        "power_chain_probe_start",
        battery_v=voltage,
        target_vx=target,
        allow_motion=allow_motion,
    )

    await driver.send_velocity(target, 0.0, 0.0)
    t0 = time.monotonic()
    await driver.emergency_stop()
    elapsed_ms = (time.monotonic() - t0) * 1000.0

    notes = (
        f"battery={voltage:.2f}V cmd={target:.3f}m/s "
        f"estop={elapsed_ms:.1f}ms (budget={esp32_cfg.emergency_stop_budget_ms:.0f}ms)"
    )
    _log.info("power_chain_probe_complete", summary=notes)
    return PowerChainResult(
        battery_voltage_v=voltage,
        commanded_velocity_mps=target,
        estop_latency_ms=elapsed_ms,
        notes=notes,
    )
