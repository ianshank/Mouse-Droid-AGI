"""Arming the loop-overrun interlock must not turn every boot into an e-stop.

The safety monitor used to receive the sensor-read segment -- a few
milliseconds -- so ``max_loop_time_ms`` was inert no matter what an overlay
set it to. It now receives each tick's real measured total, which is the whole
point of the change, and that arms a 200 ms ceiling against a first tick that
pays lazy CUDA context creation, TensorRT/ONNX engine load, or a first MuJoCo
compile. With no warm-up grace the rover emergency-stops at tick 1 on every
boot, on hardware, for a non-fault.

Both halves are pinned here, because either one alone is trivial to satisfy
the wrong way: a config that never trips is not boot-safe, it is switched off.
So every shipped overlay must survive a slow first tick **and** still trip on a
sustained overrun once warm-up is over.

Nothing here hardcodes a threshold or a tick count -- each expectation is
derived from the overlay's own resolved config, so retuning a rig updates the
test's expectations with it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mousedroid.config.loader import load_settings
from mousedroid.config.schema import Settings
from mousedroid.config.schema.reward_safety import SafetyConfig
from mousedroid.safety.monitor import MouseDroidSafetyMonitor
from mousedroid.sensing.bundle import MouseDroidObservationBundle

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"

# ``baselines.yaml`` is not an overlay: the loader merges it under a dedicated
# ``baselines`` key rather than at the document root, so passing it as one
# would resolve to something no deployment ever runs.
_NOT_AN_OVERLAY = {"baselines.yaml"}

# How much slower than the ceiling a cold first tick is assumed to be. A
# multiplier rather than a literal so the assumption scales with whatever
# ``max_loop_time_ms`` an overlay chooses: 25x the 200 ms shipped ceiling is a
# 5 s first tick, comfortably covering a TensorRT engine load.
_COLD_START_FACTOR = 25.0


def _overlays() -> list[Path]:
    """Every shipped overlay, plus the bare base config as its own case."""
    return sorted(p for p in _CONFIG_DIR.glob("*.yaml") if p.name not in _NOT_AN_OVERLAY)


def _observation() -> MouseDroidObservationBundle:
    """A bundle that is safe on every axis, so only loop timing can trip."""
    return MouseDroidObservationBundle(
        _timestamp=0.0,
        _vision_features=np.zeros(8, dtype=np.float32),
        _distance_m=1.5,
        _motor_state=np.array([0.0, 0.0, 0.0, 12.0], dtype=np.float32),
        _audio_chunk=np.zeros(16, dtype=np.float32),
        _valid_mask=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
    )


def _resolved(overlay: Path) -> Settings:
    """Resolve an overlay the way the application does: base + overlay."""
    if overlay.name == "default.yaml":
        return load_settings(config_dir=_CONFIG_DIR)
    return load_settings(overlay, config_dir=_CONFIG_DIR)


@pytest.mark.parametrize("overlay", _overlays(), ids=lambda p: p.name)
def test_no_shipped_overlay_estops_on_a_cold_first_tick(overlay: Path) -> None:
    """Tick 1 carries tick 0's duration, and tick 0 pays every one-time cost."""
    cfg = _resolved(overlay).safety
    monitor = MouseDroidSafetyMonitor(cfg)
    cold_ms = cfg.max_loop_time_ms * _COLD_START_FACTOR

    # Tick 0 reports 0.0 (no predecessor); tick 1 reports tick 0's real total.
    monitor.evaluate(_observation(), 0.0, tick_index=0)
    ctx = monitor.evaluate(_observation(), cold_ms, tick_index=1)

    assert ctx.is_emergency is False, (
        f"{overlay.name} emergency-stops on a {cold_ms:.0f} ms first tick "
        f"against its {cfg.max_loop_time_ms:.0f} ms ceiling. Feeding the "
        "monitor a real tick duration is correct; doing it with "
        f"loop_overrun_warmup_ticks={cfg.loop_overrun_warmup_ticks} means the "
        "rover e-stops at boot for accelerator warm-up, which is not a fault."
    )


@pytest.mark.parametrize("overlay", _overlays(), ids=lambda p: p.name)
def test_the_interlock_is_still_armed_once_warmup_is_over(overlay: Path) -> None:
    """The anti-cheat pin: boot safety must not be bought by disarming.

    Red if someone ever "fixes" a boot e-stop by widening the ceiling to
    infinity or setting an unbounded warm-up.
    """
    cfg = _resolved(overlay).safety
    monitor = MouseDroidSafetyMonitor(cfg)
    over_budget_ms = cfg.max_loop_time_ms * _COLD_START_FACTOR

    tick = 0
    # Burn the warm-up window with healthy ticks so the grace is spent rather
    # than masking the overrun that follows.
    for _ in range(cfg.loop_overrun_warmup_ticks):
        monitor.evaluate(_observation(), 0.0, tick_index=tick)
        tick += 1

    # Then a sustained overrun, exactly as long as the debounce demands.
    tripped = False
    for _ in range(cfg.loop_overrun_consecutive_ticks):
        tripped = monitor.evaluate(_observation(), over_budget_ms, tick_index=tick).is_emergency
        tick += 1

    assert tripped is True, (
        f"{overlay.name} never trips: {cfg.loop_overrun_consecutive_ticks} "
        f"consecutive ticks at {over_budget_ms:.0f} ms against a "
        f"{cfg.max_loop_time_ms:.0f} ms ceiling left the interlock quiet after "
        f"a {cfg.loop_overrun_warmup_ticks}-tick warm-up. A config that cannot "
        "trip is not boot-safe, it is switched off."
    )


class TestSchemaDefaultsCarryTheGuards:
    """The guards live in ``Field(default=...)``, not only in shipped YAML.

    A ``Settings()`` built in code -- which is what every test fixture and
    every embedder that does not read ``config/`` does -- must be boot-safe
    too. Putting the warm-up solely in ``config/default.yaml`` left
    ``tests/integration/test_e2e_5sec_run.py`` emergency-stopping on a slow
    MCTS plan, because it constructs ``Settings(mock_hardware=True)`` directly
    and never goes through the loader.
    """

    def test_a_bare_config_survives_a_cold_first_tick(self) -> None:
        cfg = SafetyConfig()
        monitor = MouseDroidSafetyMonitor(cfg)
        cold_ms = cfg.max_loop_time_ms * _COLD_START_FACTOR
        assert monitor.evaluate(_observation(), cold_ms, tick_index=0).is_emergency is False

    def test_a_bare_config_still_trips_on_a_sustained_overrun(self) -> None:
        cfg = SafetyConfig()
        monitor = MouseDroidSafetyMonitor(cfg)
        over_budget_ms = cfg.max_loop_time_ms * _COLD_START_FACTOR
        tick = 0
        for _ in range(cfg.loop_overrun_warmup_ticks):
            monitor.evaluate(_observation(), 0.0, tick_index=tick)
            tick += 1
        tripped = False
        for _ in range(cfg.loop_overrun_consecutive_ticks):
            tripped = monitor.evaluate(_observation(), over_budget_ms, tick_index=tick).is_emergency
            tick += 1
        assert tripped is True
