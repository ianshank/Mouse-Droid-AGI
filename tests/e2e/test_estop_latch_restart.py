"""The D-5 headline: an emergency stop survives a process restart.

The defect had two halves. ``MouseDroidSafetyMonitor.evaluate`` recomputed
``is_emergency`` from ``False`` each tick, so a clean tick resumed driving;
and ``scripts/mousedroid.service`` / ``scripts/mousedroid-docker.service``
both set ``Restart=on-failure`` while ``docker-compose.jetson.yml`` sets
``restart: unless-stopped``, so a restart cleared it too. ISO 3691-4 -- the
standard the external review recommends adopting -- forbids exactly that.

This test builds an orchestrator through the real factory, trips the latch,
tears it down, and builds a **second** orchestrator over the same experience
root. The second one must come up already latched.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from mousedroid.config.schema import Settings


def _settings(tmp_path: Path) -> Settings:
    cfg = Settings(mock_hardware=True)
    cfg.experience.path = str(tmp_path)  # type: ignore[misc]
    cfg.safety.emergency_latch.enabled = True  # type: ignore[misc]
    cfg.safety.emergency_latch.fsync = False  # type: ignore[misc]
    return cfg


def _mock_esp32() -> AsyncMock:
    esp32 = AsyncMock()
    esp32.connect = AsyncMock()
    esp32.disconnect = AsyncMock()
    esp32.send_velocity = AsyncMock()
    esp32.emergency_stop = AsyncMock()
    esp32.read_encoders = AsyncMock(
        return_value=MagicMock(left_velocity_mps=0.0, right_velocity_mps=0.0, heading_rad=0.0)
    )
    return esp32


async def test_a_latched_estop_survives_a_restart(tmp_path: Path) -> None:
    from mousedroid.factory import build_emergency_latch, build_orchestrator

    cfg = _settings(tmp_path)

    # --- run 1: trip and persist ---------------------------------------
    latch = build_emergency_latch(cfg)
    assert latch is not None
    latch.trip("forward_clearance_violation", causes=("forward_clearance_violation",))
    await latch.persist()
    assert latch.path.exists(), "nothing was written for the next run to find"

    # --- run 2: a brand-new process over the same experience root -------
    orch_b = build_orchestrator(cfg)
    esp32_b = _mock_esp32()
    orch_b._esp32 = esp32_b  # type: ignore[attr-defined]
    try:
        await orch_b.start()  # type: ignore[attr-defined]
        # It STARTED rather than refusing: a non-zero exit would meet
        # Restart=on-failure on a unit with no StartLimitBurst and
        # crash-loop, and a crash-looping process never reaches
        # _halt_actuators, so nobody would be driving the brake.
        assert orch_b._emergency_latch is not None  # type: ignore[attr-defined]
        assert orch_b._emergency_latch.is_latched is True  # type: ignore[attr-defined]
        assert orch_b._emergency_latch.record.reason == "forward_clearance_violation"  # type: ignore[attr-defined]
    finally:
        await orch_b.stop()  # type: ignore[attr-defined]


async def test_an_unlatched_root_starts_normally(tmp_path: Path) -> None:
    """The control: absence of a record is the only unlatched state."""
    from mousedroid.factory import build_orchestrator

    orch = build_orchestrator(_settings(tmp_path))
    orch._esp32 = _mock_esp32()  # type: ignore[attr-defined]
    try:
        await orch.start()  # type: ignore[attr-defined]
        assert orch._emergency_latch is not None  # type: ignore[attr-defined]
        assert orch._emergency_latch.is_latched is False  # type: ignore[attr-defined]
    finally:
        await orch.stop()  # type: ignore[attr-defined]


async def test_the_rearm_cli_is_the_way_back(tmp_path: Path) -> None:
    """Deliberate human action clears it; nothing else does."""
    from mousedroid.cli.rearm import EXIT_LATCHED, EXIT_OK, EXIT_REFUSED, _run
    from mousedroid.factory import build_emergency_latch

    cfg = _settings(tmp_path)
    latch = build_emergency_latch(cfg)
    assert latch is not None
    latch.trip("battery_critical", causes=("battery_critical",))
    await latch.persist()

    import argparse

    def _args(**kw: object) -> argparse.Namespace:
        base = {"config": None, "operator": None, "confirm_area_clear": False, "status": False}
        base.update(kw)
        return argparse.Namespace(**base)

    import mousedroid.cli.rearm as rearm_mod

    original = rearm_mod.load_settings
    rearm_mod.load_settings = lambda *a, **k: cfg  # type: ignore[assignment]
    try:
        assert await _run(_args(status=True)) == EXIT_LATCHED
        assert latch.path.exists(), "--status must not clear anything"

        assert await _run(_args(operator="tester")) == EXIT_REFUSED
        assert latch.path.exists(), "a missing --confirm-area-clear must not clear"

        assert await _run(_args(confirm_area_clear=True)) == EXIT_REFUSED
        assert latch.path.exists(), "an unattributed re-arm is not a re-arm"

        assert await _run(_args(operator="tester", confirm_area_clear=True)) == EXIT_OK
        assert not latch.path.exists()
    finally:
        rearm_mod.load_settings = original  # type: ignore[assignment]
