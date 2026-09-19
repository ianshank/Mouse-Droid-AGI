"""Emergency-stop latch: fail-closed, hot-loop-safe, restart-surviving.

Peer review D-5. ``MouseDroidSafetyMonitor.evaluate`` recomputed
``is_emergency`` from ``False`` every tick, so the instant a triggering
condition cleared the next tick resumed driving with no human in the loop --
and the shipped units (`Restart=on-failure`, `restart: unless-stopped`) meant
a restart cleared it too. ISO 3691-4 requires reset only by deliberate human
action.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mousedroid.config.schema import SafetyConfig
from mousedroid.safety.latch import (
    LATCH_FILENAME,
    LATCH_SCHEMA_VERSION,
    EmergencyLatchProtocol,
    FileEmergencyLatch,
)
from mousedroid.safety.monitor import MouseDroidSafetyMonitor


def _latch(tmp_path: Path) -> FileEmergencyLatch:
    return FileEmergencyLatch(tmp_path / "estop_latch", fsync=False)


def test_latch_satisfies_the_protocol(tmp_path: Path) -> None:
    assert isinstance(_latch(tmp_path), EmergencyLatchProtocol)


def test_trip_performs_no_file_io(tmp_path: Path) -> None:
    """The hot-loop guarantee: ``trip`` runs at 30 Hz and must not touch disk.

    This is the assertion that fails if someone inlines the write into
    ``trip`` — which would also land on the MCP tool bridge's async request
    path, since that calls ``evaluate`` too.
    """
    latch = _latch(tmp_path)
    assert latch.trip("forward_clearance_violation") is True
    assert not (tmp_path / "estop_latch").exists()


def test_trip_reports_only_the_transition(tmp_path: Path) -> None:
    """So a caller can log the trip exactly once at 30 Hz."""
    latch = _latch(tmp_path)
    assert latch.trip("battery_critical") is True
    assert latch.trip("battery_critical") is False
    assert latch.is_latched is True


async def test_persist_then_load_round_trips(tmp_path: Path) -> None:
    latch = _latch(tmp_path)
    latch.trip("lidar_emergency", causes=("lidar_emergency", "sensor_stale"))
    await latch.persist()

    fresh = _latch(tmp_path)
    assert await fresh.load() is True
    assert fresh.record.reason == "lidar_emergency"
    assert fresh.record.causes == ("lidar_emergency", "sensor_stale")
    assert fresh.record.tripped_at_iso


async def test_absence_is_the_only_unlatched_state(tmp_path: Path) -> None:
    assert await _latch(tmp_path).load() is False


async def test_no_temp_file_is_left_behind(tmp_path: Path) -> None:
    latch = _latch(tmp_path)
    latch.trip("loop_overrun")
    await latch.persist()
    assert [p.name for p in (tmp_path / "estop_latch").iterdir()] == [LATCH_FILENAME]


@pytest.mark.parametrize(
    "payload",
    [
        "{not json",
        "[]",
        '{"latched": "yes", "schema_version": 1}',
        '{"latched": true}',
        '{"latched": true, "schema_version": 99}',
        "null",
    ],
    ids=[
        "unparseable",
        "not-an-object",
        "latched-not-bool",
        "no-version",
        "future-version",
        "json-null",
    ],
)
async def test_damaged_records_fail_closed(tmp_path: Path, payload: str) -> None:
    """A corrupt record means a latch was written and the write failed.

    Deliberately inverts ``slot_store.load_active``, which returns ``None``
    on a corrupt manifest — there "no active slot" is the safe answer, here
    the safe answer is the opposite.
    """
    latch = _latch(tmp_path)
    latch.path.parent.mkdir(parents=True, exist_ok=True)
    latch.path.write_text(payload, encoding="utf-8")
    assert await latch.load() is True


async def test_a_tombstone_reads_as_clear(tmp_path: Path) -> None:
    """An explicit ``latched: false`` is a deliberate record, not damage."""
    latch = _latch(tmp_path)
    latch.path.parent.mkdir(parents=True, exist_ok=True)
    latch.path.write_text(
        json.dumps({"schema_version": LATCH_SCHEMA_VERSION, "latched": False}), encoding="utf-8"
    )
    assert await latch.load() is False


async def test_rearm_clears_and_is_idempotent(tmp_path: Path) -> None:
    latch = _latch(tmp_path)
    latch.trip("battery_critical")
    await latch.persist()

    assert await latch.rearm(operator="tester") is True
    assert latch.is_latched is False
    assert not latch.path.exists()
    assert await latch.rearm(operator="tester") is False


def test_no_expiry_knob_exists() -> None:
    """An age-based auto-clear IS an automatic restart after an e-stop.

    Pinned so a future "just add a TTL" change has to argue with this test.
    """
    fields = set(type(SafetyConfig().emergency_latch).model_fields)
    assert not any("age" in f or "ttl" in f or "expire" in f for f in fields)


def test_no_fail_open_knob_exists() -> None:
    """Making the fail-closed read tunable would only configure it back open."""
    fields = set(type(SafetyConfig().emergency_latch).model_fields)
    assert not any("fail_open" in f or "corrupt" in f for f in fields)


# -- the defect itself ------------------------------------------------------


def _obs():  # type: ignore[no-untyped-def]
    from unittest.mock import MagicMock

    import numpy as np

    obs = MagicMock()
    obs.distance_m = 5.0
    obs.motor_state = np.array([0.0, 0.0, 0.0, 12.0], dtype=np.float32)
    obs.valid_mask = np.ones(4, dtype=np.float32)
    obs.timestamp = 1.0
    # Normalised range fractions: all-ones is maximum range in every
    # sector, i.e. clear. All-zeros would read as an obstacle at 0 m and
    # trip the LiDAR interlock instead of the one under test.
    obs.lidar_features = np.ones(1, dtype=np.float32)
    return obs


def test_the_emergency_no_longer_clears_itself(tmp_path: Path) -> None:
    """The D-5 assertion: a clean tick must not resume driving.

    Without a latch this is exactly what happened — ``is_emergency`` went
    True on the tripping tick and False on the very next clean one.
    """
    cfg = SafetyConfig(min_forward_clearance_m=10.0)  # trips on distance 5.0
    monitor = MouseDroidSafetyMonitor(cfg, latch=_latch(tmp_path))
    assert monitor.evaluate(_obs(), 10.0).is_emergency is True

    clear_cfg = SafetyConfig(min_forward_clearance_m=0.2)  # would not trip
    monitor._cfg = clear_cfg  # type: ignore[attr-defined]
    for _ in range(5):
        ctx = monitor.evaluate(_obs(), 10.0)
        assert ctx.is_emergency is True, "the latch released without a human"
        assert ctx.emergency_latched is True


def test_without_a_latch_the_legacy_behaviour_is_preserved(tmp_path: Path) -> None:
    """``enabled: false`` must be genuinely inert, not merely defaulted off."""
    monitor = MouseDroidSafetyMonitor(SafetyConfig(min_forward_clearance_m=10.0))
    assert monitor.evaluate(_obs(), 10.0).is_emergency is True
    monitor._cfg = SafetyConfig(min_forward_clearance_m=0.2)  # type: ignore[attr-defined]
    ctx = monitor.evaluate(_obs(), 10.0)
    assert ctx.is_emergency is False
    assert ctx.emergency_latched is False
