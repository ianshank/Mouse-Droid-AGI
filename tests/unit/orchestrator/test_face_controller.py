"""Tests for the affect → expression mapping in :class:`FaceController`."""

from __future__ import annotations

import pytest

from mousedroid.config.schema import FaceDisplayConfig
from mousedroid.hardware.display.expressions import Expression
from mousedroid.hardware.display.mock_face_driver import MockFaceDriver
from mousedroid.orchestrator.face_controller import FaceController


class _FakeClock:
    """Deterministic monotonic-clock substitute for hysteresis tests."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, dt: float) -> None:
        self.now += dt


@pytest.fixture
def cfg() -> FaceDisplayConfig:
    return FaceDisplayConfig(enabled=True, min_dwell_s=0.5, idle_sleepy_after_s=10.0)


@pytest.fixture
def clock() -> _FakeClock:
    return _FakeClock()


@pytest.fixture
def controller(cfg: FaceDisplayConfig, clock: _FakeClock) -> tuple[FaceController, MockFaceDriver]:
    drv = MockFaceDriver(cfg)
    fc = FaceController(drv, cfg, clock=clock)
    return fc, drv


@pytest.mark.parametrize(
    ("valence", "arousal", "is_emergency", "is_idle", "expected"),
    [
        (0.0, 0.0, True, False, Expression.EMERGENCY),
        (0.9, 0.9, True, False, Expression.EMERGENCY),  # emergency wins over affect
        (0.6, 0.0, False, False, Expression.HAPPY),
        (-0.6, 0.0, False, False, Expression.SAD),
        (0.0, 0.8, False, False, Expression.ALERT),
        (-0.5, 0.6, False, False, Expression.ANGRY),  # angry beats alert+sad
        (0.0, -0.6, False, False, Expression.SLEEPY),
        (0.0, 0.0, False, False, Expression.NEUTRAL),
    ],
)
async def test_classification_table(
    controller: tuple[FaceController, MockFaceDriver],
    clock: _FakeClock,
    cfg: FaceDisplayConfig,
    valence: float,
    arousal: float,
    is_emergency: bool,
    is_idle: bool,
    expected: Expression,
) -> None:
    fc, drv = controller
    await fc.start()
    # Skip past dwell so the first transition is unblocked.
    clock.advance(cfg.min_dwell_s + 0.01)
    await fc.update(
        valence=valence,
        arousal=arousal,
        is_emergency=is_emergency,
        is_idle=is_idle,
    )
    assert drv.current is expected
    assert fc.current is expected


async def test_dwell_blocks_quick_flicker(
    controller: tuple[FaceController, MockFaceDriver],
    clock: _FakeClock,
    cfg: FaceDisplayConfig,
) -> None:
    fc, drv = controller
    await fc.start()
    clock.advance(cfg.min_dwell_s + 0.01)
    await fc.update(valence=0.6, arousal=0.0, is_emergency=False, is_idle=False)
    assert drv.current is Expression.HAPPY

    # Within the dwell window, attempt to flip — must be ignored.
    clock.advance(cfg.min_dwell_s / 2.0)
    await fc.update(valence=-0.6, arousal=0.0, is_emergency=False, is_idle=False)
    assert drv.current is Expression.HAPPY

    # After the dwell elapses, the next change goes through.
    clock.advance(cfg.min_dwell_s)
    await fc.update(valence=-0.6, arousal=0.0, is_emergency=False, is_idle=False)
    assert drv.current is Expression.SAD


async def test_emergency_bypasses_dwell(
    controller: tuple[FaceController, MockFaceDriver],
    clock: _FakeClock,
    cfg: FaceDisplayConfig,
) -> None:
    fc, drv = controller
    await fc.start()
    clock.advance(cfg.min_dwell_s + 0.01)
    await fc.update(valence=0.6, arousal=0.0, is_emergency=False, is_idle=False)
    assert drv.current is Expression.HAPPY

    # No time advance — emergency must still take effect.
    await fc.update(valence=0.6, arousal=0.0, is_emergency=True, is_idle=False)
    assert drv.current is Expression.EMERGENCY


async def test_idle_eventually_goes_sleepy(
    controller: tuple[FaceController, MockFaceDriver],
    clock: _FakeClock,
    cfg: FaceDisplayConfig,
) -> None:
    fc, drv = controller
    await fc.start()

    # First idle update before idle_sleepy_after_s — stays NEUTRAL.
    clock.advance(cfg.min_dwell_s + 0.01)
    await fc.update(valence=0.0, arousal=0.0, is_emergency=False, is_idle=True)
    assert drv.current is Expression.NEUTRAL

    # After enough idle time, controller switches to SLEEPY.
    clock.advance(cfg.idle_sleepy_after_s + 1.0)
    await fc.update(valence=0.0, arousal=0.0, is_emergency=False, is_idle=True)
    assert drv.current is Expression.SLEEPY

    # Activity resets idle and we go back through NEUTRAL when affect is flat.
    clock.advance(cfg.min_dwell_s + 0.01)
    await fc.update(valence=0.0, arousal=0.0, is_emergency=False, is_idle=False)
    assert drv.current is Expression.NEUTRAL


async def test_start_renders_neutral_first(
    controller: tuple[FaceController, MockFaceDriver],
) -> None:
    fc, drv = controller
    await fc.start()
    assert drv.current is Expression.NEUTRAL
    assert drv.history[:2] == ["start", "expr:neutral"]


async def test_no_redundant_writes_when_expression_unchanged(
    controller: tuple[FaceController, MockFaceDriver],
    clock: _FakeClock,
    cfg: FaceDisplayConfig,
) -> None:
    fc, drv = controller
    await fc.start()
    clock.advance(cfg.min_dwell_s + 0.01)
    await fc.update(valence=0.6, arousal=0.0, is_emergency=False, is_idle=False)
    happy_count_before = drv.expressions.count(Expression.HAPPY)
    clock.advance(cfg.min_dwell_s + 0.01)
    await fc.update(valence=0.6, arousal=0.0, is_emergency=False, is_idle=False)
    assert drv.expressions.count(Expression.HAPPY) == happy_count_before
