"""Tests for the in-memory mock face driver."""

from __future__ import annotations

import pytest

from mousedroid.config.schema import FaceDisplayConfig
from mousedroid.hardware.display.expressions import Expression
from mousedroid.hardware.display.mock_face_driver import MockFaceDriver
from mousedroid.hardware.protocols import FaceDisplayProtocol


@pytest.fixture
def cfg() -> FaceDisplayConfig:
    return FaceDisplayConfig(enabled=True)


def test_mock_driver_satisfies_protocol(cfg: FaceDisplayConfig) -> None:
    drv = MockFaceDriver(cfg)
    assert isinstance(drv, FaceDisplayProtocol)


async def test_lifecycle_records_in_order(cfg: FaceDisplayConfig) -> None:
    drv = MockFaceDriver(cfg)
    await drv.start()
    await drv.show_expression(Expression.HAPPY)
    await drv.show_text("hello")
    await drv.show_expression(Expression.ALERT)
    await drv.stop()

    assert drv.history == [
        "start",
        "expr:happy",
        "text:hello",
        "expr:alert",
        "stop",
    ]
    assert drv.current is Expression.ALERT
    assert drv.expressions == [Expression.HAPPY, Expression.ALERT]
    assert drv.texts == ["hello"]


async def test_stop_is_idempotent(cfg: FaceDisplayConfig) -> None:
    drv = MockFaceDriver(cfg)
    await drv.start()
    await drv.stop()
    await drv.stop()
    # Only one stop event recorded.
    assert drv.history.count("stop") == 1


async def test_stop_before_start_is_noop(cfg: FaceDisplayConfig) -> None:
    drv = MockFaceDriver(cfg)
    await drv.stop()
    assert drv.history == []
    assert drv.started is False


async def test_tick_blink_restores_previous(cfg: FaceDisplayConfig) -> None:
    drv = MockFaceDriver(cfg)
    await drv.start()
    await drv.show_expression(Expression.HAPPY)
    await drv.tick_blink()
    assert drv.current is Expression.HAPPY
    assert "expr:blink" in drv.history
    # blink occurred between two HAPPY entries
    happy_idx = [i for i, h in enumerate(drv.history) if h == "expr:happy"]
    blink_idx = drv.history.index("expr:blink")
    assert happy_idx[0] < blink_idx < happy_idx[1]
