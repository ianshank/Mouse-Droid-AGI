"""Property test: every (valence, arousal) input yields a valid Expression."""

from __future__ import annotations

import asyncio

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mousedroid.config.schema import FaceDisplayConfig
from mousedroid.hardware.display.expressions import Expression
from mousedroid.hardware.display.mock_face_driver import MockFaceDriver
from mousedroid.orchestrator.face_controller import FaceController


@settings(max_examples=200, deadline=None)
@given(
    valence=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
    arousal=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
    is_emergency=st.booleans(),
    is_idle=st.booleans(),
)
def test_controller_always_returns_valid_expression(
    valence: float,
    arousal: float,
    is_emergency: bool,
    is_idle: bool,
) -> None:
    """For any affect input, the controller chooses some valid Expression."""
    cfg = FaceDisplayConfig(enabled=True, min_dwell_s=0.0)
    drv = MockFaceDriver(cfg)
    fc = FaceController(drv, cfg, clock=lambda: 0.0)

    async def go() -> None:
        await fc.start()
        await fc.update(
            valence=valence,
            arousal=arousal,
            is_emergency=is_emergency,
            is_idle=is_idle,
        )

    asyncio.run(go())
    assert drv.current is not None
    assert isinstance(drv.current, Expression)


@pytest.mark.parametrize("expr", list(Expression))
def test_every_expression_is_string_serialisable(expr: Expression) -> None:
    """Expression is a ``str`` enum — values round-trip through string."""
    assert isinstance(expr.value, str)
    assert Expression(expr.value) is expr
