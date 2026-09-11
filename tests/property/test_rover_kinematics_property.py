"""Hypothesis properties for skid-steer wheel ↔ body conversion."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mousedroid.sim.kinematics import (
    body_velocity_to_wheels,
    policy_action_to_body,
    wheels_to_body_velocity,
)

_finite = st.floats(min_value=-30.0, max_value=30.0, allow_nan=False, allow_infinity=False)
_radius = st.floats(min_value=0.01, max_value=0.2, allow_nan=False, allow_infinity=False)
_track = st.floats(min_value=0.05, max_value=0.6, allow_nan=False, allow_infinity=False)


@given(left=_finite, right=_finite, radius=_radius, track=_track)
@settings(max_examples=80)
def test_wheels_body_round_trip(left: float, right: float, radius: float, track: float) -> None:
    vx, omega = wheels_to_body_velocity(left, right, wheel_radius_m=radius, track_width_m=track)
    left2, right2 = body_velocity_to_wheels(vx, omega, wheel_radius_m=radius, track_width_m=track)
    assert left2 == pytest.approx(left, rel=1e-5, abs=1e-6)
    assert right2 == pytest.approx(right, rel=1e-5, abs=1e-6)


@given(vx=_finite, omega=_finite, radius=_radius, track=_track)
@settings(max_examples=80)
def test_body_wheels_round_trip(vx: float, omega: float, radius: float, track: float) -> None:
    left, right = body_velocity_to_wheels(vx, omega, wheel_radius_m=radius, track_width_m=track)
    vx2, omega2 = wheels_to_body_velocity(left, right, wheel_radius_m=radius, track_width_m=track)
    assert vx2 == pytest.approx(vx, rel=1e-5, abs=1e-6)
    assert omega2 == pytest.approx(omega, rel=1e-5, abs=1e-6)


@given(left=_finite, right=_finite, radius=_radius, track=_track)
@settings(max_examples=40)
def test_policy_clip_fan_out_is_finite(
    left: float, right: float, radius: float, track: float
) -> None:
    action = np.array([left, right], dtype=np.float32)
    body = policy_action_to_body(
        action,
        mode="differential",
        wheel_radius_m=radius,
        track_width_m=track,
    )
    assert np.isfinite(body).all()
    assert body.shape == (3,)
    assert body[1] == pytest.approx(0.0)
