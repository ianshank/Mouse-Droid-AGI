"""Automated Quality Assurance (AQA) -- schema + protocol hygiene for S-2.

S-2 is the peer-review finding that a failed LiDAR read failed OPEN:
``SensorManager._safe_lidar_read`` substituted ``np.ones(feature_dim)``, and
because LiDAR features are normalised range fractions
(``min_in_sector / max_range``) that vector asserted maximum range in every
sector. ``MouseDroidSafetyMonitor`` converted it to ``lidar_max_range_m`` of
clearance in all directions and kept driving.

The fix adds ``SafetyConfig.lidar_unavailable_policy`` and
``lidar_unavailable_grace_s``. Hygiene here is checked off ``model_fields``
(the ``FieldInfo``) rather than a live instance: instantiating only proves the
default is *legal*, not that it is *declared* the way we think, and a refactor
replacing ``Field(...)`` with a plain class attribute must still be caught.
"""

from __future__ import annotations

import inspect

import pytest
from pydantic import ValidationError
from pydantic.fields import FieldInfo

from mousedroid.config.schema import SafetyConfig
from mousedroid.safety.monitor import LIDAR_UNAVAILABLE_DIST_M, MouseDroidSafetyMonitor
from mousedroid.safety.protocol import SafetyMonitorProtocol


def test_lidar_unavailable_policy_has_description() -> None:
    """The field explains the fail-open behaviour it replaces."""
    info: FieldInfo = SafetyConfig.model_fields["lidar_unavailable_policy"]
    assert info.description
    assert len(info.description) > 20, info.description


def test_lidar_unavailable_policy_default_is_ignore() -> None:
    """Default preserves pre-S-2 behaviour (CLAUDE.md invariant 6)."""
    info: FieldInfo = SafetyConfig.model_fields["lidar_unavailable_policy"]
    assert info.default == "ignore"


def test_lidar_unavailable_grace_s_has_description() -> None:
    info: FieldInfo = SafetyConfig.model_fields["lidar_unavailable_grace_s"]
    assert info.description
    assert len(info.description) > 20, info.description


def test_lidar_unavailable_grace_s_default_is_zero() -> None:
    """Zero grace, so opting into a policy does not also opt into a delay."""
    info: FieldInfo = SafetyConfig.model_fields["lidar_unavailable_grace_s"]
    assert info.default == 0.0


@pytest.mark.parametrize("policy", ["ignore", "degrade", "emergency"])
def test_every_documented_policy_loads(policy: str) -> None:
    """The Literal accepts exactly the three values the description names."""
    assert SafetyConfig(lidar_unavailable_policy=policy).lidar_unavailable_policy == policy


def test_unknown_policy_is_rejected_at_load() -> None:
    """A typo must fail at YAML-parse time, not degrade silently to fail-open.

    This is the failure mode that matters: ``lidar_unavailable_policy:
    emergancy`` silently falling back to ``ignore`` would give an operator a
    safety interlock they believe is armed and is not.
    """
    with pytest.raises(ValidationError, match=r"lidar_unavailable_policy"):
        SafetyConfig(lidar_unavailable_policy="emergancy")


def test_negative_grace_is_rejected_at_load() -> None:
    """``ge=0`` -- a negative window is meaningless, not a stricter one."""
    with pytest.raises(ValidationError, match=r"lidar_unavailable_grace_s"):
        SafetyConfig(lidar_unavailable_grace_s=-1.0)


def test_positive_grace_loads_cleanly() -> None:
    """The same validator, satisfied -- proves it is not always-raise."""
    assert SafetyConfig(lidar_unavailable_grace_s=0.25).lidar_unavailable_grace_s == 0.25


def test_worst_case_distance_is_not_configurable() -> None:
    """The fail-closed clearance is a constant, deliberately.

    Every clearance comparison in the monitor and the projector is a strict
    ``<`` against a ``gt=0`` threshold, so 0.0 fails all of them no matter how
    an operator tunes ``min_forward_clearance_m``. Exposing it as config would
    only add a way to configure the fail-closed path back open, so there must
    be no ``SafetyConfig`` field carrying this value.
    """
    assert LIDAR_UNAVAILABLE_DIST_M == 0.0
    assert not [
        name for name in SafetyConfig.model_fields if "unavailable" in name and "dist" in name
    ]


def test_monitor_still_satisfies_safety_monitor_protocol() -> None:
    """Bare ``isinstance`` only proves attribute presence, so check arity too.

    ``evaluate`` grew no parameters in this change -- the new behaviour is
    driven entirely by config -- and this pins that, because a signature change
    here would silently break every ``SafetyMonitorProtocol`` test double.
    """
    monitor = MouseDroidSafetyMonitor(SafetyConfig())
    assert isinstance(monitor, SafetyMonitorProtocol)
    sig = inspect.signature(monitor.evaluate)
    assert list(sig.parameters) == ["observation", "loop_time_ms", "tick_index"]
    assert sig.parameters["tick_index"].kind is inspect.Parameter.KEYWORD_ONLY


def test_clearance_helper_returns_the_documented_triple() -> None:
    """``_evaluate_lidar_clearance`` is the seam the tiers above drive."""
    monitor = MouseDroidSafetyMonitor(SafetyConfig(lidar_unavailable_policy="emergency"))
    result = monitor._evaluate_lidar_clearance(None, 0.0)
    assert result == (LIDAR_UNAVAILABLE_DIST_M, False, True)
