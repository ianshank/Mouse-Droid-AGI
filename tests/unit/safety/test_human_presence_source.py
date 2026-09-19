"""Human-presence source: the D-1 plumbing is typed and feedable.

Peer review D-1. ``MouseDroidSafetyMonitor.evaluate`` read presence off the
observation with ``getattr(observation, "human_detected", False)``. No such
field existed anywhere, nothing ever assigned one, and the ``getattr``
default meant ``mypy --strict`` could not see the gap -- so three interlocks
were unreachable code and three config budgets had no consumable input.

The decisive test here is
``test_a_live_source_reaches_the_safety_context``: it is impossible to write
against the pre-fix code, because there was no seam to inject through. That
it passes is the proof the plumbing is now feedable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest

from mousedroid.config.schema import SafetyConfig, Settings
from mousedroid.safety.monitor import MouseDroidSafetyMonitor
from mousedroid.sensing.human_presence import (
    HumanPresence,
    HumanPresenceProtocol,
    NullHumanPresenceDetector,
)


@dataclass(frozen=True)
class _LiveHumanPresence:
    """A stand-in for a detector that actually works."""

    detected: bool = True
    distance_m: float = 0.1

    @property
    def source_name(self) -> str:
        return "test_live"

    @property
    def can_detect(self) -> bool:
        return True

    def sample(self, observation: object) -> HumanPresence:
        del observation
        return HumanPresence(detected=self.detected, distance_m=self.distance_m)


def _observation() -> object:
    """A minimal ObservationProtocol-shaped stub."""
    cfg = Settings(mock_hardware=True)

    class _Obs:
        timestamp = 1.0
        vision_features = np.zeros(cfg.model.vision_dim, dtype=np.float32)
        distance_m = 5.0
        motor_state = np.array([0.0, 0.0, 0.0, 12.0], dtype=np.float32)
        audio_chunk = np.zeros(4, dtype=np.float32)
        lidar_features = np.zeros(max(cfg.model.lidar_dim, 1), dtype=np.float32)
        valid_mask = np.ones(5, dtype=bool)
        n_modalities = 5

    return _Obs()


def test_null_detector_reports_nobody() -> None:
    det = NullHumanPresenceDetector()
    assert det.can_detect is False
    assert det.source_name == "null"
    reading = det.sample(_observation())
    assert reading.detected is False
    assert reading.distance_m == math.inf


def test_null_detector_does_not_allocate_per_tick() -> None:
    """``sample`` is on the 30 Hz path; the absent reading is shared."""
    det = NullHumanPresenceDetector()
    assert det.sample(_observation()) is det.sample(_observation())


def test_null_detector_satisfies_the_protocol() -> None:
    assert isinstance(NullHumanPresenceDetector(), HumanPresenceProtocol)


def test_capability_flags_are_real_properties() -> None:
    """A stub with ``can_detect = 1`` would otherwise satisfy isinstance."""
    for name in ("can_detect", "source_name"):
        assert isinstance(NullHumanPresenceDetector.__dict__[name], property)


def test_a_live_source_reaches_the_safety_context() -> None:
    """The D-1 assertion — unwritable against the pre-fix code.

    Before the fix there was no seam: presence came from a ``getattr`` on
    the observation, and no observation type had the field, so no test
    could make ``human_detected`` True through any supported path.
    """
    monitor = MouseDroidSafetyMonitor(
        SafetyConfig(),
        human_presence=_LiveHumanPresence(detected=True, distance_m=0.1),
    )
    ctx = monitor.evaluate(_observation(), 10.0)
    assert ctx.human_detected is True
    assert ctx.human_dist_m == pytest.approx(0.1)


def test_a_close_human_now_trips_the_emergency_branch() -> None:
    """``evaluate``'s human branch fires for the first time."""
    cfg = SafetyConfig()
    monitor = MouseDroidSafetyMonitor(
        cfg,
        human_presence=_LiveHumanPresence(
            detected=True, distance_m=cfg.min_forward_clearance_m / 2.0
        ),
    )
    assert monitor.evaluate(_observation(), 10.0).is_emergency is True


def test_the_default_path_is_behaviourally_unchanged() -> None:
    """Null source must reproduce exactly what the getattr default produced."""
    ctx = MouseDroidSafetyMonitor(SafetyConfig()).evaluate(_observation(), 10.0)
    assert ctx.human_detected is False
    assert ctx.human_dist_m == math.inf


def test_existing_callers_need_no_change() -> None:
    """Positional single-arg construction still works (~20 call sites)."""
    monitor = MouseDroidSafetyMonitor(SafetyConfig())
    assert monitor.evaluate(_observation(), 10.0) is not None


def test_the_getattr_pair_is_gone() -> None:
    """Un-reintroducibility pin: the untyped read must not come back."""
    import inspect

    source = inspect.getsource(MouseDroidSafetyMonitor.evaluate)
    assert 'getattr(observation, "human_detected"' not in source
    assert 'getattr(observation, "human_dist_m"' not in source


def test_observation_protocol_was_not_widened() -> None:
    """The decision pin, with its reason.

    Adding these to ``ObservationProtocol`` breaks all 14 structural
    implementers -- two of them in ``src/``, which fails the blocking
    ``typecheck`` stage -- because a Protocol member's default does not
    help structural implementers. ``_RecordObservation`` could satisfy it
    only by hardcoding ``False``: the same lie, type-checked. A future PR
    that wants to widen the protocol has to argue with this test.
    """
    from mousedroid.sensing.protocol import ObservationProtocol

    assert not hasattr(ObservationProtocol, "human_detected")
    assert not hasattr(ObservationProtocol, "human_dist_m")
