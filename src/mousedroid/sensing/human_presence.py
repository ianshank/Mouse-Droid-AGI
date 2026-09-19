"""Human-presence source for the safety interlocks (peer review D-1).

``MouseDroidSafetyMonitor.evaluate`` used to read human presence off the
observation defensively::

    human_detected = bool(getattr(observation, "human_detected", False))
    human_dist_m = float(getattr(observation, "human_dist_m", float("inf")))

No such field exists on :class:`~mousedroid.sensing.protocol.ObservationProtocol`
or :class:`~mousedroid.sensing.bundle.MouseDroidObservationBundle`, and nothing
in ``src/`` ever assigned one. So ``SafetyContext.human_detected`` was
permanently ``False`` and three interlocks were unreachable code: the Law-1
full stop in :meth:`~mousedroid.agents.navigation.MouseDroidNavigationAgent.act`,
the human-proximity clamp in
:meth:`~mousedroid.safety.projector.GeometricSafetyProjector.project`, and the
human branch of ``evaluate`` itself. Three declared config budgets --
``SafetyProjectorConfig.human_keepout_m``, ``human_proximity_speed_mps`` and
``ThreeLawsConfig.human_safety_radius_m`` -- had no consumable input.

**The ``getattr`` default is what hid it.** A typed member would have failed
``mypy --strict``. But putting the field on ``ObservationProtocol`` is the
wrong repair: that protocol is the minimum fused-sensor contract feeding the
RSSM encoder, it is ``@runtime_checkable``, and a Protocol member has no
usable default for *structural* implementers -- so adding one breaks all 14
of them, two in ``src/`` (``MouseDroidObservationBundle`` and
``learning.on_device.seed_states._RecordObservation``), failing the blocking
``typecheck`` stage. An experience record has no human channel at all, so it
could only satisfy the member by hardcoding ``False`` -- the same lie,
type-checked.

So presence becomes its own typed, factory-built collaborator instead. The
gap does not close -- no detector ships -- but it becomes *visible*: the
monitor holds a ``HumanPresenceProtocol``, the call is type-checked, and
``NullHumanPresenceDetector.can_detect`` is ``False``, which the monitor
reports once at construction. That mirrors how S-11 was fixed for the
chassis heartbeat: the dangerous case used to be silent and
indistinguishable from success, so it was made to announce itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol, runtime_checkable

if TYPE_CHECKING:
    from mousedroid.sensing.protocol import ObservationProtocol

__all__ = ["HumanPresence", "HumanPresenceProtocol", "NullHumanPresenceDetector"]

NULL_SOURCE_NAME: Final[str] = "null"
"""Identifier the Null detector reports, so logs name the source."""


@dataclass(frozen=True, slots=True)
class HumanPresence:
    """One human-presence reading.

    ``distance_m`` defaults to infinity rather than zero so an absent
    reading reads as "nobody anywhere near", matching
    :attr:`mousedroid.safety.context.SafetyContext.human_dist_m`'s own
    default. Zero would mean "a human is touching the rover" and would
    e-stop permanently.
    """

    detected: bool = False
    distance_m: float = math.inf


_ABSENT: Final[HumanPresence] = HumanPresence()
"""Module-level singleton returned by the Null detector.

``sample`` is on the 30 Hz path, so the no-detector case must not allocate
per tick. Safe to share because :class:`HumanPresence` is frozen.
"""


@runtime_checkable
class HumanPresenceProtocol(Protocol):
    """A source of human-presence readings for the safety layer."""

    @property
    def source_name(self) -> str:
        """Short identifier for logs and telemetry."""
        ...

    @property
    def can_detect(self) -> bool:
        """Whether this source is *structurally capable* of seeing a human.

        ``False`` means it can never report one, whatever the world does --
        the interlocks that depend on it are inert and an operator should
        be told. This is deliberately distinct from a live source that is
        momentarily unhealthy: that returns ``True`` here and reports
        ``detected=False`` from :meth:`sample`, because "my camera is
        blocked" and "there is provably no human detector" are different
        facts and only the second is a permanent capability gap.
        """
        ...

    def sample(self, observation: ObservationProtocol) -> HumanPresence:
        """Return the current reading.

        Takes the observation so a future classifier over
        ``observation.vision_features``, or a ``SensorManager``-stamped
        field, both fit without changing this seam or its callers.
        """
        ...


class NullHumanPresenceDetector:
    """The only implementation today: reports nobody, always.

    Named rather than implicit so the absence is greppable and so
    :attr:`can_detect` can say ``False`` out loud. Mirrors
    :class:`~mousedroid.health.watchdog.NullNotifier` and
    :class:`~mousedroid.harness.hooks.NullHookRegistry`.
    """

    @property
    def source_name(self) -> str:
        """Identify this source in logs."""
        return NULL_SOURCE_NAME

    @property
    def can_detect(self) -> bool:
        """Always ``False`` -- there is no detector wired in this repo."""
        return False

    def sample(self, observation: ObservationProtocol) -> HumanPresence:
        """Return the shared absent reading without allocating."""
        del observation  # no source to consult
        return _ABSENT


# Static conformance check, evaluated at import and discarded. Mirrors
# ``harness/hooks.py``: ``runtime_checkable`` only proves attribute
# presence, so this makes mypy verify the shape as well.
_PROTOCOL_CHECK_NULL: HumanPresenceProtocol = NullHumanPresenceDetector()
del _PROTOCOL_CHECK_NULL
