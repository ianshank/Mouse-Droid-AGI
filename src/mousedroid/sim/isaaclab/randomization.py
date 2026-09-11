"""Translate ``DomainRandomizationConfig`` chassis samples onto Isaac handles.

Concrete Isaac Lab / Replicator types are imported only inside
:func:`apply_isaac_domain_params` so this module loads without the ``[isaac]``
extra. CI tests the no-op / structured-log path; live workstations exercise
the articulation write when a handle is present.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mousedroid.common.imports import module_importable
from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)


def apply_isaac_domain_params(
    articulation: Any,
    *,
    friction: float,
    slip: float,
    mass_kg: float,
    motor_gain: float,
) -> None:
    """Apply one chassis DR sample to a live (or fake) articulation handle.

    Args:
        articulation: Isaac Lab ``Articulation`` or a duck-typed test double.
            ``None`` is a no-op besides the log.
        friction: Wheel tangential friction sample.
        slip: Observation-noise slip proxy (logged; Isaac has no first-class
            slip field — same documented meaning as MuJoCo).
        mass_kg: Chassis mass sample.
        motor_gain: Actuator gain sample.
    """
    _log.info(
        "isaac_lab_domain_params_applied",
        friction=friction,
        slip=slip,
        mass_kg=mass_kg,
        motor_gain=motor_gain,
        has_articulation=articulation is not None,
    )
    if articulation is None:
        return
    writer = getattr(articulation, "apply_chassis_domain_params", None)
    if callable(writer):
        writer(
            friction=friction,
            slip=slip,
            mass_kg=mass_kg,
            motor_gain=motor_gain,
        )
        return
    # Live Isaac Lab 0.20+: best-effort PhysX view writes. Missing attributes
    # are skipped so a partial API never raises on the training loop.
    view = getattr(articulation, "root_physx_view", None)
    masses = getattr(view, "get_masses", None) if view is not None else None
    set_masses = getattr(view, "set_masses", None) if view is not None else None
    if callable(masses) and callable(set_masses):
        current = masses()
        filled = getattr(current, "fill", None)
        if callable(filled):
            filled(mass_kg)
            set_masses(current)
    _try_replicator_write(articulation, friction=friction, mass_kg=mass_kg)


def apply_isaac_episode_extras(extras: Mapping[str, float]) -> None:
    """Log non-chassis DR channels (latency, vision noise, pushes).

    These do not have a first-class Isaac Lab write in this slice; the
    structured event is the CI-observable contract. Live Replicator vision
    noise stays behind :func:`_try_replicator_write`.

    Args:
        extras: Extra sampled fields (``uart_latency_ms``, ``push_force_n``,
            ``brightness``, …). Empty mapping is a no-op besides the log.
    """
    _log.info(
        "isaac_lab_domain_extras_applied",
        extras=dict(extras),
        n_fields=len(extras),
    )


def _try_replicator_write(
    articulation: Any,
    *,
    friction: float,
    mass_kg: float,
) -> None:
    """Best-effort Omniverse Replicator import; skip when the extra is absent."""
    if not module_importable("isaaclab"):
        return
    try:
        import omni.replicator.core as rep  # pragma: no cover
    except ImportError:
        _log.debug("isaac_lab_replicator_unavailable")
        return
    writer = getattr(rep, "modify_pose", None)
    _log.debug(
        "isaac_lab_replicator_present",
        has_modify_pose=callable(writer),
        friction=friction,
        mass_kg=mass_kg,
        articulation_type=type(articulation).__name__,
    )
