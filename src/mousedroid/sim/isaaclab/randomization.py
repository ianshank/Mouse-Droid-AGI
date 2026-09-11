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

_CHASSIS_KEYS: tuple[str, ...] = ("friction", "slip", "mass_kg", "motor_gain")


def domain_write_coverage(
    *,
    duck_writer: bool,
    mass_written: bool,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Classify which chassis DR fields a handle could actually write.

    Args:
        duck_writer: True when ``apply_chassis_domain_params`` is present.
        mass_written: True when the PhysX mass path succeeded.

    Returns:
        ``(wrote, skipped)`` field-name tuples in chassis-key order.
    """
    if duck_writer:
        return _CHASSIS_KEYS, ()
    wrote_list: list[str] = []
    if mass_written:
        wrote_list.append("mass_kg")
    wrote = tuple(wrote_list)
    skipped = tuple(name for name in _CHASSIS_KEYS if name not in wrote)
    return wrote, skipped


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
        _log_domain_coverage(
            wrote=(),
            skipped=_CHASSIS_KEYS,
            friction=friction,
            slip=slip,
            mass_kg=mass_kg,
            motor_gain=motor_gain,
        )
        return
    writer = getattr(articulation, "apply_chassis_domain_params", None)
    if callable(writer):
        writer(
            friction=friction,
            slip=slip,
            mass_kg=mass_kg,
            motor_gain=motor_gain,
        )
        wrote, skipped = domain_write_coverage(duck_writer=True, mass_written=True)
        _log_domain_coverage(
            wrote=wrote,
            skipped=skipped,
            friction=friction,
            slip=slip,
            mass_kg=mass_kg,
            motor_gain=motor_gain,
        )
        return
    # Live Isaac Lab 0.20+: best-effort PhysX view writes. Missing attributes
    # are skipped so a partial API never raises on the training loop.
    mass_written = _write_physx_mass(articulation, mass_kg)
    _try_replicator_write(articulation, friction=friction, mass_kg=mass_kg)
    wrote, skipped = domain_write_coverage(duck_writer=False, mass_written=mass_written)
    _log_domain_coverage(
        wrote=wrote,
        skipped=skipped,
        friction=friction,
        slip=slip,
        mass_kg=mass_kg,
        motor_gain=motor_gain,
    )


def apply_isaac_episode_extras(extras: Mapping[str, float]) -> None:
    """Log non-chassis DR channels (latency, vision noise, pushes).

    These do not have a first-class Isaac Lab write in this slice; the
    structured event is the CI-observable contract. Replicator is a
    present-check only (:func:`_try_replicator_write` does not write).

    Args:
        extras: Extra sampled fields (``uart_latency_ms``, ``push_force_n``,
            ``brightness``, …). Empty mapping is a no-op besides the log.
    """
    _log.info(
        "isaac_lab_domain_extras_applied",
        extras=dict(extras),
        n_fields=len(extras),
    )


def _write_physx_mass(articulation: Any, mass_kg: float) -> bool:
    """Best-effort PhysX mass write. Returns True when the buffer was filled."""
    view = getattr(articulation, "root_physx_view", None)
    masses = getattr(view, "get_masses", None) if view is not None else None
    set_masses = getattr(view, "set_masses", None) if view is not None else None
    if not callable(masses) or not callable(set_masses):
        return False
    current = masses()
    filled = getattr(current, "fill", None)
    if not callable(filled):
        return False
    filled(mass_kg)
    set_masses(current)
    return True


def _log_domain_coverage(
    *,
    wrote: tuple[str, ...],
    skipped: tuple[str, ...],
    friction: float,
    slip: float,
    mass_kg: float,
    motor_gain: float,
) -> None:
    """Log partial / unapplied DR writes without raising on a live loop."""
    if not wrote:
        _log.info(
            "isaac_lab_domain_params_unapplied",
            wrote=wrote,
            skipped=skipped,
            friction=friction,
            slip=slip,
            mass_kg=mass_kg,
            motor_gain=motor_gain,
        )
        return
    if skipped:
        _log.info(
            "isaac_lab_domain_params_partial",
            wrote=wrote,
            skipped=skipped,
            friction=friction,
            slip=slip,
            mass_kg=mass_kg,
            motor_gain=motor_gain,
        )


def _try_replicator_write(
    articulation: Any,
    *,
    friction: float,
    mass_kg: float,
) -> None:
    """Best-effort Omniverse Replicator *import* check; this slice does not write.

    A successful import only logs ``isaac_lab_replicator_present``. Chassis
    mass/friction writes stay on the duck-typed
    ``apply_chassis_domain_params`` / PhysX mass path above.
    """
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
