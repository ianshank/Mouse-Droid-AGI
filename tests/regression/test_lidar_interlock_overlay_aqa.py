"""The LiDAR interlock is armed on exactly one shipped stack (peer review D-3/D-4).

``SafetyConfig.lidar_unavailable_policy`` and ``SafetyProjectorConfig`` have
shipped since S-2 with nobody opting in, so the interlock they implement was
dead configuration. This batch arms both in ``config/jetson_lidar_only.yaml``
and **only** there.

The asymmetry is not a tidiness choice, and these pins exist because an
earlier draft of the change got it backwards. ``_evaluate_lidar_clearance``
takes its "LiDAR available" branch only when ``len(lidar_features) > 0``, and
LiDAR reaches the observation only when ``model.lidar_dim > 0``.
``config/default.yaml`` sets ``lidar_dim: 0``; ``jetson_production.yaml``
overrides no ``model:`` block. So on the production stack the features are
empty on every tick, and a non-``ignore`` policy there would not arm an
interlock -- it would brake or emergency-stop the rover permanently.

:func:`test_production_alone_stays_inert` is the pin that would have caught
that draft, which makes it the most valuable test in this file.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mousedroid.config.loader import load_settings
from mousedroid.config.schema import Settings

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _REPO_ROOT / "config"

_PRODUCTION = _CONFIG_DIR / "jetson_production.yaml"
_LIDAR_ONLY = _CONFIG_DIR / "jetson_lidar_only.yaml"

#: ``baselines.yaml`` is merged under its own key by the loader, not applied
#: as an overlay; ``default.yaml`` is the base every stack already starts from.
_NOT_AN_OVERLAY = {"baselines.yaml", "default.yaml"}

_OTHER_OVERLAYS = sorted(
    p.name
    for p in _CONFIG_DIR.glob("*.yaml")
    if p.name not in _NOT_AN_OVERLAY and p.name != _LIDAR_ONLY.name
)


def _stacked() -> Settings:
    """The load the lidar-only rig actually performs, per that file's header."""
    return load_settings(_PRODUCTION, _LIDAR_ONLY, config_dir=_CONFIG_DIR)


# -- D-3/D-4: armed on the stack that has a LiDAR ---------------------------


def test_stacked_load_arms_the_emergency_policy() -> None:
    """Testing the overlay alone would prove nothing -- it is never loaded alone."""
    assert _stacked().safety.lidar_unavailable_policy == "emergency"


def test_stacked_load_carries_a_non_zero_grace() -> None:
    """``emergency`` with a 0.0 grace e-stops on the first tick without features.

    The monitor's own comment is explicit that a zero grace "really is no
    grace", so pairing the two would be the footgun, not the interlock.
    """
    assert _stacked().safety.lidar_unavailable_grace_s > 0.0


def test_stacked_load_enables_the_projector() -> None:
    assert _stacked().safety.projector.enabled is True


def test_the_stack_is_the_only_one_where_lidar_reaches_the_observation() -> None:
    """The fact the whole D-3 analysis turns on, pinned rather than narrated."""
    assert _stacked().model.lidar_dim > 0
    assert load_settings(_PRODUCTION, config_dir=_CONFIG_DIR).model.lidar_dim == 0


# -- The pin that would have caught the first draft -------------------------


def test_production_alone_stays_inert() -> None:
    """Arming production would brake or e-stop the rig permanently, every tick.

    ``model.lidar_dim`` is 0 there, so ``_evaluate_lidar_clearance`` falls
    through to the policy branch on every tick. This is the assertion that
    fails if someone "closes the gap" in ``jetson_production.yaml``.
    """
    cfg = load_settings(_PRODUCTION, config_dir=_CONFIG_DIR)
    assert cfg.safety.lidar_unavailable_policy == "ignore"
    assert cfg.safety.projector.enabled is False


@pytest.mark.parametrize("overlay", _OTHER_OVERLAYS)
def test_every_other_overlay_is_untouched(overlay: str) -> None:
    """Fleet-scale: the change is targeted, not a default flip in disguise."""
    cfg = load_settings(_CONFIG_DIR / overlay, config_dir=_CONFIG_DIR)
    assert cfg.safety.lidar_unavailable_policy == "ignore", overlay
    assert cfg.safety.projector.enabled is False, overlay


def test_the_other_overlay_set_is_non_empty() -> None:
    """Guard against a glob that silently matches nothing."""
    assert _OTHER_OVERLAYS


# -- Derived budgets, so the numbers are relations and not constants --------


def test_blind_travel_during_the_grace_stays_inside_the_brake_distance() -> None:
    """During the grace the monitor reports clear, so the rover drives blind.

    Pinning ``grace * max_velocity <= lidar_brake_distance_m`` states the
    real budget: the grace must not be able to carry the rover through the
    whole zone in which the projector would have been braking. It also
    deliberately does NOT assert the weaker, more obvious
    ``<= min_forward_clearance_m`` -- that relation is already violated
    (0.25 m of travel against a 0.20 m threshold) and pretending otherwise
    would be the dishonest version of this test.
    """
    safety = _stacked().safety
    blind_travel_m = safety.lidar_unavailable_grace_s * safety.max_velocity_mps
    assert blind_travel_m <= safety.projector.lidar_brake_distance_m, (
        f"{blind_travel_m} m of blind travel exceeds the "
        f"{safety.projector.lidar_brake_distance_m} m brake distance"
    )


def test_the_grace_does_not_try_to_absorb_a_whole_failed_acquisition() -> None:
    """Sizing the grace to ride out a full acquisition is the tempting mistake.

    It would need ``grace > scan_acquisition_timeout_s`` (1.0 s), i.e. 0.5 m
    of blind travel at ``max_velocity_mps`` -- 2.5x the forward-clearance
    threshold. After D-24 it is also unnecessary: a *slow* scan comes back
    partial-but-real with ``sensor_responding`` true and never reaches the
    policy branch at all.
    """
    cfg = _stacked()
    assert cfg.lidar is not None
    assert cfg.safety.lidar_unavailable_grace_s < cfg.lidar.scan_acquisition_timeout_s


# -- The three-way footgun --------------------------------------------------


@pytest.mark.parametrize(
    "overlay",
    sorted(p.name for p in _CONFIG_DIR.glob("*.yaml") if p.name not in _NOT_AN_OVERLAY),
)
def test_no_shipped_config_pairs_emergency_with_a_latch_and_no_grace(overlay: str) -> None:
    """``emergency`` + ``emergency_latch`` + grace 0.0 = one dropped scan, then
    a rover that will not move until an operator physically clears it.

    Any one of the three is fine. The combination is the failure mode, so it
    is pinned as a combination rather than as three separate defaults.
    """
    cfg = load_settings(_CONFIG_DIR / overlay, config_dir=_CONFIG_DIR)
    safety = cfg.safety
    armed = safety.lidar_unavailable_policy == "emergency"
    latching = safety.emergency_latch.enabled
    assert not (armed and latching and safety.lidar_unavailable_grace_s == 0.0), overlay
