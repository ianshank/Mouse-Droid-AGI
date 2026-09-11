"""Automated Quality Assurance (AQA) — F-043 Isaac nested schema hygiene."""

from __future__ import annotations

import inspect

from pydantic.fields import FieldInfo

from mousedroid.config.schema import (
    RoverIsaacSimConfig,
    RoverObservationConfig,
    RoverSimConfig,
)
from mousedroid.factory.world_model import build_rover_env


def _desc(model: type, name: str) -> FieldInfo:
    info = model.model_fields[name]
    assert isinstance(info, FieldInfo)
    return info


def test_isaac_device_fields_have_descriptions() -> None:
    for name in (
        "device_headless",
        "device_gui",
        "prim_path",
        "contact_prim_glob",
        "usd_path",
        "actuator_stiffness",
        "contact_history_length",
        "track_air_time",
    ):
        info = _desc(RoverIsaacSimConfig, name)
        assert info.description
        assert len(info.description) > 20, info.description


def test_parent_battery_and_observation_lidar_range_described() -> None:
    batt = _desc(RoverSimConfig, "battery_voltage_const_v")
    assert batt.description
    assert len(batt.description) > 20
    rng = _desc(RoverObservationConfig, "lidar_max_range_m")
    assert rng.description
    assert len(rng.description) > 20
    isaac = _desc(RoverSimConfig, "isaac")
    assert isaac.description
    assert len(isaac.description) > 20


def test_isaac_field_defaults_match_pre_f043_literals() -> None:
    """Pinned off FieldInfo, not a live instance — see regression-pair skill."""
    assert _desc(RoverIsaacSimConfig, "device_headless").default == "cuda:0"
    assert _desc(RoverIsaacSimConfig, "device_gui").default == "cpu"
    assert _desc(RoverIsaacSimConfig, "prim_path").default == "/World/envs/env_.*/Robot"
    assert _desc(RoverIsaacSimConfig, "contact_prim_glob").default == "/World/envs/env_.*/Robot/.*"
    assert _desc(RoverIsaacSimConfig, "usd_path").default is None
    assert _desc(RoverIsaacSimConfig, "actuator_stiffness").default == 0.0
    assert _desc(RoverIsaacSimConfig, "contact_history_length").default == 0
    assert _desc(RoverIsaacSimConfig, "track_air_time").default is False


def test_shared_battery_default_is_twelve() -> None:
    assert _desc(RoverSimConfig, "battery_voltage_const_v").default == 12.0
    assert _desc(RoverObservationConfig, "lidar_max_range_m").default == 4.0


def test_build_rover_env_docstring_does_not_claim_mujoco_unimplemented() -> None:
    doc = inspect.getdoc(build_rover_env)
    assert doc is not None
    assert "NotImplementedError" not in doc
    assert "mujoco" in doc.lower()
    assert "isaac_lab" in doc or "Isaac Lab" in doc
