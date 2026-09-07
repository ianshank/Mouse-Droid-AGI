"""Automated Quality Assurance (AQA) — F-040 IMU fusion slot hygiene."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from pydantic.fields import FieldInfo

from mousedroid.config.schema import ModelConfig
from mousedroid.constants import (
    DEFAULT_MOTOR_STATE_DIM,
    N_SENSOR_MODALITIES_WITH_IMU,
    N_SENSOR_MODALITIES_WITH_LIDAR,
    SENSOR_SLOT_MAP,
)
from mousedroid.world_model.encoder import MultimodalEncoder
from mousedroid.world_model.observation_packer import PackedObservation


def test_imu_dim_has_description() -> None:
    info: FieldInfo = ModelConfig.model_fields["imu_dim"]
    assert info.description
    assert len(info.description) > 20, info.description


def test_imu_proj_dim_has_description() -> None:
    info: FieldInfo = ModelConfig.model_fields["imu_proj_dim"]
    assert info.description
    assert len(info.description) > 20, info.description


def test_imu_dim_default_is_zero() -> None:
    info: FieldInfo = ModelConfig.model_fields["imu_dim"]
    assert info.default == 0


def test_imu_proj_dim_default_is_32() -> None:
    info: FieldInfo = ModelConfig.model_fields["imu_proj_dim"]
    assert info.default == 32


def test_imu_dim_without_proj_raises() -> None:
    with pytest.raises(ValidationError, match="imu"):
        ModelConfig(imu_dim=3, imu_proj_dim=0)


def test_slot_map_imu_is_five_and_packed_width_six() -> None:
    assert SENSOR_SLOT_MAP["imu"] == 5
    assert N_SENSOR_MODALITIES_WITH_IMU == 6
    assert N_SENSOR_MODALITIES_WITH_LIDAR == 5
    assert DEFAULT_MOTOR_STATE_DIM == 4


def test_default_encoder_has_no_imu_branch() -> None:
    enc = MultimodalEncoder(ModelConfig())
    assert enc.imu_enabled is False
    assert not hasattr(enc, "imu_proj")


def test_packed_observation_imu_defaults_none() -> None:
    """New field is last and defaulted so positional lidar construction still works."""
    imu_field = PackedObservation.__dataclass_fields__["imu"]
    assert imu_field.default is None
