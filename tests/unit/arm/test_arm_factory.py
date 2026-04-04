"""Tests for arm factory functions."""

from __future__ import annotations

import pytest

from mousedroid.arm.hardware.mock_arm_driver import MockArmDriver
from mousedroid.arm.protocols import (
    ArmControllerProtocol,
    ArmDriverProtocol,
    ArmPerceptionProtocol,
)
from mousedroid.config.schema import (
    ArmConfig,
    ArmPlanningConfig,
    ArmTaskConfig,
    ArmTrainingConfig,
    PlatformType,
    Settings,
)
from mousedroid.factory import (
    build_arm_controller,
    build_arm_driver,
    build_arm_environment,
    build_arm_perception,
    build_arm_planner,
)


def _arm_settings(**kwargs: object) -> Settings:
    """Create Settings with arm config populated."""
    defaults = {
        "mock_hardware": True,
        "platform": PlatformType.ROBOT_ARM,
        "arm": ArmConfig(),
        "arm_task": ArmTaskConfig(),
        "arm_training": ArmTrainingConfig(),
        "arm_planning": ArmPlanningConfig(),
    }
    defaults.update(kwargs)
    return Settings(**defaults)  # type: ignore[arg-type]


class TestBuildArmDriver:
    """Test build_arm_driver factory."""

    def test_mock_returns_mock_driver(self) -> None:
        cfg = _arm_settings(mock_hardware=True)
        driver = build_arm_driver(cfg)
        assert isinstance(driver, MockArmDriver)
        assert isinstance(driver, ArmDriverProtocol)

    def test_missing_arm_config_raises(self) -> None:
        cfg = Settings(mock_hardware=True)
        with pytest.raises(ValueError, match="arm config required"):
            build_arm_driver(cfg)


class TestBuildArmPlanner:
    """Test build_arm_planner factory."""

    def test_builds_symbolic_planner(self) -> None:
        cfg = _arm_settings()
        planner = build_arm_planner(cfg)
        assert planner is not None

    def test_missing_planning_config_raises(self) -> None:
        cfg = Settings(mock_hardware=True, arm_task=ArmTaskConfig())
        with pytest.raises(ValueError, match="arm_planning and arm_task"):
            build_arm_planner(cfg)


class TestBuildArmEnvironment:
    """Test build_arm_environment factory."""

    def test_builds_hanoi_env(self) -> None:
        cfg = _arm_settings()
        env = build_arm_environment(cfg)
        assert env is not None

    def test_builds_laundry_env(self) -> None:
        cfg = _arm_settings(
            arm_task=ArmTaskConfig(task_type="laundry_sorting"),
        )
        env = build_arm_environment(cfg)
        assert env is not None

    def test_missing_task_config_raises(self) -> None:
        cfg = Settings(mock_hardware=True, arm_training=ArmTrainingConfig())
        with pytest.raises(ValueError, match="arm_task and arm_training"):
            build_arm_environment(cfg)


class TestBuildArmController:
    """Test build_arm_controller factory."""

    def test_builds_controller(self) -> None:
        cfg = _arm_settings()
        controller = build_arm_controller(cfg)
        assert isinstance(controller, ArmControllerProtocol)

    def test_missing_config_raises(self) -> None:
        cfg = Settings(mock_hardware=True)
        with pytest.raises(ValueError, match="arm and arm_training"):
            build_arm_controller(cfg)


class TestBuildArmPerception:
    """Test build_arm_perception factory."""

    def test_builds_perception(self) -> None:
        from mousedroid.config.schema import ArmPerceptionConfig

        cfg = _arm_settings(arm_perception=ArmPerceptionConfig())
        perception = build_arm_perception(cfg)
        assert isinstance(perception, ArmPerceptionProtocol)

    def test_missing_config_raises(self) -> None:
        cfg = Settings(mock_hardware=True)
        with pytest.raises(ValueError, match="arm_perception and arm_task"):
            build_arm_perception(cfg)
