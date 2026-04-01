"""Tests for robot arm configuration models."""

from __future__ import annotations

import pytest

from mousedroid.config.schema import (
    ArmConfig,
    ArmCurriculumConfig,
    ArmPerceptionConfig,
    ArmPlanningConfig,
    ArmSimConfig,
    ArmTaskConfig,
    ArmTrainingConfig,
    PlatformType,
    Settings,
)


class TestPlatformType:
    """Test PlatformType enum extension."""

    def test_robot_arm_value(self) -> None:
        assert PlatformType.ROBOT_ARM == "robot_arm"

    def test_mouse_droid_still_exists(self) -> None:
        assert PlatformType.MOUSE_DROID == "mouse_droid"


class TestArmConfig:
    """Test ArmConfig Pydantic model."""

    def test_defaults(self) -> None:
        cfg = ArmConfig()
        assert cfg.dof == 6
        assert cfg.gripper_type == "parallel"
        assert len(cfg.home_position) == 6

    def test_custom_dof(self) -> None:
        cfg = ArmConfig(dof=4, home_position=[0.0, 0.0, 0.0, 0.0])
        assert cfg.dof == 4
        assert len(cfg.home_position) == 4

    def test_home_dof_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="home_position length"):
            ArmConfig(dof=6, home_position=[0.0, 0.0, 0.0])


class TestArmSimConfig:
    """Test ArmSimConfig Pydantic model."""

    def test_defaults(self) -> None:
        cfg = ArmSimConfig()
        assert cfg.timestep_s == 0.002
        assert cfg.domain_randomization is True
        assert cfg.mass_range_pct == 20.0

    def test_custom_randomization(self) -> None:
        cfg = ArmSimConfig(mass_range_pct=50.0, friction_range=0.5)
        assert cfg.mass_range_pct == 50.0
        assert cfg.friction_range == 0.5


class TestArmPerceptionConfig:
    """Test ArmPerceptionConfig Pydantic model."""

    def test_defaults(self) -> None:
        cfg = ArmPerceptionConfig()
        assert cfg.depth_camera_type == "realsense_d435i"
        assert cfg.pose_estimator == "pnp"
        assert cfg.yolo_confidence_threshold == 0.5

    def test_mock_camera(self) -> None:
        cfg = ArmPerceptionConfig(depth_camera_type="mock")
        assert cfg.depth_camera_type == "mock"


class TestArmPlanningConfig:
    """Test ArmPlanningConfig Pydantic model."""

    def test_defaults(self) -> None:
        cfg = ArmPlanningConfig()
        assert cfg.planner_backend == "pyperplan"
        assert cfg.llm_replanner_enabled is False
        assert cfg.max_replan_attempts == 3


class TestArmTrainingConfig:
    """Test ArmTrainingConfig Pydantic model."""

    def test_defaults(self) -> None:
        cfg = ArmTrainingConfig()
        assert cfg.algorithm == "sac_her"
        assert cfg.learning_rate == 3e-4
        assert cfg.buffer_size == 1_000_000
        assert cfg.her_goal_selection == "future"
        assert cfg.reward_grasp == 0.1
        assert cfg.penalty_collision == -0.5
        assert cfg.seed == 42

    def test_custom_algorithm(self) -> None:
        cfg = ArmTrainingConfig(algorithm="ppo")
        assert cfg.algorithm == "ppo"


class TestArmCurriculumConfig:
    """Test ArmCurriculumConfig Pydantic model."""

    def test_defaults(self) -> None:
        cfg = ArmCurriculumConfig()
        assert cfg.enabled is True
        assert cfg.stages == [1, 2, 3, 5]
        assert cfg.promotion_threshold == 0.8
        assert cfg.warm_start is True


class TestArmTaskConfig:
    """Test ArmTaskConfig Pydantic model."""

    def test_defaults(self) -> None:
        cfg = ArmTaskConfig()
        assert cfg.task_type == "tower_of_hanoi"
        assert cfg.num_disks == 3
        assert cfg.num_pegs == 3
        assert len(cfg.peg_positions) == 3

    def test_peg_count_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="peg_positions length"):
            ArmTaskConfig(num_pegs=4, peg_positions=[[0, 0, 0], [1, 0, 0]])

    def test_basket_count_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="basket_positions length"):
            ArmTaskConfig(num_baskets=2, basket_positions=[[0, 0, 0]])

    def test_laundry_task(self) -> None:
        cfg = ArmTaskConfig(task_type="laundry_sorting")
        assert cfg.task_type == "laundry_sorting"


class TestSettingsArmFields:
    """Test that arm fields on Settings are optional and default to None."""

    def test_arm_fields_default_none(self) -> None:
        cfg = Settings(mock_hardware=True)
        assert cfg.arm is None
        assert cfg.arm_sim is None
        assert cfg.arm_perception is None
        assert cfg.arm_planning is None
        assert cfg.arm_training is None
        assert cfg.arm_curriculum is None
        assert cfg.arm_task is None

    def test_arm_fields_populated(self) -> None:
        cfg = Settings(
            mock_hardware=True,
            platform=PlatformType.ROBOT_ARM,
            arm=ArmConfig(),
            arm_task=ArmTaskConfig(),
            arm_training=ArmTrainingConfig(),
        )
        assert cfg.arm is not None
        assert cfg.arm.dof == 6
        assert cfg.arm_task is not None
        assert cfg.arm_task.num_disks == 3
