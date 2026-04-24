"""Regression test: existing configs still load after arm schema extension.

Verifies that adding optional arm config fields to Settings does not
break loading of existing YAML configuration files.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from mousedroid.config.schema import PlatformType, Settings

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = REPO_ROOT / "config"


class TestBackwardsCompatibility:
    """Existing config files must load without errors."""

    def test_default_yaml_loads(self) -> None:
        """default.yaml loads with new schema — no arm fields required."""
        cfg_path = CONFIG_DIR / "default.yaml"
        if not cfg_path.exists():
            return  # Skip if running from different working dir

        with open(cfg_path) as f:
            raw = yaml.safe_load(f)

        cfg = Settings(**raw)
        assert cfg.platform == PlatformType.MOUSE_DROID
        # Arm fields should all be None
        assert cfg.arm is None
        assert cfg.arm_sim is None
        assert cfg.arm_perception is None
        assert cfg.arm_planning is None
        assert cfg.arm_training is None
        assert cfg.arm_curriculum is None
        assert cfg.arm_task is None

    def test_robot_arm_yaml_loads(self) -> None:
        """robot_arm_default.yaml loads correctly."""
        cfg_path = CONFIG_DIR / "robot_arm_default.yaml"
        if not cfg_path.exists():
            return

        with open(cfg_path) as f:
            raw = yaml.safe_load(f)

        cfg = Settings(**raw)
        assert cfg.platform == PlatformType.ROBOT_ARM
        assert cfg.arm is not None
        assert cfg.arm.dof == 6
        assert cfg.arm_task is not None
        assert cfg.arm_task.num_disks == 3

    def test_mock_hardware_yaml_loads(self) -> None:
        """mock_hardware.yaml loads unchanged."""
        cfg_path = CONFIG_DIR / "mock_hardware.yaml"
        if not cfg_path.exists():
            return

        with open(cfg_path) as f:
            raw = yaml.safe_load(f)

        cfg = Settings(**raw)
        assert cfg.mock_hardware is True

    def test_settings_minimal_construction(self) -> None:
        """Settings can be created with just mock_hardware=True."""
        cfg = Settings(mock_hardware=True)
        assert cfg.platform == PlatformType.MOUSE_DROID
        assert cfg.arm is None

    def test_settings_with_arm_and_mouse_droid_fields(self) -> None:
        """Mouse droid fields still work alongside arm None defaults."""
        cfg = Settings(
            mock_hardware=True,
            platform=PlatformType.MOUSE_DROID,
            debug=True,
        )
        assert cfg.debug is True
        assert cfg.arm is None
        assert cfg.esp32.protocol == "serial"

    def test_legacy_robot_arm_group_sections_are_migrated(self) -> None:
        """Legacy robot_arm nested sections map to canonical top-level arm sections."""
        cfg = Settings.model_validate(
            {
                "mock_hardware": True,
                "platform": "robot_arm",
                "robot_arm": {
                    "hardware": {
                        "dof": 7,
                        "home_position": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    },
                    "sim": {"timestep_s": 0.004},
                    "perception": {"depth_camera_type": "mock"},
                    "planning": {"planning_timeout_s": 7.5},
                    "training": {"algorithm": "ppo"},
                    "curriculum": {"enabled": False},
                    "task": {"task_type": "pick_place"},
                },
            }
        )
        assert cfg.arm is not None
        assert cfg.arm.dof == 7
        assert cfg.arm_sim is not None
        assert cfg.arm_sim.timestep_s == 0.004
        assert cfg.arm_perception is not None
        assert cfg.arm_perception.depth_camera_type == "mock"
        assert cfg.arm_planning is not None
        assert cfg.arm_planning.planning_timeout_s == 7.5
        assert cfg.arm_training is not None
        assert cfg.arm_training.algorithm == "ppo"
        assert cfg.arm_curriculum is not None
        assert cfg.arm_curriculum.enabled is False
        assert cfg.arm_task is not None
        assert cfg.arm_task.task_type == "pick_place"

    def test_legacy_top_level_arm_aliases_are_migrated(self) -> None:
        """Legacy arm_* top-level blocks map to canonical arm section names."""
        cfg = Settings.model_validate(
            {
                "mock_hardware": True,
                "platform": "robot_arm",
                "arm_hardware": {
                    "dof": 5,
                    "home_position": [0.0, 0.0, 0.0, 0.0, 0.0],
                },
                "arm_simulation": {"timestep_s": 0.01},
                "arm_vision": {"depth_camera_type": "mock"},
                "arm_symbolic_planning": {"planner_backend": "pyperplan"},
                "arm_rl_training": {"algorithm": "sac"},
                "arm_curriculum_learning": {"enabled": False},
                "arm_tasks": {"task_type": "tower_of_hanoi", "num_disks": 4},
            }
        )
        assert cfg.arm is not None
        assert cfg.arm.dof == 5
        assert cfg.arm_sim is not None
        assert cfg.arm_sim.timestep_s == 0.01
        assert cfg.arm_perception is not None
        assert cfg.arm_perception.depth_camera_type == "mock"
        assert cfg.arm_planning is not None
        assert cfg.arm_planning.planner_backend == "pyperplan"
        assert cfg.arm_training is not None
        assert cfg.arm_training.algorithm == "sac"
        assert cfg.arm_curriculum is not None
        assert cfg.arm_curriculum.enabled is False
        assert cfg.arm_task is not None
        assert cfg.arm_task.num_disks == 4

    def test_canonical_arm_sections_win_over_legacy_aliases(self) -> None:
        """Canonical arm sections take precedence when both key variants are present."""
        cfg = Settings.model_validate(
            {
                "mock_hardware": True,
                "platform": "robot_arm",
                "arm": {"dof": 6},
                "arm_hardware": {
                    "dof": 7,
                    "home_position": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                },
                "arm_sim": {"timestep_s": 0.002},
                "robot_arm": {
                    "sim": {"timestep_s": 0.05},
                },
            }
        )
        assert cfg.arm is not None
        assert cfg.arm.dof == 6
        assert cfg.arm_sim is not None
        assert cfg.arm_sim.timestep_s == 0.002
