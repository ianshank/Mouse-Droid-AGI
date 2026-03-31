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
