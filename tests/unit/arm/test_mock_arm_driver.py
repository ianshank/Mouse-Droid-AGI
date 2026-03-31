"""Tests for mock arm driver."""

from __future__ import annotations

import numpy as np
import pytest

from mousedroid.arm.hardware.mock_arm_driver import MockArmDriver
from mousedroid.config.schema import ArmConfig


def _make_driver(dof: int = 6) -> MockArmDriver:
    """Create mock driver with test defaults."""
    cfg = ArmConfig(dof=dof, home_position=[0.0] * dof)
    return MockArmDriver(cfg)


class TestMockArmDriver:
    """Test MockArmDriver functionality."""

    @pytest.mark.asyncio
    async def test_initial_joint_states_at_home(self) -> None:
        driver = _make_driver()
        joints = await driver.get_joint_states()
        np.testing.assert_array_equal(joints, np.zeros(6))

    @pytest.mark.asyncio
    async def test_send_joint_command(self) -> None:
        driver = _make_driver()
        target = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        await driver.send_joint_command(target)
        joints = await driver.get_joint_states()
        np.testing.assert_allclose(joints, target)

    @pytest.mark.asyncio
    async def test_send_wrong_dof_raises(self) -> None:
        driver = _make_driver(dof=6)
        with pytest.raises(ValueError, match="Expected 6"):
            await driver.send_joint_command(np.zeros(4))

    @pytest.mark.asyncio
    async def test_joint_command_clamps_to_pi(self) -> None:
        driver = _make_driver()
        target = np.array([10.0, -10.0, 0.0, 0.0, 0.0, 0.0])
        await driver.send_joint_command(target)
        joints = await driver.get_joint_states()
        assert np.all(joints <= np.pi)
        assert np.all(joints >= -np.pi)

    @pytest.mark.asyncio
    async def test_close_gripper_returns_force(self) -> None:
        driver = _make_driver()
        force = await driver.close_gripper()
        assert force == 1.0  # First close

    @pytest.mark.asyncio
    async def test_close_gripper_twice_returns_zero(self) -> None:
        driver = _make_driver()
        await driver.close_gripper()
        force = await driver.close_gripper()
        assert force == 0.0  # Already closed

    @pytest.mark.asyncio
    async def test_open_then_close_returns_force(self) -> None:
        driver = _make_driver()
        await driver.close_gripper()
        await driver.open_gripper()
        force = await driver.close_gripper()
        assert force == 1.0

    @pytest.mark.asyncio
    async def test_home_resets_to_home_position(self) -> None:
        driver = _make_driver()
        await driver.send_joint_command(np.ones(6))
        await driver.home()
        joints = await driver.get_joint_states()
        np.testing.assert_array_equal(joints, np.zeros(6))

    @pytest.mark.asyncio
    async def test_start_stop(self) -> None:
        driver = _make_driver()
        await driver.start()
        await driver.stop()

    def test_dof_property(self) -> None:
        driver = _make_driver(dof=4)
        assert driver.dof == 4

    @pytest.mark.asyncio
    async def test_emergency_stop(self) -> None:
        driver = _make_driver()
        await driver.emergency_stop()  # Should not raise
