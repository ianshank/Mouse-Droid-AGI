"""Tests for arm action primitives."""

from __future__ import annotations

import numpy as np
import pytest

from mousedroid.arm.control.primitives import ActionPrimitives
from mousedroid.arm.hardware.mock_arm_driver import MockArmDriver
from mousedroid.config.schema import ArmConfig


def _make_cfg() -> ArmConfig:
    return ArmConfig(dof=6, home_position=[0.0] * 6)


def _make_primitives() -> tuple[ActionPrimitives, MockArmDriver]:
    cfg = _make_cfg()
    driver = MockArmDriver(cfg)
    return ActionPrimitives(cfg, driver), driver


class TestTransit:
    """Tests for transit primitive."""

    @pytest.mark.asyncio
    async def test_transit_success(self) -> None:
        prims, _driver = _make_primitives()
        target = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float64)
        result = await prims.transit(target)
        assert result is True

    @pytest.mark.asyncio
    async def test_transit_updates_joints(self) -> None:
        prims, driver = _make_primitives()
        target = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float64)
        await prims.transit(target)
        joints = await driver.get_joint_states()
        np.testing.assert_allclose(joints, target)

    @pytest.mark.asyncio
    async def test_transit_failure_returns_false(self) -> None:
        prims, driver = _make_primitives()
        # Wrong DOF to trigger ValueError
        target = np.array([0.1, 0.2], dtype=np.float64)
        result = await prims.transit(target)
        assert result is False


class TestGrasp:
    """Tests for grasp primitive."""

    @pytest.mark.asyncio
    async def test_grasp_returns_force(self) -> None:
        prims, _driver = _make_primitives()
        pose = np.zeros(6, dtype=np.float64)
        force = await prims.grasp(pose)
        assert force > 0.0

    @pytest.mark.asyncio
    async def test_grasp_failure_returns_zero(self) -> None:
        prims, _driver = _make_primitives()
        bad_pose = np.array([0.1], dtype=np.float64)
        force = await prims.grasp(bad_pose)
        assert force == 0.0


class TestPlace:
    """Tests for place primitive."""

    @pytest.mark.asyncio
    async def test_place_success(self) -> None:
        prims, _driver = _make_primitives()
        pose = np.zeros(6, dtype=np.float64)
        result = await prims.place(pose)
        assert result is True

    @pytest.mark.asyncio
    async def test_place_failure_returns_false(self) -> None:
        prims, _driver = _make_primitives()
        bad_pose = np.array([0.1], dtype=np.float64)
        result = await prims.place(bad_pose)
        assert result is False


class TestHome:
    """Tests for home primitive."""

    @pytest.mark.asyncio
    async def test_home_success(self) -> None:
        prims, _driver = _make_primitives()
        result = await prims.home()
        assert result is True
