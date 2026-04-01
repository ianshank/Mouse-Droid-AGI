"""Tests for ArmController wrapper."""

from __future__ import annotations

import numpy as np
import pytest

from mousedroid.arm.control.controller import ArmController
from mousedroid.arm.control.primitives import ActionPrimitives
from mousedroid.arm.control.sac_agent import SACAgent
from mousedroid.arm.hardware.mock_arm_driver import MockArmDriver
from mousedroid.config.schema import ArmConfig, ArmTrainingConfig


def _make_controller() -> ArmController:
    arm_cfg = ArmConfig(dof=6, home_position=[0.0] * 6)
    driver = MockArmDriver(arm_cfg)
    agent = SACAgent(ArmTrainingConfig())
    primitives = ActionPrimitives(arm_cfg, driver)
    return ArmController(agent, primitives)


class TestExecuteAction:
    """Tests for execute_action."""

    @pytest.mark.asyncio
    async def test_success_returns_true(self) -> None:
        ctrl = _make_controller()
        action = np.zeros(6, dtype=np.float64)
        result = await ctrl.execute_action(action)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_wrong_dof_returns_not_success(self) -> None:
        ctrl = _make_controller()
        bad_action = np.zeros(2, dtype=np.float64)  # wrong DOF
        result = await ctrl.execute_action(bad_action)
        # primitives.transit catches the ValueError and returns False
        assert result["success"] is False


class TestExecutePrimitive:
    """Tests for execute_primitive."""

    @pytest.mark.asyncio
    async def test_grasp(self) -> None:
        ctrl = _make_controller()
        target = np.zeros(6, dtype=np.float64)
        result = await ctrl.execute_primitive("grasp", target)
        assert result is True

    @pytest.mark.asyncio
    async def test_place(self) -> None:
        ctrl = _make_controller()
        target = np.zeros(6, dtype=np.float64)
        result = await ctrl.execute_primitive("place", target)
        assert result is True

    @pytest.mark.asyncio
    async def test_transit(self) -> None:
        ctrl = _make_controller()
        target = np.zeros(6, dtype=np.float64)
        result = await ctrl.execute_primitive("transit", target)
        assert result is True

    @pytest.mark.asyncio
    async def test_home(self) -> None:
        ctrl = _make_controller()
        target = np.zeros(6, dtype=np.float64)
        result = await ctrl.execute_primitive("home", target)
        assert result is True

    @pytest.mark.asyncio
    async def test_unknown_primitive_returns_false(self) -> None:
        ctrl = _make_controller()
        target = np.zeros(6, dtype=np.float64)
        result = await ctrl.execute_primitive("nonexistent", target)
        assert result is False
