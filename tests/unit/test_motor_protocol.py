"""Tests for MotorControlProtocol, GroundMotorAdapter, and DroneMotorAdapter."""

from __future__ import annotations

from unittest.mock import AsyncMock

import numpy as np
import pytest

from mousedroid.comms.drone_adapter import DroneMotorAdapter
from mousedroid.comms.ground_adapter import GroundMotorAdapter
from mousedroid.comms.motor_protocol import MotorControlProtocol
from mousedroid.comms.protocol import EncoderReading

# -- Protocol conformance --------------------------------------------------


def test_ground_adapter_satisfies_protocol():
    """GroundMotorAdapter is recognised as MotorControlProtocol at runtime."""
    esp32 = AsyncMock()
    adapter = GroundMotorAdapter(esp32)
    assert isinstance(adapter, MotorControlProtocol)


def test_drone_adapter_satisfies_protocol():
    """DroneMotorAdapter is recognised as MotorControlProtocol at runtime."""
    fc = AsyncMock()
    fc.armed = False
    adapter = DroneMotorAdapter(fc)
    assert isinstance(adapter, MotorControlProtocol)


# -- GroundMotorAdapter ----------------------------------------------------


async def test_ground_adapter_connect():
    esp32 = AsyncMock()
    adapter = GroundMotorAdapter(esp32)
    await adapter.connect()
    esp32.connect.assert_awaited_once()


async def test_ground_adapter_disconnect():
    esp32 = AsyncMock()
    adapter = GroundMotorAdapter(esp32)
    await adapter.disconnect()
    esp32.disconnect.assert_awaited_once()


async def test_ground_adapter_emergency_stop():
    esp32 = AsyncMock()
    adapter = GroundMotorAdapter(esp32)
    await adapter.emergency_stop()
    esp32.emergency_stop.assert_awaited_once()


async def test_ground_adapter_send_command():
    """send_command extracts [vx, vy, omega] and calls esp32.send_velocity."""
    esp32 = AsyncMock()
    adapter = GroundMotorAdapter(esp32)
    cmd = np.array([0.5, 0.3, 0.2], dtype=np.float32)
    await adapter.send_command(cmd)
    esp32.send_velocity.assert_awaited_once()
    args = esp32.send_velocity.await_args.args
    assert args[0] == pytest.approx(0.5)
    assert args[1] == pytest.approx(0.3)
    assert args[2] == pytest.approx(0.2)


async def test_ground_adapter_send_command_partial():
    """send_command handles partial arrays gracefully."""
    esp32 = AsyncMock()
    adapter = GroundMotorAdapter(esp32)
    cmd = np.array([0.5], dtype=np.float32)
    await adapter.send_command(cmd)
    esp32.send_velocity.assert_awaited_once_with(0.5, 0.0, 0.0)


async def test_ground_adapter_read_state():
    """read_state returns [left_vel, right_vel, heading, battery_v]."""
    esp32 = AsyncMock()
    esp32.read_encoders = AsyncMock(
        return_value=EncoderReading(left_velocity_mps=1.0, right_velocity_mps=0.5, heading_rad=0.1)
    )
    esp32.get_battery_voltage = AsyncMock(return_value=12.0)
    adapter = GroundMotorAdapter(esp32)

    state = await adapter.read_state()
    assert state.shape == (4,)
    assert state[0] == pytest.approx(1.0)
    assert state[1] == pytest.approx(0.5)
    assert state[2] == pytest.approx(0.1)
    assert state[3] == pytest.approx(12.0)


async def test_ground_adapter_get_battery():
    esp32 = AsyncMock()
    esp32.get_battery_voltage = AsyncMock(return_value=11.5)
    adapter = GroundMotorAdapter(esp32)
    voltage = await adapter.get_battery_voltage()
    assert voltage == pytest.approx(11.5)


def test_ground_adapter_platform_type():
    esp32 = AsyncMock()
    adapter = GroundMotorAdapter(esp32)
    assert adapter.platform_type == "mouse_droid"


# -- DroneMotorAdapter -----------------------------------------------------


async def test_drone_adapter_connect():
    fc = AsyncMock()
    fc.armed = False
    adapter = DroneMotorAdapter(fc)
    await adapter.connect()
    fc.connect.assert_awaited_once()


async def test_drone_adapter_disconnect():
    fc = AsyncMock()
    fc.armed = False
    adapter = DroneMotorAdapter(fc)
    await adapter.disconnect()
    fc.disconnect.assert_awaited_once()


async def test_drone_adapter_emergency_stop():
    fc = AsyncMock()
    fc.armed = False
    adapter = DroneMotorAdapter(fc)
    await adapter.emergency_stop()
    fc.emergency_stop.assert_awaited_once()


async def test_drone_adapter_send_command():
    """send_command extracts [vn, ve, vd, yaw_rate] and calls fc.send_velocity_ned."""
    fc = AsyncMock()
    fc.armed = False
    adapter = DroneMotorAdapter(fc)
    cmd = np.array([1.0, 0.5, -0.3, 0.2], dtype=np.float32)
    await adapter.send_command(cmd)
    fc.send_velocity_ned.assert_awaited_once()
    args = fc.send_velocity_ned.await_args.args
    assert args[0] == pytest.approx(1.0)
    assert args[1] == pytest.approx(0.5)
    assert args[2] == pytest.approx(-0.3)
    assert args[3] == pytest.approx(0.2)


async def test_drone_adapter_send_command_partial():
    """send_command handles partial arrays gracefully."""
    fc = AsyncMock()
    fc.armed = False
    adapter = DroneMotorAdapter(fc)
    cmd = np.array([1.0, 0.5], dtype=np.float32)
    await adapter.send_command(cmd)
    fc.send_velocity_ned.assert_awaited_once_with(1.0, 0.5, 0.0, 0.0)


async def test_drone_adapter_read_state():
    """read_state returns 7-element state vector."""
    fc = AsyncMock()
    fc.armed = True
    fc.get_imu_data = AsyncMock(
        return_value=np.array([0.1, 0.2, 9.8, 0.01, 0.02, 0.03], dtype=np.float32)
    )
    fc.get_altitude_m = AsyncMock(return_value=10.0)
    fc.get_battery_voltage = AsyncMock(return_value=16.0)
    adapter = DroneMotorAdapter(fc)

    state = await adapter.read_state()
    assert state.shape == (7,)
    assert state[4] == pytest.approx(10.0)  # altitude
    assert state[5] == pytest.approx(16.0)  # battery
    assert state[6] == pytest.approx(1.0)  # armed flag


async def test_drone_adapter_get_battery():
    fc = AsyncMock()
    fc.armed = False
    fc.get_battery_voltage = AsyncMock(return_value=15.5)
    adapter = DroneMotorAdapter(fc)
    voltage = await adapter.get_battery_voltage()
    assert voltage == pytest.approx(15.5)


def test_drone_adapter_platform_type():
    fc = AsyncMock()
    fc.armed = False
    adapter = DroneMotorAdapter(fc)
    assert adapter.platform_type == "drone"
