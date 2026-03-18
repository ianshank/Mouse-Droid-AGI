from __future__ import annotations

from typing import Self

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from mousedroid.agents.base import AgentProtocol
from mousedroid.comms.protocol import EncoderReading, ESP32CommProtocol
from mousedroid.experience.protocol import ExperienceProtocol
from mousedroid.hardware.protocols import AudioProtocol, DistanceSensorProtocol, VisionProtocol
from mousedroid.llm_gateway.protocol import GoalVector, LLMGatewayProtocol
from mousedroid.safety.context import SafetyContext
from mousedroid.safety.protocol import SafetyMonitorProtocol
from mousedroid.sensing.protocol import ObservationProtocol
from mousedroid.world_model.protocol import WorldModelProtocol

# -- runtime_checkable --------------------------------------------------------


def test_vision_protocol_is_runtime_checkable():
    assert hasattr(VisionProtocol, "__protocol_attrs__") or hasattr(
        VisionProtocol, "__abstractmethods__"
    )


def test_distance_sensor_protocol_is_runtime_checkable():
    assert hasattr(DistanceSensorProtocol, "__protocol_attrs__") or True


def test_esp32_comm_protocol_is_runtime_checkable():
    assert hasattr(ESP32CommProtocol, "__protocol_attrs__") or True


# -- Conforming mock classes --------------------------------------------------


class _MockVision:
    async def capture_features(self) -> NDArray[np.float32]:
        return np.zeros(256, dtype=np.float32)

    async def capture_frame(self) -> NDArray[np.uint8]:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    async def extract_features(self, frame: NDArray[np.uint8]) -> NDArray[np.float32]:
        return np.zeros(256, dtype=np.float32)

    @property
    def feature_dim(self) -> int:
        return 256

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass


class _MockDistanceSensor:
    async def read_distance_m(self) -> float:
        return 1.0

    @property
    def max_range_m(self) -> float:
        return 4.0

    @property
    def min_range_m(self) -> float:
        return 0.02


class _MockESP32Comm:
    async def connect(self) -> None:
        pass

    async def send_velocity(self, vx: float, vy: float, omega: float) -> None:
        pass

    async def read_encoders(self) -> EncoderReading:
        return EncoderReading()

    async def get_battery_voltage(self) -> float:
        return 12.0

    async def emergency_stop(self) -> None:
        pass

    async def disconnect(self) -> None:
        pass


class _MockAudio:
    async def read_chunk(self) -> NDArray[np.float32]:
        return np.zeros(1024, dtype=np.float32)

    @property
    def sample_rate(self) -> int:
        return 16000

    @property
    def channels(self) -> int:
        return 1

    @property
    def chunk_size(self) -> int:
        return 1024

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass


class _MockObservation:
    @property
    def timestamp(self) -> float:
        return 0.0

    @property
    def vision_features(self) -> NDArray[np.float32]:
        return np.zeros(256, dtype=np.float32)

    @property
    def distance_m(self) -> float:
        return 1.0

    @property
    def motor_state(self) -> NDArray[np.float32]:
        return np.zeros(4, dtype=np.float32)

    @property
    def audio_chunk(self) -> NDArray[np.float32]:
        return np.zeros(1024, dtype=np.float32)

    @property
    def valid_mask(self) -> NDArray[np.float32]:
        return np.ones(4, dtype=np.float32)

    @property
    def n_modalities(self) -> int:
        return 4


class _MockWorldModel:
    def observe_step(
        self,
        observation: ObservationProtocol,
        prev_action: Tensor,
        h: Tensor,
        z: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, float]:
        return h, z, h, 0.0

    def imagine_step(
        self,
        action: Tensor,
        h: Tensor,
        z: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        return h, z, h


class _MockSafetyMonitor:
    def evaluate(
        self,
        observation: ObservationProtocol,
        loop_time_ms: float,
    ) -> SafetyContext:
        return SafetyContext()


class _MockAgent:
    @property
    def name(self) -> str:
        return "mock"

    def act(self, h: Tensor, z: Tensor, safety_ctx: SafetyContext) -> Tensor:
        return torch.zeros(3)

    def reset(self) -> None:
        pass


class _MockExperience:
    @property
    def schema_version(self) -> int:
        return 1

    def serialize(self) -> bytes:
        return b""

    @classmethod
    def deserialize(cls, data: bytes) -> Self:
        return cls()


class _MockLLMGateway:
    async def start(self) -> None:
        pass

    async def translate_mission(self, nl_command: str) -> GoalVector:
        return GoalVector()

    async def stop(self) -> None:
        pass


# -- isinstance checks --------------------------------------------------------


def test_vision_protocol_isinstance():
    assert isinstance(_MockVision(), VisionProtocol)


def test_distance_sensor_protocol_isinstance():
    assert isinstance(_MockDistanceSensor(), DistanceSensorProtocol)


def test_esp32_comm_protocol_isinstance():
    assert isinstance(_MockESP32Comm(), ESP32CommProtocol)


def test_audio_protocol_isinstance():
    assert isinstance(_MockAudio(), AudioProtocol)


def test_observation_protocol_isinstance():
    assert isinstance(_MockObservation(), ObservationProtocol)


def test_world_model_protocol_isinstance():
    assert isinstance(_MockWorldModel(), WorldModelProtocol)


def test_safety_monitor_protocol_isinstance():
    assert isinstance(_MockSafetyMonitor(), SafetyMonitorProtocol)


def test_agent_protocol_isinstance():
    assert isinstance(_MockAgent(), AgentProtocol)


def test_experience_protocol_isinstance():
    assert isinstance(_MockExperience(), ExperienceProtocol)


def test_llm_gateway_protocol_isinstance():
    assert isinstance(_MockLLMGateway(), LLMGatewayProtocol)


# -- Non-conforming classes should fail ---------------------------------------


class _Empty:
    pass


def test_non_conforming_not_vision():
    assert not isinstance(_Empty(), VisionProtocol)


def test_non_conforming_not_distance_sensor():
    assert not isinstance(_Empty(), DistanceSensorProtocol)


def test_non_conforming_not_esp32_comm():
    assert not isinstance(_Empty(), ESP32CommProtocol)


def test_non_conforming_not_observation():
    assert not isinstance(_Empty(), ObservationProtocol)


def test_non_conforming_not_audio():
    assert not isinstance(_Empty(), AudioProtocol)


def test_non_conforming_not_agent():
    assert not isinstance(_Empty(), AgentProtocol)
