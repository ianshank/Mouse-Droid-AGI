"""Experience record — serializable experience data with schema versioning."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Any

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

import msgpack
import numpy as np
from numpy.typing import NDArray

from mousedroid.constants import (
    DEFAULT_ACTION_DIM,
    DEFAULT_MAX_DISTANCE_M,
    DEFAULT_MOTOR_STATE_DIM,
    DEFAULT_VISION_DIM,
)
from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)

SCHEMA_VERSION: int = 1


@dataclass
class MouseDroidExperienceRecord:
    """Experience record for MouseDroid observations and actions.

    Schema version is fixed at 1 — new record types get new version constants.
    """

    schema_version: int = SCHEMA_VERSION
    timestamp: float = field(default_factory=time.time)
    vision_features: NDArray[np.float32] = field(
        default_factory=lambda: np.zeros(DEFAULT_VISION_DIM, dtype=np.float32),
    )
    distance_m: float = DEFAULT_MAX_DISTANCE_M
    motor_state: NDArray[np.float32] = field(
        default_factory=lambda: np.zeros(DEFAULT_MOTOR_STATE_DIM, dtype=np.float32),
    )
    action: NDArray[np.float32] = field(
        default_factory=lambda: np.zeros(DEFAULT_ACTION_DIM, dtype=np.float32),
    )
    reward: float = 0.0
    surprise: float = 0.0

    @property
    def embedding(self) -> NDArray[np.float32]:
        """Return vision features as the embedding for memory consolidation."""
        return self.vision_features

    def serialize(self) -> bytes:
        """Serialize record to msgpack bytes.

        Returns:
            Msgpack-encoded bytes.
        """
        data: dict[str, Any] = {
            "schema_version": self.schema_version,
            "timestamp": self.timestamp,
            "vision_features": self.vision_features.tobytes(),
            "vision_features_shape": list(self.vision_features.shape),
            "distance_m": self.distance_m,
            "motor_state": self.motor_state.tobytes(),
            "motor_state_shape": list(self.motor_state.shape),
            "action": self.action.tobytes(),
            "action_shape": list(self.action.shape),
            "reward": self.reward,
            "surprise": self.surprise,
        }
        result: bytes = msgpack.packb(data)
        return result

    @classmethod
    def deserialize(cls, data: bytes) -> Self:
        """Deserialize record from msgpack bytes.

        Args:
            data: Msgpack-encoded bytes.

        Returns:
            Deserialized record instance.

        Raises:
            ValueError: If schema version does not match.
        """
        unpacked: dict[str, Any] = msgpack.unpackb(data, raw=False)
        version = unpacked["schema_version"]
        if version != SCHEMA_VERSION:
            msg = f"Unknown schema version: {version}"
            raise ValueError(msg)
        return cls(
            schema_version=version,
            timestamp=unpacked["timestamp"],
            vision_features=np.frombuffer(
                unpacked["vision_features"],
                dtype=np.float32,
            ).reshape(unpacked["vision_features_shape"]),
            distance_m=unpacked["distance_m"],
            motor_state=np.frombuffer(
                unpacked["motor_state"],
                dtype=np.float32,
            ).reshape(unpacked["motor_state_shape"]),
            action=np.frombuffer(
                unpacked["action"],
                dtype=np.float32,
            ).reshape(unpacked["action_shape"]),
            reward=unpacked["reward"],
            surprise=unpacked["surprise"],
        )


def deserialize_any(data: bytes) -> MouseDroidExperienceRecord:
    """Auto-detect schema version and deserialize to correct record type.

    Args:
        data: Msgpack-encoded bytes.

    Returns:
        Deserialized experience record.

    Raises:
        ValueError: If schema version is unknown.
    """
    unpacked: dict[str, Any] = msgpack.unpackb(data, raw=False)
    version = unpacked.get("schema_version")
    if version == SCHEMA_VERSION:
        return MouseDroidExperienceRecord.deserialize(data)
    msg = f"Unknown schema version: {version}"
    raise ValueError(msg)
