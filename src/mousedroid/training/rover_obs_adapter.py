"""Adapter: rover env obs dict (+ step info) -> RSSM encoder-input tensors."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from mousedroid.constants import N_SENSOR_MODALITIES_WITH_LIDAR, SENSOR_SLOT_MAP

_VISION_SLOT = SENSOR_SLOT_MAP["vision"]
_N_SLOTS = N_SENSOR_MODALITIES_WITH_LIDAR  # [vision, ultrasonic, motor, audio, lidar]


class RoverObsAdapter:
    """Map a rover observation dict to the modality vectors the RSSM encoder reads.

    Vision is OMITTED (the pretraining RSSM is built with ``vision_dim=0``); the
    mask's vision slot is set to 0. ``motor_state`` is synthesised as
    ``[vx, vy=0, omega, battery_v]`` from the env ``info`` (skid-steer ⇒ vy=0).
    Pure + deterministic — no hidden state.
    """

    def __init__(self, *, battery_v: float) -> None:
        """Initialise the adapter.

        Args:
            battery_v: Constant battery voltage stamped into ``motor_state[3]``.
        """
        self._battery_v = float(battery_v)

    def adapt(
        self, obs: dict[str, NDArray[np.float32]], info: dict[str, Any]
    ) -> dict[str, NDArray[np.float32]]:
        """Adapt one ``(obs, info)`` pair to RSSM encoder inputs.

        Args:
            obs: Rover observation dict (``imu``/``chassis_pose``/``wheel_vel``/``lidar``).
            info: The env ``step`` info dict (carries ``vx_body_mps`` / ``omega_rads``).

        Returns:
            Dict with ``motor`` (4,), ``ultrasonic`` (1,), ``valid_mask`` (5,), and
            ``lidar`` (N,) when the rover exposes a lidar.
        """
        vx = float(info.get("vx_body_mps", 0.0))
        omega = float(info.get("omega_rads", 0.0))
        motor = np.asarray([vx, 0.0, omega, self._battery_v], dtype=np.float32)

        lidar = np.asarray(obs.get("lidar", np.zeros(0, dtype=np.float32)), dtype=np.float32)
        # Forward range = min normalised lidar sector (or 1.0 when no lidar).
        forward = float(lidar.min()) if lidar.size else 1.0
        ultrasonic = np.asarray([forward], dtype=np.float32)

        mask = np.ones(_N_SLOTS, dtype=np.float32)
        mask[_VISION_SLOT] = 0.0  # vision omitted

        out: dict[str, NDArray[np.float32]] = {
            "motor": motor,
            "ultrasonic": ultrasonic,
            "valid_mask": mask,
        }
        if lidar.size:
            out["lidar"] = lidar
        return out
