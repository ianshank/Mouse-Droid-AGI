"""Kalman-filtered sensor fusion: MiDaS depth + ultrasonic distance.

Fuses monocular depth estimation with HC-SR04 ultrasonic point readings
to produce calibrated absolute-distance depth maps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import FusionConfig

_log = get_logger(__name__)


@dataclass
class FusedDepthResult:
    """Result of depth fusion."""

    depth_map: NDArray[np.float32]
    """Calibrated depth map in meters, shape ``(H, W)``."""

    center_distance_m: float
    """Fused center distance in meters (Kalman-filtered)."""

    ultrasonic_distance_m: float
    """Raw ultrasonic reading in meters."""

    timestamp: float
    """Unix timestamp."""


class KalmanDepthFusion:
    """1D Kalman filter fusing ultrasonic point + MiDaS depth center.

    The ultrasonic sensor provides an absolute distance measurement
    at the center of the field of view. MiDaS provides a relative
    inverse-depth map. The Kalman filter tracks the scale factor
    between MiDaS and real-world distance.

    The fused output is a calibrated depth map in meters.

    Parameters
    ----------
    cfg:
        Fusion configuration with Kalman noise parameters.
    """

    def __init__(self, cfg: FusionConfig) -> None:
        self._cfg = cfg
        # Kalman state: [distance_m, scale_factor]
        self._x = np.array([1.0, 1.0], dtype=np.float64)  # state
        self._P = np.eye(2, dtype=np.float64) * 1.0  # covariance
        self._Q = np.eye(2, dtype=np.float64) * cfg.kalman_process_noise  # process noise
        self._R_ultrasonic = cfg.kalman_ultrasonic_noise  # measurement noise
        self._R_midas = cfg.kalman_midas_noise  # MiDaS noise

    def fuse(
        self,
        depth_map: NDArray[np.float32],
        ultrasonic_m: float,
        timestamp: float,
    ) -> FusedDepthResult:
        """Fuse depth map with ultrasonic reading.

        Args:
            depth_map: Normalised MiDaS inverse depth, shape ``(H, W)``,
                values in [0, 1] where higher = closer.
            ultrasonic_m: HC-SR04 distance in metres (0.02–4.0 m range).
            timestamp: Unix timestamp.

        Returns:
            Fused depth result with calibrated depth map.
        """
        h, w = depth_map.shape

        # Extract center region (matching approximate ultrasonic FOV)
        fov_frac = self._cfg.ultrasonic_fov_fraction
        cy, cx = h // 2, w // 2
        half_h = max(1, int(h * fov_frac / 2))
        half_w = max(1, int(w * fov_frac / 2))
        center_region = depth_map[
            cy - half_h : cy + half_h,
            cx - half_w : cx + half_w,
        ]
        midas_center = float(np.mean(center_region)) if center_region.size > 0 else 0.5

        # --- Kalman predict ---
        # State transition: distance and scale factor evolve slowly
        F = np.eye(2, dtype=np.float64)
        self._x = F @ self._x
        self._P = F @ self._P @ F.T + self._Q

        # --- Kalman update (ultrasonic measurement) ---
        if 0.02 <= ultrasonic_m <= 4.0:
            # Valid ultrasonic reading
            H_ultra = np.array([[1.0, 0.0]], dtype=np.float64)
            z_ultra = np.array([ultrasonic_m], dtype=np.float64)
            R_ultra = np.array([[self._R_ultrasonic]], dtype=np.float64)

            y_ultra = z_ultra - H_ultra @ self._x
            S_ultra = H_ultra @ self._P @ H_ultra.T + R_ultra
            K_ultra = self._P @ H_ultra.T @ np.linalg.inv(S_ultra)
            self._x = self._x + (K_ultra @ y_ultra).flatten()
            self._P = (np.eye(2) - K_ultra @ H_ultra) @ self._P

        # --- Kalman update (MiDaS center depth) ---
        if midas_center > 1e-4:
            # MiDaS inverse depth → distance estimate using scale
            midas_dist = 1.0 / (midas_center + 1e-6)

            # Scale factor update via simple adaptive correction
            if self._x[0] > 0.02:
                new_scale = self._x[0] / midas_dist
                scale_err = new_scale - self._x[1]
                self._x[1] += 0.1 * scale_err  # Smooth scale adaptation

        # --- Apply calibrated scale to full depth map ---
        scale = max(self._x[1], 0.01)
        calibrated_depth = np.where(
            depth_map > 1e-4,
            scale / (depth_map + 1e-6),
            0.0,
        ).astype(np.float32)

        # Clip to reasonable range
        calibrated_depth = np.clip(calibrated_depth, 0.0, 10.0)

        fused_distance = float(self._x[0])

        return FusedDepthResult(
            depth_map=calibrated_depth,
            center_distance_m=max(0.0, fused_distance),
            ultrasonic_distance_m=ultrasonic_m,
            timestamp=timestamp,
        )
