"""Phase 1 domain randomization utilities for sim-to-real pretraining."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from mousedroid.config.schema import DomainRandomizationConfig, RangeF

_UINT8_MAX_F = np.float32(255.0)


@dataclass(frozen=True)
class EpisodeParams:
    """Per-episode randomization parameters sampled from configured ranges."""

    visual: Mapping[str, float] = field(default_factory=dict)
    camera: Mapping[str, float] = field(default_factory=dict)
    chassis: Mapping[str, float] = field(default_factory=dict)
    comms: Mapping[str, float] = field(default_factory=dict)
    disturbance: Mapping[str, float] = field(default_factory=dict)
    feature: Mapping[str, float] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        """Return True when no episode-specific parameters are populated."""
        return not any(
            (
                self.visual,
                self.camera,
                self.chassis,
                self.comms,
                self.disturbance,
                self.feature,
            )
        )


def _sample_range(rng_cfg: RangeF, rng: np.random.Generator) -> float:
    """Sample one value from an inclusive floating-point range."""
    if rng_cfg.low == rng_cfg.high:
        return rng_cfg.low
    return float(rng.uniform(rng_cfg.low, rng_cfg.high))


class DomainRandomizer:
    """Sample per-episode domain randomization parameters."""

    def __init__(self, cfg: DomainRandomizationConfig) -> None:
        self._cfg = cfg

    @property
    def enabled(self) -> bool:
        """Return True when domain randomization is enabled."""
        return self._cfg.enabled

    def sample(self, rng: np.random.Generator) -> EpisodeParams:
        """Sample one episode worth of parameters from the configured ranges."""
        if not self.enabled:
            return EpisodeParams()

        return EpisodeParams(
            visual={
                "brightness": _sample_range(self._cfg.brightness, rng),
                "contrast": _sample_range(self._cfg.contrast, rng),
                "hue_shift_deg": _sample_range(self._cfg.hue_shift_deg, rng),
                "gaussian_noise_std": _sample_range(self._cfg.gaussian_noise_std, rng),
                "motion_blur_px": _sample_range(self._cfg.motion_blur_px, rng),
            },
            camera={
                "fov_deg": _sample_range(self._cfg.fov_deg, rng),
                "pitch_deg": _sample_range(self._cfg.cam_pitch_deg, rng),
                "yaw_deg": _sample_range(self._cfg.cam_yaw_deg, rng),
                "height_m": _sample_range(self._cfg.cam_height_m, rng),
            },
            chassis={
                "friction": _sample_range(self._cfg.wheel_friction, rng),
                "slip": _sample_range(self._cfg.wheel_slip, rng),
                "mass_kg": _sample_range(self._cfg.chassis_mass_kg, rng),
                "motor_gain": _sample_range(self._cfg.motor_gain, rng),
            },
            comms={
                "uart_latency_ms": _sample_range(self._cfg.uart_latency_ms, rng),
                "encoder_dropout_prob": _sample_range(self._cfg.encoder_dropout_prob, rng),
            },
            disturbance={
                "push_force_n": _sample_range(self._cfg.push_force_n, rng),
                "push_event_prob": float(self._cfg.push_event_prob),
            },
            feature={
                "noise_std": _sample_range(self._cfg.feature_noise_std, rng),
            },
        )


def apply_visual_randomization(
    frame: NDArray[np.uint8] | NDArray[np.float32],
    params: EpisodeParams,
    rng: np.random.Generator,
) -> NDArray[np.uint8] | NDArray[np.float32]:
    """Apply lightweight visual perturbations to a frame."""
    if params.is_empty or not params.visual:
        return frame

    visual = params.visual
    original_dtype = frame.dtype
    randomized = frame.astype(np.float32, copy=False)
    if original_dtype == np.uint8:
        randomized = randomized / _UINT8_MAX_F

    randomized = randomized * np.float32(visual.get("brightness", 1.0))
    randomized = (randomized - 0.5) * np.float32(visual.get("contrast", 1.0)) + 0.5

    noise_std = float(visual.get("gaussian_noise_std", 0.0))
    if noise_std > 0.0:
        randomized = randomized + rng.normal(0.0, noise_std, size=randomized.shape).astype(
            np.float32
        )

    blur_radius = int(round(float(visual.get("motion_blur_px", 0.0))))
    if blur_radius > 0 and randomized.ndim >= 2:
        window = blur_radius * 2 + 1
        if randomized.ndim == 3:
            padded = np.pad(
                randomized,
                ((0, 0), (blur_radius, blur_radius), (0, 0)),
                mode="edge",
            )
            blurred = np.zeros_like(randomized)
            for offset in range(window):
                blurred += padded[:, offset : offset + randomized.shape[1], :]
        else:
            padded = np.pad(randomized, ((0, 0), (blur_radius, blur_radius)), mode="edge")
            blurred = np.zeros_like(randomized)
            for offset in range(window):
                blurred += padded[:, offset : offset + randomized.shape[1]]
        randomized = blurred / np.float32(window)

    randomized = np.clip(randomized, 0.0, 1.0)

    if original_dtype == np.uint8:
        return np.clip(randomized * _UINT8_MAX_F, 0.0, _UINT8_MAX_F).astype(np.uint8)
    return randomized.astype(np.float32, copy=False)
def apply_feature_noise(
    features: NDArray[np.float32],
    params: Mapping[str, float],
    rng: np.random.Generator,
) -> NDArray[np.float32]:
    """Apply additive Gaussian feature noise while preserving shape."""
    feature_array = np.asarray(features, dtype=np.float32)
    noise_std = float(params.get("noise_std", 0.0))
    if noise_std == 0.0:
        return feature_array.copy()
    noise = rng.normal(0.0, noise_std, size=feature_array.shape).astype(np.float32)
    return feature_array + noise


__all__ = [
    "DomainRandomizer",
    "EpisodeParams",
    "apply_feature_noise",
    "apply_visual_randomization",
]
