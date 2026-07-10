"""Domain randomization for sim-to-real RSSM pretraining (Phase 1).

Per-episode randomization of physical and sensor parameters so the world
model and downstream policies generalize beyond a single nominal simulator
configuration. All ranges come from
:class:`mousedroid.config.schema.DomainRandomizationConfig`; nothing here is
hardcoded.

Reproducibility: every public function takes an explicit
:class:`numpy.random.Generator` so seeds propagate from the training pipeline
and a disabled config produces empty :class:`EpisodeParams`, leaving the data
generator output byte-identical to the pre-feature behaviour.

References:
    Tobin et al. (2017), Domain randomization for transferring deep neural
    networks from simulation to the real world.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import numpy as np

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mousedroid.config.schema import DomainRandomizationConfig, RangeF

_log = get_logger(__name__)

# uint8 range max — used to map between [0, 255] uint8 frames and [0, 1] float
# frames during visual randomization. Not a tunable; documented here so the
# hardcoded-values gate doesn't flag the literal in transform bodies.
_UINT8_MAX_F = np.float32(255.0)  # hardcoded-ok: uint8 range max, not a tunable


@dataclass(frozen=True)
class EpisodeParams:
    """Concrete physical/sensor parameters drawn for one synthetic episode.

    All sub-mappings are empty when randomization is disabled, making the
    "DR off" path indistinguishable from the legacy data generator output.
    """

    visual: Mapping[str, float] = field(default_factory=dict)
    camera: Mapping[str, float] = field(default_factory=dict)
    range_sensor: Mapping[str, float] = field(default_factory=dict)
    chassis: Mapping[str, float] = field(default_factory=dict)
    comms: Mapping[str, float] = field(default_factory=dict)
    disturbance: Mapping[str, float] = field(default_factory=dict)
    feature: Mapping[str, float] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        """True if no group carries any sampled parameter."""
        return not (
            self.visual
            or self.camera
            or self.range_sensor
            or self.chassis
            or self.comms
            or self.disturbance
            or self.feature
        )


class DomainRandomizer:
    """Stateless sampler that draws an :class:`EpisodeParams` bundle from config.

    The randomizer never owns its own RNG; callers thread one through so
    seeding is centralized in the training pipeline.
    """

    def __init__(self, cfg: DomainRandomizationConfig) -> None:
        self._cfg = cfg

    @property
    def enabled(self) -> bool:
        """Whether the underlying config has randomization enabled."""
        return self._cfg.enabled

    def sample(self, rng: np.random.Generator) -> EpisodeParams:
        """Draw a fresh per-episode parameter bundle.

        Args:
            rng: Per-call RNG; identical seeds produce identical bundles.

        Returns:
            Sampled :class:`EpisodeParams`. Empty when config is disabled.
        """
        if not self._cfg.enabled:
            _log.debug("dr_disabled")
            return EpisodeParams()

        sample = _sample_range
        params = EpisodeParams(
            visual={
                "brightness": sample(self._cfg.brightness, rng),
                "contrast": sample(self._cfg.contrast, rng),
                "hue_shift_deg": sample(self._cfg.hue_shift_deg, rng),
                "gaussian_noise_std": sample(self._cfg.gaussian_noise_std, rng),
                "motion_blur_px": sample(self._cfg.motion_blur_px, rng),
            },
            camera={
                "fov_deg": sample(self._cfg.fov_deg, rng),
                "pitch_deg": sample(self._cfg.cam_pitch_deg, rng),
                "yaw_deg": sample(self._cfg.cam_yaw_deg, rng),
                "height_m": sample(self._cfg.cam_height_m, rng),
            },
            range_sensor={
                "noise_m": sample(self._cfg.ultrasonic_noise_m, rng),
                "dropout_prob": sample(self._cfg.ultrasonic_dropout_prob, rng),
            },
            chassis={
                "friction": sample(self._cfg.wheel_friction, rng),
                "slip": sample(self._cfg.wheel_slip, rng),
                "mass_kg": sample(self._cfg.chassis_mass_kg, rng),
                "motor_gain": sample(self._cfg.motor_gain, rng),
            },
            comms={
                "uart_latency_ms": sample(self._cfg.uart_latency_ms, rng),
                "encoder_dropout_prob": sample(self._cfg.encoder_dropout_prob, rng),
            },
            disturbance={
                "push_force_n": sample(self._cfg.push_force_n, rng),
                "push_event_prob": float(self._cfg.push_event_prob),
            },
            feature={
                "noise_std": sample(self._cfg.feature_noise_std, rng),
            },
        )
        _log.debug(
            "dr_sampled",
            visual=dict(params.visual),
            chassis=dict(params.chassis),
            comms=dict(params.comms),
        )
        return params


def _sample_range(rng_cfg: RangeF, rng: np.random.Generator) -> float:
    """Draw a uniform sample from a :class:`RangeF` (degenerate ranges return ``low``)."""
    if rng_cfg.low == rng_cfg.high:
        return float(rng_cfg.low)
    return float(rng.uniform(rng_cfg.low, rng_cfg.high))


# ---------------------------------------------------------------------------
# Observation-space transforms
# ---------------------------------------------------------------------------


def apply_visual_randomization(
    frame: NDArray[np.generic],
    params: Mapping[str, float],
    rng: np.random.Generator,
) -> NDArray[np.generic]:
    """Apply per-episode visual randomization to a raw RGB frame.

    Brightness scaling, contrast around per-frame mean, and additive Gaussian
    noise. Hue shift and motion blur are reserved for future image-space
    integration once raw frames flow through the data generator.

    Args:
        frame: ``(H, W, 3)`` ``uint8`` or ``float32`` in ``[0, 1]``.
        params: Visual sub-mapping from :class:`EpisodeParams`.
        rng: Per-frame RNG for the additive noise realization.

    Returns:
        Randomized frame, same dtype and shape as ``frame``.
    """
    is_uint8 = frame.dtype == np.uint8
    work: NDArray[np.float32] = frame.astype(np.float32, copy=True)
    if is_uint8:
        work = (work / _UINT8_MAX_F).astype(np.float32, copy=False)

    brightness = float(params.get("brightness", 1.0))
    contrast = float(params.get("contrast", 1.0))
    noise_std = float(params.get("gaussian_noise_std", 0.0))

    work = (work * np.float32(brightness)).astype(np.float32, copy=False)
    mean = work.mean(axis=(0, 1), keepdims=True)
    work = ((work - mean) * np.float32(contrast) + mean).astype(np.float32, copy=False)

    if noise_std > 0.0:
        noise = rng.standard_normal(work.shape, dtype=np.float32) * np.float32(noise_std)
        work = (work + noise).astype(np.float32, copy=False)

    np.clip(work, 0.0, 1.0, out=work)

    if is_uint8:
        # ``cast`` because numpy's astype on the (Any-typed) product returns
        # Any under mypy 2.2.0's stricter [no-any-return] check.
        return cast("NDArray[np.generic]", (work * _UINT8_MAX_F).astype(np.uint8))
    return work.astype(frame.dtype, copy=False)


def apply_range_sensor_randomization(
    distance_m: float,
    params: Mapping[str, float],
    rng: np.random.Generator,
) -> float:
    """Add Gaussian noise + stochastic dropout to a single range reading.

    Args:
        distance_m: Nominal range reading.
        params: Range-sensor sub-mapping from :class:`EpisodeParams`.
        rng: Per-call RNG.

    Returns:
        Noisy reading, or ``float('nan')`` when a dropout event fires.
    """
    dropout_prob = float(params.get("dropout_prob", 0.0))
    noise_m = float(params.get("noise_m", 0.0))
    if dropout_prob > 0.0 and rng.random() < dropout_prob:
        return float("nan")
    if noise_m == 0.0:
        return float(distance_m)
    return float(distance_m + rng.normal(0.0, noise_m))


def apply_feature_noise(
    features: NDArray[np.generic],
    params: Mapping[str, float],
    rng: np.random.Generator,
) -> NDArray[np.generic]:
    """Add Gaussian noise to a post-CNN feature vector.

    Args:
        features: ``(D,)`` or ``(N, D)`` float feature tensor.
        params: Feature sub-mapping from :class:`EpisodeParams`.
        rng: Per-call RNG.

    Returns:
        Noisy feature tensor, same shape and dtype as ``features``.
    """
    noise_std = float(params.get("noise_std", 0.0))
    if noise_std == 0.0:
        return features
    noise = rng.standard_normal(features.shape, dtype=np.float32) * np.float32(noise_std)
    # ``cast`` because numpy's astype returns Any under mypy 2.2.0's [no-any-return].
    return cast(
        "NDArray[np.generic]",
        (features.astype(np.float32, copy=False) + noise).astype(features.dtype, copy=False),
    )


__all__ = [
    "DomainRandomizer",
    "EpisodeParams",
    "apply_feature_noise",
    "apply_range_sensor_randomization",
    "apply_visual_randomization",
]
