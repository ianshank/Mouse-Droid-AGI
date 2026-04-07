"""Domain randomization for sim-to-real transfer.

Applies configurable randomization to MuJoCo physics parameters,
visual properties, and camera poses to improve policy robustness.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import ArmSimConfig

_log = get_logger(__name__)


class DomainRandomizer:
    """Apply domain randomization to simulation parameters.

    Randomizes mass, friction, lighting, and camera pose within
    configured ranges to bridge the sim-to-real gap.

    Args:
        sim_cfg: Simulation configuration with randomization ranges.
        seed: Random seed for reproducibility.
    """

    def __init__(self, sim_cfg: ArmSimConfig, seed: int = 42) -> None:
        """Initialise domain randomizer.

        Args:
            sim_cfg: Sim config with randomization ranges.
            seed: Random seed.
        """
        self._cfg = sim_cfg
        self._rng = np.random.default_rng(seed)
        self._enabled = sim_cfg.domain_randomization
        _log.info(
            "domain_randomizer_init",
            enabled=self._enabled,
            mass_range_pct=sim_cfg.mass_range_pct,
            friction_range=sim_cfg.friction_range,
        )

    def randomize_mass(self, nominal_mass: float) -> float:
        """Randomize object mass within configured range.

        Args:
            nominal_mass: Nominal mass value (kg).

        Returns:
            Randomized mass value.
        """
        if not self._enabled:
            return nominal_mass
        pct = self._cfg.mass_range_pct / 100.0
        scale = self._rng.uniform(1.0 - pct, 1.0 + pct)
        return float(nominal_mass * scale)

    def randomize_friction(self, nominal_friction: float) -> float:
        """Randomize surface friction coefficient.

        Args:
            nominal_friction: Nominal friction coefficient.

        Returns:
            Randomized friction (clamped to [0, inf]).
        """
        if not self._enabled:
            return nominal_friction
        delta = self._rng.uniform(-self._cfg.friction_range, self._cfg.friction_range)
        return float(max(0.0, nominal_friction + delta))

    def randomize_position(self, nominal_position: NDArray[np.float64]) -> NDArray[np.float64]:
        """Randomize object position with Gaussian noise.

        Args:
            nominal_position: Nominal XYZ position, shape ``(3,)``.

        Returns:
            Randomized position.
        """
        if not self._enabled:
            return nominal_position
        noise = self._rng.normal(0.0, self._cfg.position_noise_m, size=3)
        return nominal_position + noise

    def randomize_lighting(self, nominal_intensity: float) -> float:
        """Randomize scene lighting intensity.

        Args:
            nominal_intensity: Nominal light intensity [0, 1].

        Returns:
            Randomized intensity clamped to [0, 1].
        """
        if not self._enabled:
            return nominal_intensity
        delta = self._rng.uniform(-self._cfg.lighting_variation, self._cfg.lighting_variation)
        return float(np.clip(nominal_intensity + delta, 0.0, 1.0))

    def randomize_camera_pose(self, nominal_angles_deg: NDArray[np.float64]) -> NDArray[np.float64]:
        """Randomize camera orientation.

        Args:
            nominal_angles_deg: Nominal Euler angles in degrees, shape ``(3,)``.

        Returns:
            Randomized angles in degrees.
        """
        if not self._enabled:
            return nominal_angles_deg
        noise = self._rng.uniform(
            -self._cfg.camera_pose_noise_deg,
            self._cfg.camera_pose_noise_deg,
            size=3,
        )
        return nominal_angles_deg + noise

    def apply_all(self, scene_params: dict[str, Any]) -> dict[str, Any]:
        """Apply all randomizations to a scene parameter dict.

        Args:
            scene_params: Dict with keys: mass, friction, position, lighting, camera_angles.

        Returns:
            Randomized scene parameters.
        """
        randomized = dict(scene_params)

        if "mass" in randomized:
            randomized["mass"] = self.randomize_mass(randomized["mass"])
        if "friction" in randomized:
            randomized["friction"] = self.randomize_friction(randomized["friction"])
        if "position" in randomized:
            randomized["position"] = self.randomize_position(randomized["position"])
        if "lighting" in randomized:
            randomized["lighting"] = self.randomize_lighting(randomized["lighting"])
        if "camera_angles" in randomized:
            randomized["camera_angles"] = self.randomize_camera_pose(randomized["camera_angles"])

        return randomized
