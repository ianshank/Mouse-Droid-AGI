"""Tests for domain randomization."""

from __future__ import annotations

import numpy as np

from mousedroid.arm.environments.domain_randomizer import DomainRandomizer
from mousedroid.config.schema import ArmSimConfig


def _make_randomizer(enabled: bool = True, seed: int = 42) -> DomainRandomizer:
    """Create domain randomizer with test defaults."""
    cfg = ArmSimConfig(domain_randomization=enabled)
    return DomainRandomizer(cfg, seed=seed)


class TestDomainRandomizer:
    """Test DomainRandomizer randomization functions."""

    def test_mass_randomization_within_range(self) -> None:
        dr = _make_randomizer(seed=42)
        nominal = 1.0
        for _ in range(100):
            randomized = dr.randomize_mass(nominal)
            assert 0.8 <= randomized <= 1.2  # ±20%

    def test_mass_disabled_returns_nominal(self) -> None:
        dr = _make_randomizer(enabled=False)
        assert dr.randomize_mass(1.0) == 1.0

    def test_friction_randomization_non_negative(self) -> None:
        dr = _make_randomizer(seed=42)
        for _ in range(100):
            friction = dr.randomize_friction(0.5)
            assert friction >= 0.0

    def test_friction_disabled_returns_nominal(self) -> None:
        dr = _make_randomizer(enabled=False)
        assert dr.randomize_friction(0.5) == 0.5

    def test_position_noise_applied(self) -> None:
        dr = _make_randomizer(seed=42)
        pos = np.array([0.3, 0.0, 0.1])
        randomized = dr.randomize_position(pos)
        # Should be different but close
        assert not np.array_equal(pos, randomized)
        assert np.allclose(pos, randomized, atol=0.05)

    def test_position_disabled_returns_nominal(self) -> None:
        dr = _make_randomizer(enabled=False)
        pos = np.array([0.3, 0.0, 0.1])
        result = dr.randomize_position(pos)
        np.testing.assert_array_equal(pos, result)

    def test_lighting_clamped_to_0_1(self) -> None:
        dr = _make_randomizer(seed=42)
        for _ in range(100):
            lit = dr.randomize_lighting(0.5)
            assert 0.0 <= lit <= 1.0

    def test_camera_pose_noise_applied(self) -> None:
        dr = _make_randomizer(seed=42)
        angles = np.array([0.0, 0.0, 0.0])
        randomized = dr.randomize_camera_pose(angles)
        assert not np.array_equal(angles, randomized)

    def test_apply_all(self) -> None:
        dr = _make_randomizer(seed=42)
        params = {
            "mass": 1.0,
            "friction": 0.5,
            "position": np.array([0.3, 0.0, 0.1]),
            "lighting": 0.5,
            "camera_angles": np.array([0.0, 0.0, 0.0]),
        }
        result = dr.apply_all(params)
        assert result["mass"] != 1.0  # randomized
        assert "friction" in result
        assert "position" in result

    def test_deterministic_with_same_seed(self) -> None:
        dr1 = _make_randomizer(seed=123)
        dr2 = _make_randomizer(seed=123)
        assert dr1.randomize_mass(1.0) == dr2.randomize_mass(1.0)
