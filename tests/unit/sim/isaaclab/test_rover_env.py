"""Isaac Lab rover env body tests (Tier C4).

The whole module is skipped on CI hosts without ``isaaclab`` via
:func:`pytest.importorskip` — there is no point exercising the live
``build/reset/step`` path without the optional dependency. Operators
validate end-to-end on Linux + Isaac Sim post-merge per ADR-009.

Even when skipped, the file MUST parse cleanly + import its module-
level symbols (``RoverIsaacLabEnv``, ``RoverRewardConfig``,
``ROVER_WHEEL_JOINT_NAMES``) so ``mypy --strict`` exercises the type
surface on every CI run regardless of whether Isaac Lab is installed.
"""

from __future__ import annotations

import numpy as np
import pytest

# Skip the entire module when isaaclab is not importable. The C4 plan
# pins this as the gate; the 9 tests below ONLY run on a host with
# Isaac Lab installed.
isaaclab = pytest.importorskip("isaaclab")

from mousedroid.config.schema import (
    DomainRandomizationConfig,
    RoverConfig,
    RoverRewardConfig,
    RoverSimConfig,
)
from mousedroid.sim.isaaclab.constants import (
    ROVER_NUM_WHEELS,
)
from mousedroid.sim.isaaclab.rover_env import RoverIsaacLabEnv
from mousedroid.sim.mock_rover_env import MockRoverEnv

# Robot dimensions — match the URDF + RobotConfig defaults so the
# wheel-radius / track-width math is byte-identical to MockRoverEnv.
_WHEEL_RADIUS_M = 0.042
_TRACK_WIDTH_M = 0.20


def _make_cfg(*, dr_enabled: bool = False) -> tuple[RoverConfig, DomainRandomizationConfig]:
    """Build a RoverConfig with an explicit reward block (required by C4 build)."""
    return (
        RoverConfig(
            sim=RoverSimConfig(backend="isaac_lab"),
            reward=RoverRewardConfig(),
        ),
        DomainRandomizationConfig(enabled=dr_enabled),
    )


def _make_env(*, dr_enabled: bool = False) -> RoverIsaacLabEnv:
    cfg, dr = _make_cfg(dr_enabled=dr_enabled)
    return RoverIsaacLabEnv(
        cfg,
        wheel_radius_m=_WHEEL_RADIUS_M,
        track_width_m=_TRACK_WIDTH_M,
        domain_randomization=dr,
    )


class TestRoverIsaacLabEnv:
    """C4 body tests — all gate on ``pytest.importorskip('isaaclab')``."""

    def test_build_succeeds(self) -> None:
        """``build()`` constructs the sim context without raising."""
        env = _make_env()
        env.build()
        assert env.action_dim == 2
        env.close()

    def test_reset_returns_obs_dict_with_expected_keys(self) -> None:
        """``reset()`` emits the configured observation keys."""
        env = _make_env()
        env.build()
        obs, info = env.reset(seed=0)
        assert set(obs.keys()) == set(env.observation_keys)
        assert info["step_idx"] == 0
        env.close()

    def test_reset_observation_excludes_ultrasonic(self) -> None:
        """Rover baseline must not emit HC-SR04 / ultrasonic channels."""
        env = _make_env()
        env.build()
        obs, _ = env.reset(seed=0)
        for key in obs:
            assert "ultrasonic" not in key
            assert "hc_sr04" not in key
            assert "distance" not in key
        env.close()

    def test_step_returns_gymnasium_5_tuple(self) -> None:
        """``step()`` returns ``(obs, reward, terminated, truncated, info)``."""
        env = _make_env()
        env.build()
        env.reset(seed=0)
        result = env.step(np.zeros(2, dtype=np.float32))
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, dict)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "step_idx" in info
        env.close()

    def test_step_clips_action_to_wheel_velocity_bounds(self) -> None:
        """Wheel velocities post-clip respect ``cfg.action.max_wheel_rad_s``."""
        env = _make_env()
        env.build()
        env.reset(seed=0)
        # Over-the-cap action — must be clipped before forwarding.
        oversized = np.array([1e6, -1e6], dtype=np.float32)
        _, _, _, _, info = env.step(oversized)
        cap = env._cfg.action.max_wheel_rad_s
        for v in info["wheel_velocities"]:
            assert -cap <= v <= cap
        env.close()

    def test_step_fans_2d_action_onto_4_wheels(self) -> None:
        """The 2-D action fans onto 4 wheels in [FL, FR, RL, RR] = [L, R, L, R]."""
        env = _make_env()
        env.build()
        env.reset(seed=0)
        action = np.array([0.5, -0.3], dtype=np.float32)
        _, _, _, _, info = env.step(action)
        wheels = info["wheel_velocities"]
        assert len(wheels) == ROVER_NUM_WHEELS
        # Alternating layout per ROVER_WHEEL_JOINT_NAMES.
        # FL == RL == left ; FR == RR == right.
        assert wheels[0] == pytest.approx(wheels[2])  # FL == RL
        assert wheels[1] == pytest.approx(wheels[3])  # FR == RR
        assert wheels[0] == pytest.approx(0.5)
        assert wheels[1] == pytest.approx(-0.3)
        env.close()

    def test_domain_randomization_applied_on_reset_when_enabled(self) -> None:
        """``reset()`` populates ``info['episode_params']`` when DR is on."""
        env = _make_env(dr_enabled=True)
        env.build()
        _, info = env.reset(seed=42)
        assert info["dr_enabled"] is True
        assert "episode_params" in info
        # Chassis sub-mapping is populated from the DR config.
        assert info["episode_params"].chassis  # non-empty
        env.close()

    def test_observation_contract_matches_mock_rover_env(self) -> None:
        """Cross-backend contract: same 2-D action → same wheel layout.

        Pins the ``[FL, FR, RL, RR] = [left, right, left, right]``
        layout that the C4 peer-review surfaced. If the Isaac Lab
        fan-out reorders, this test fires.
        """
        cfg, _ = _make_cfg()
        mock_env = MockRoverEnv(cfg, wheel_radius_m=_WHEEL_RADIUS_M, track_width_m=_TRACK_WIDTH_M)
        mock_env.reset(seed=0)
        action = np.array([0.7, -0.4], dtype=np.float32)
        _, _, _, _, _mock_info = mock_env.step(action)
        # Mock stores per-wheel velocities on its internal state.
        mock_wheels = mock_env._wheel_vel.tolist()

        env = _make_env()
        env.build()
        env.reset(seed=0)
        _, _, _, _, isaac_info = env.step(action)
        isaac_wheels = isaac_info["wheel_velocities"]
        env.close()

        assert isaac_wheels == pytest.approx(mock_wheels)
        # Both backends must agree on the observation key set under
        # the default toggles.
        assert set(env.observation_keys) == set(mock_env.observation_keys)

    def test_random_rollout_produces_finite_observations(self) -> None:
        """50-step random rollout: every obs tensor is finite (no NaN/Inf)."""
        env = _make_env()
        env.build()
        env.reset(seed=0)
        rng = np.random.default_rng(0)
        for _ in range(50):
            action = rng.uniform(-1.0, 1.0, size=(2,)).astype(np.float32)
            obs, _reward, terminated, truncated, _info = env.step(action)
            for key, value in obs.items():
                assert np.all(np.isfinite(value)), f"non-finite at key={key}"
            if terminated or truncated:
                env.reset(seed=0)
        env.close()
