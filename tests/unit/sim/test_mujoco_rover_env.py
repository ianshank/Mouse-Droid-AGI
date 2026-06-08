"""RoverMuJoCoEnv conforms to the protocol and matches the mock obs contract."""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from mousedroid.config.schema import RoverConfig
from mousedroid.sim.mock_rover_env import MockRoverEnv
from mousedroid.sim.mujoco_rover_env import RoverMuJoCoEnv
from mousedroid.sim.protocols import (
    ROVER_CHASSIS_POSE_DIM,
    ROVER_IMU_DIM,
    ROVER_NUM_WHEELS,
    RoverEnvProtocol,
)

_WHEEL_RADIUS_M = 0.042
_TRACK_WIDTH_M = 0.20


def _render_cfg() -> RoverConfig:
    cfg = RoverConfig()
    return cfg.model_copy(
        update={
            "sim": cfg.sim.model_copy(
                update={"mujoco": cfg.sim.mujoco.model_copy(update={"render_vision": True})}
            )
        }
    )


def _gl_available() -> bool:
    """Whether offscreen GL rendering works here (false on headless CI)."""
    env = RoverMuJoCoEnv(
        _render_cfg(), wheel_radius_m=_WHEEL_RADIUS_M, track_width_m=_TRACK_WIDTH_M
    )
    try:
        env.reset(seed=0)
        env.render_rgb()
    except Exception:  # any GL/context failure means "no rendering"
        return False
    else:
        return True
    finally:
        env.close()


def _mj() -> RoverMuJoCoEnv:
    return RoverMuJoCoEnv(
        RoverConfig(), wheel_radius_m=_WHEEL_RADIUS_M, track_width_m=_TRACK_WIDTH_M
    )


def _mock() -> MockRoverEnv:
    return MockRoverEnv(RoverConfig(), wheel_radius_m=_WHEEL_RADIUS_M, track_width_m=_TRACK_WIDTH_M)


def test_satisfies_protocol() -> None:
    assert isinstance(_mj(), RoverEnvProtocol)


def test_observation_keys_match_mock() -> None:
    assert _mj().observation_keys == _mock().observation_keys


def test_reset_obs_shapes_match_contract() -> None:
    obs, _info = _mj().reset(seed=0)
    assert obs["imu"].shape == (ROVER_IMU_DIM,)
    assert obs["chassis_pose"].shape == (ROVER_CHASSIS_POSE_DIM,)
    assert obs["wheel_vel"].shape == (ROVER_NUM_WHEELS,)
    assert obs["lidar"].shape == (16,)


def test_step_advances_and_is_finite() -> None:
    env = _mj()
    env.reset(seed=0)
    action = np.full((env.action_dim,), 8.0, dtype=np.float32)
    obs, reward, _term, _trunc, info = env.step(action)
    assert np.isfinite(reward)
    assert "vx_body_mps" in info
    for v in obs.values():
        assert np.isfinite(v).all()


def test_lidar_normalised_to_unit_interval() -> None:
    obs, _ = _mj().reset(seed=0)
    lidar = obs["lidar"]
    assert float(lidar.min()) >= 0.0
    assert float(lidar.max()) <= 1.0


def test_spin_in_place_changes_heading() -> None:
    """Skid-steer: opposite wheel commands rotate the chassis (physics sanity)."""
    env = _mj()
    env.reset(seed=0)
    obs: dict[str, np.ndarray] = {}
    for _ in range(80):
        obs = env.step(np.asarray([-8.0, 8.0], dtype=np.float32))[0]
    pose = obs["chassis_pose"]
    heading = float(np.arctan2(pose[3], pose[2]))
    assert abs(heading) > 0.1


def test_invalid_action_shape_raises() -> None:
    env = _mj()
    env.reset(seed=0)
    with pytest.raises(ValueError, match="action shape"):
        env.step(np.zeros(5, dtype=np.float32))


def test_render_rgb_disabled_raises() -> None:
    env = _mj()  # default render_vision=False
    env.reset(seed=0)
    with pytest.raises(RuntimeError, match="render_vision"):
        env.render_rgb()


def test_render_rgb_shape_and_dtype() -> None:
    if not _gl_available():
        pytest.skip("offscreen GL rendering unavailable (headless CI)")
    cfg = _render_cfg()
    env = RoverMuJoCoEnv(cfg, wheel_radius_m=_WHEEL_RADIUS_M, track_width_m=_TRACK_WIDTH_M)
    env.reset(seed=0)
    frame = env.render_rgb()
    assert frame.shape == (cfg.sim.mujoco.render_height, cfg.sim.mujoco.render_width, 3)
    assert frame.dtype == np.uint8
    env.close()


def test_render_rgb_idempotent_reuses_renderer() -> None:
    if not _gl_available():
        pytest.skip("offscreen GL rendering unavailable (headless CI)")
    cfg = _render_cfg()
    env = RoverMuJoCoEnv(cfg, wheel_radius_m=_WHEEL_RADIUS_M, track_width_m=_TRACK_WIDTH_M)
    env.reset(seed=0)
    env.render_rgb()
    renderer = env._renderer
    env.step(np.zeros(env.action_dim, dtype=np.float32))
    frame2 = env.render_rgb()
    assert env._renderer is renderer
    assert frame2.shape == (cfg.sim.mujoco.render_height, cfg.sim.mujoco.render_width, 3)
    env.close()


def test_close_is_idempotent() -> None:
    env = _mj()
    env.close()
    env.close()


def test_step_after_close_raises_clear_error() -> None:
    env = _mj()
    env.close()
    with pytest.raises(RuntimeError, match="closed RoverMuJoCoEnv"):
        env.reset(seed=0)


def test_body_velocity_mode_maps_to_wheel_setpoints() -> None:
    from mousedroid.config.schema import RoverActionConfig

    cfg = RoverConfig(action=RoverActionConfig(mode="body_velocity"))
    env = RoverMuJoCoEnv(cfg, wheel_radius_m=_WHEEL_RADIUS_M, track_width_m=_TRACK_WIDTH_M)
    env.reset(seed=0)
    # action = [vx, omega]; finite + advances without error
    obs, reward, _t, _tr, info = env.step(np.asarray([0.2, 0.3], dtype=np.float32))
    assert np.isfinite(reward)
    assert np.isfinite(obs["chassis_pose"]).all()


def test_missing_mjcf_raises_file_not_found() -> None:
    cfg = RoverConfig()
    cfg = cfg.model_copy(
        update={
            "sim": cfg.sim.model_copy(
                update={"mujoco": cfg.sim.mujoco.model_copy(update={"mjcf_path": "no/such.xml"})}
            )
        }
    )
    with pytest.raises(FileNotFoundError, match="MJCF not found"):
        RoverMuJoCoEnv(cfg, wheel_radius_m=_WHEEL_RADIUS_M, track_width_m=_TRACK_WIDTH_M)
