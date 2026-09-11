"""Unit tests for Isaac Lab scene helpers (no isaaclab import)."""

from __future__ import annotations

from mousedroid.config.schema import RoverIsaacSimConfig, RoverSimConfig
from mousedroid.sim.isaaclab.scene import (
    isaac_sim_device,
    resolve_usd_path,
    simulation_cfg_kwargs,
)


def test_device_preserves_headless_branch() -> None:
    headless = RoverSimConfig(headless=True)
    gui = RoverSimConfig(headless=False)
    assert isaac_sim_device(headless) == "cuda:0"
    assert isaac_sim_device(gui) == "cpu"


def test_device_honours_nested_overrides() -> None:
    sim = RoverSimConfig(
        headless=True,
        isaac=RoverIsaacSimConfig(device_headless="cuda:1", device_gui="cpu"),
    )
    assert isaac_sim_device(sim) == "cuda:1"
    sim_gui = RoverSimConfig(
        headless=False,
        isaac=RoverIsaacSimConfig(device_headless="cuda:0", device_gui="cpu"),
    )
    assert isaac_sim_device(sim_gui) == "cpu"


def test_usd_path_none_replaces_urdf_suffix() -> None:
    assert resolve_usd_path("assets/rover/mse6_4wd.urdf", None) == "assets/rover/mse6_4wd.usd"


def test_usd_path_override_wins() -> None:
    assert resolve_usd_path("a.urdf", "/tmp/custom.usd") == "/tmp/custom.usd"


def test_usd_path_override_keeps_nonstandard_suffix() -> None:
    """Operator override is trusted; a non-.usd suffix is not jailed."""
    assert resolve_usd_path("a.urdf", "/tmp/custom.usda") == "/tmp/custom.usda"
    assert resolve_usd_path("a.urdf", "/tmp/custom.bin") == "/tmp/custom.bin"


class _CfgWithNumEnvs:
    def __init__(self, dt: float, device: str, num_envs: int = 1) -> None:
        self.dt = dt
        self.device = device
        self.num_envs = num_envs


class _CfgWithoutNumEnvs:
    def __init__(self, dt: float, device: str) -> None:
        self.dt = dt
        self.device = device


def test_simulation_cfg_kwargs_includes_num_envs_when_supported() -> None:
    kwargs = simulation_cfg_kwargs(_CfgWithNumEnvs, dt=0.008, device="cpu", num_envs=4)
    assert kwargs["num_envs"] == 4
    assert kwargs["dt"] == 0.008
    assert kwargs["device"] == "cpu"


def test_simulation_cfg_kwargs_omits_num_envs_when_unsupported() -> None:
    kwargs = simulation_cfg_kwargs(_CfgWithoutNumEnvs, dt=0.008, device="cuda:0", num_envs=8)
    assert "num_envs" not in kwargs
    assert kwargs["device"] == "cuda:0"


def test_simulation_cfg_kwargs_omits_num_envs_without_signature() -> None:
    kwargs = simulation_cfg_kwargs(object(), dt=0.008, device="cpu", num_envs=2)
    assert "num_envs" not in kwargs
    assert kwargs["dt"] == 0.008
