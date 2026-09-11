"""Isaac Lab scene-knob helpers that do not import Isaac Lab.

Device selection, USD path derivation, and ``SimulationCfg`` kwargs stay
here so CI can pin the F-043 contract without the ``[isaac]`` extra. Live
construction still happens inside ``RoverIsaacLabEnv._wire_isaaclab_scene``.
"""

from __future__ import annotations

import inspect
from typing import Any

from mousedroid.config.schema import RoverIsaacSimConfig, RoverSimConfig
from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)


def isaac_sim_device(sim: RoverSimConfig) -> str:
    """Return the physics device string matching today's headless branch.

    Args:
        sim: Rover simulation block. ``headless=True`` uses
            ``isaac.device_headless`` (``cuda:0``); GUI uses ``device_gui``
            (``cpu``).

    Returns:
        Device string forwarded to Isaac Lab ``SimulationCfg.device``.
    """
    isaac: RoverIsaacSimConfig = sim.isaac
    if sim.headless:
        return isaac.device_headless
    return isaac.device_gui


def resolve_usd_path(urdf_path: str, usd_path: str | None) -> str:
    """Resolve the USD asset path, defaulting to the URDF-stem ``.usd``.

    Args:
        urdf_path: ``RoverSimConfig.urdf_path``.
        usd_path: Optional override from ``RoverIsaacSimConfig.usd_path``.
            ``None`` keeps the convert_urdf_to_usd.py contract
            (``*.urdf`` → ``*.usd``).

    Returns:
        Filesystem path string for ``UsdFileCfg``.
    """
    if usd_path is not None:
        return usd_path
    return urdf_path.replace(".urdf", ".usd")


def simulation_cfg_kwargs(
    sim_cfg_cls: Any,
    *,
    dt: float,
    device: str,
    num_envs: int,
) -> dict[str, Any]:
    """Build ``SimulationCfg`` kwargs, wiring ``num_envs`` only when supported.

    Args:
        sim_cfg_cls: Isaac Lab ``SimulationCfg`` (or a duck-typed test double).
        dt: Physics step (s).
        device: Physics device string.
        num_envs: Parallel env count from ``RoverSimConfig.num_envs``.

    Returns:
        Keyword arguments safe to splat into ``sim_cfg_cls(...)``. When the
        class has no ``num_envs`` parameter, the count is logged and omitted
        so a live Isaac API mismatch cannot TypeError the training loop.
    """
    kwargs: dict[str, Any] = {"dt": dt, "device": device}
    try:
        params = inspect.signature(sim_cfg_cls).parameters
    except (TypeError, ValueError):
        _log.info("isaac_lab_num_envs_not_on_simulation_cfg", num_envs=num_envs)
        return kwargs
    if "num_envs" in params:
        kwargs["num_envs"] = num_envs
    else:
        _log.info("isaac_lab_num_envs_not_on_simulation_cfg", num_envs=num_envs)
    return kwargs
