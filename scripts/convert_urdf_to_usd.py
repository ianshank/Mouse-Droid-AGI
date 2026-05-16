"""Convert ``assets/rover/mse6_4wd.urdf`` to a USD scene Isaac Lab can load.

Tier B Track B3 Story 1 — runs **once** on a workstation with Isaac Sim
+ Isaac Lab installed. The resulting ``.usd`` is committed under
``assets/rover/`` so subsequent CI runs and downstream B3 stories don't
have to re-run the conversion.

Why this script lives separate from ``isaaclab/`` source:

* Module-level imports of ``isaaclab.sim.converters`` pull in the full
  Omniverse stack and would explode every ``import mousedroid.*`` on
  hosts without Isaac Sim. The script is invoked as a one-off CLI; the
  production code path (``RoverIsaacLabEnv.build``) reads the committed
  ``.usd``, not the URDF.
* Conversion parameters are CLI-overridable so operators can change
  them without touching code. The schema-side ``RoverSimConfig`` does
  NOT carry conversion knobs (it captures the runtime simulation
  parameters); this script's parameters are tooling defaults that the
  operator picks once and re-uses for every regeneration.

CLI usage::

    python scripts/convert_urdf_to_usd.py \\
        --urdf assets/rover/mse6_4wd.urdf \\
        --output assets/rover/mse6_4wd.usd

Requirements
------------
* NVIDIA Isaac Sim (free for individual research)
* NVIDIA Omniverse Launcher account
* NVIDIA GPU with RTX cores (RTX 2070+ recommended)
* Linux (Ubuntu 22.04 — Isaac Sim does not support native Windows
  simulation streaming)
* ``pip install -e ".[isaac]"``

Output
------
A binary ``.usd`` ready to load via ``UsdFileCfg(usd_path=...)`` in
:class:`ArticulationCfg`. Committed to the repo (single-file, ~few MB,
rarely changes).

The conversion is deterministic given the same URDF + parameters;
operators rerun this script after any URDF change to keep the ``.usd``
in sync. The script forces regeneration when the output already exists
(``force_usd_conversion=True``) so a stale ``.usd`` cannot survive a
URDF edit.

Sensor-frame preservation
-------------------------
The rover URDF wires IMU / LiDAR / camera as **fixed-jointed children**
of ``base_link``. The default :class:`UrdfConverterCfg.merge_fixed_joints`
collapses fixed joints into their parent, which would dissolve the
``imu_link``, ``lidar_link``, and ``camera_link`` prims that
:data:`~mousedroid.sim.isaaclab.constants.ROVER_SENSOR_LINK_NAMES`
references for sensor attachment in B3 Story 2. We therefore default
``merge_fixed_joints=False`` so those frames survive into the produced
USD. The smoke test in ``test_urdf_to_usd.py`` asserts the sensor and
wheel prims are present in the converted asset to catch regressions.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

# Conversion parameters — chosen for the MSE-6 wheeled rover. Operators can
# override via CLI if needed.
_DEFAULT_FIX_BASE: bool = False  # rover is mobile, base is free-floating
# DO NOT default to True — the URDF's fixed sensor joints would be merged into
# base_link, dissolving the imu_link/lidar_link/camera_link prims that
# ROVER_SENSOR_LINK_NAMES references for B3 Story 2 sensor attachment.
_DEFAULT_MERGE_FIXED_JOINTS: bool = False
_DEFAULT_CONVEX_DECOMPOSE_MESH: bool = True
_DEFAULT_SELF_COLLISION: bool = False


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--urdf",
        type=Path,
        default=Path("assets/rover/mse6_4wd.urdf"),
        help="Source URDF path (default: assets/rover/mse6_4wd.urdf)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets/rover/mse6_4wd.usd"),
        help="Destination .usd path (default: assets/rover/mse6_4wd.usd)",
    )
    parser.add_argument(
        "--fix-base",
        action="store_true",
        default=_DEFAULT_FIX_BASE,
        help="Pin the chassis base to world (default: %(default)s)",
    )
    parser.add_argument(
        "--merge-fixed-joints",
        action="store_true",
        default=_DEFAULT_MERGE_FIXED_JOINTS,
        help=(
            "Collapse URDF fixed joints into their parents. Default is "
            "FALSE because the MSE-6 URDF attaches sensors via fixed "
            "joints; merging would dissolve imu_link/lidar_link/camera_link "
            "prims that downstream Isaac Lab wiring depends on."
        ),
    )
    parser.add_argument(
        "--no-convex-decompose",
        action="store_false",
        dest="convex_decompose_mesh",
        default=_DEFAULT_CONVEX_DECOMPOSE_MESH,
        help="Skip convex decomposition of mesh collision shapes",
    )
    parser.add_argument(
        "--self-collision",
        action="store_true",
        default=_DEFAULT_SELF_COLLISION,
        help="Enable self-collision (slower; rarely needed for wheeled robots)",
    )
    return parser.parse_args(argv)


def _launch_isaac_app() -> Any:
    """Initialise the Isaac Sim application required by ``isaaclab.sim.converters``.

    Isaac Lab's converter APIs depend on a running Kit / Omniverse
    context. Without an :class:`isaaclab.app.AppLauncher` instance,
    importing ``isaaclab.sim.converters`` succeeds but the subsequent
    converter call raises a Kit-not-running error. We launch a headless
    app, run the conversion, and close the app from ``main`` so the
    standalone CLI invocation works on an operator workstation.

    Returns:
        The live ``SimulationApp`` instance — caller is responsible for
        calling ``.close()`` once the conversion finishes.
    """
    # Lazy import — keeps the module loadable on hosts without Isaac Sim
    # for the smoke tests that exercise argparse defaults / docstrings.
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=True)
    return launcher.app


def convert(
    *,
    urdf_path: Path,
    output_path: Path,
    fix_base: bool = _DEFAULT_FIX_BASE,
    merge_fixed_joints: bool = _DEFAULT_MERGE_FIXED_JOINTS,
    convex_decompose_mesh: bool = _DEFAULT_CONVEX_DECOMPOSE_MESH,
    self_collision: bool = _DEFAULT_SELF_COLLISION,
) -> Path:
    """Run the URDF -> USD conversion via Isaac Lab's UrdfConverter.

    Per the Isaac Lab 0.20+ API contract, :class:`UrdfConverter` performs
    the conversion during ``__init__`` and exposes ``converter.usd_path``
    pointing at the produced USD. There is no ``convert_in_place()``
    method — that was the pre-0.20 API.

    Force-regeneration is enabled (``force_usd_conversion=True``) so a
    stale USD on disk cannot mask a URDF edit; rerunning this script
    after a URDF change always produces a fresh artifact.

    Args:
        urdf_path: Source URDF file path.
        output_path: Target .usd file path.
        fix_base: Pin the chassis base.
        merge_fixed_joints: Collapse URDF fixed joints. Default ``False``
            for the MSE-6 rover to preserve sensor frames.
        convex_decompose_mesh: Decompose mesh collisions.
        self_collision: Enable self-collision in the produced USD.

    Returns:
        The path Isaac Lab's converter wrote the USD to. This is
        normally ``output_path`` but the converter may normalise
        directory separators or extensions — the caller should trust the
        returned value for downstream operations.

    Raises:
        ImportError: If ``isaaclab`` is not installed.
        FileNotFoundError: If ``urdf_path`` does not exist.
    """
    if not urdf_path.is_file():
        msg = f"URDF source not found: {urdf_path}"
        raise FileNotFoundError(msg)

    # Lazy import — keeps the module loadable on hosts without Isaac Lab
    # for the smoke tests (which use `importorskip("isaaclab")`).
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = UrdfConverterCfg(
        asset_path=str(urdf_path),
        usd_dir=str(output_path.parent),
        usd_file_name=output_path.name,
        fix_base=fix_base,
        merge_fixed_joints=merge_fixed_joints,
        convex_decompose_mesh=convex_decompose_mesh,
        self_collision=self_collision,
        # Force regeneration. Isaac Lab caches converted USDs and
        # otherwise reuses a stale .usd silently after a URDF edit.
        force_usd_conversion=True,
    )
    # The converter runs during __init__; usd_path is the canonical
    # destination after conversion completes.
    converter = UrdfConverter(cfg)
    produced = Path(converter.usd_path)
    return produced


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)
    args = _parse_args(argv)
    logger.info("urdf_to_usd_started urdf=%s output=%s", args.urdf, args.output)

    # Isaac Lab requires a running Kit context before any converter call.
    # Launch the app, do the conversion, then close.
    simulation_app = _launch_isaac_app()
    try:
        produced = convert(
            urdf_path=args.urdf,
            output_path=args.output,
            fix_base=args.fix_base,
            merge_fixed_joints=args.merge_fixed_joints,
            convex_decompose_mesh=args.convex_decompose_mesh,
            self_collision=args.self_collision,
        )
        logger.info(
            "urdf_to_usd_finished output=%s bytes=%d",
            produced,
            produced.stat().st_size,
        )
    finally:
        simulation_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
