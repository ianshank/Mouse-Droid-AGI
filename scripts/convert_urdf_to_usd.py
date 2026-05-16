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
* Conversion parameters (``fix_base``, ``merge_fixed_joints``,
  ``convex_decompose_mesh``, ``self_collision``) come from
  :class:`RoverSimConfig` / sensible defaults, never magic numbers.

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
in sync.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Conversion parameters — defaults match the Isaac Lab tutorial for
# wheeled mobile robots. Operators can override via CLI if needed.
_DEFAULT_FIX_BASE: bool = False  # rover is mobile, base is free-floating
_DEFAULT_MERGE_FIXED_JOINTS: bool = True
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
        help="Pin the chassis base to world (overrides default: %(default)s)",
    )
    parser.add_argument(
        "--no-merge-fixed-joints",
        action="store_false",
        dest="merge_fixed_joints",
        default=_DEFAULT_MERGE_FIXED_JOINTS,
        help="Skip merging URDF fixed joints into parent bodies",
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


def convert(
    *,
    urdf_path: Path,
    output_path: Path,
    fix_base: bool = _DEFAULT_FIX_BASE,
    merge_fixed_joints: bool = _DEFAULT_MERGE_FIXED_JOINTS,
    convex_decompose_mesh: bool = _DEFAULT_CONVEX_DECOMPOSE_MESH,
    self_collision: bool = _DEFAULT_SELF_COLLISION,
) -> None:
    """Run the URDF -> USD conversion via Isaac Lab's UrdfConverter.

    The conversion is performed in-place (``UrdfConverter.convert_in_place``)
    so the destination directory is the same as the source by convention.
    The caller asserts the output path matches before the import.

    Args:
        urdf_path: Source URDF file path.
        output_path: Target .usd file path.
        fix_base: Pin the chassis base.
        merge_fixed_joints: Collapse URDF fixed joints.
        convex_decompose_mesh: Decompose mesh collisions.
        self_collision: Enable self-collision in the produced USD.

    Raises:
        ImportError: If ``isaaclab`` is not installed.
        FileNotFoundError: If ``urdf_path`` does not exist.
    """
    if not urdf_path.is_file():
        msg = f"URDF source not found: {urdf_path}"
        raise FileNotFoundError(msg)

    # Lazy import — keeps the module loadable on hosts without Isaac Lab
    # for the smoke test (which uses `importorskip("isaaclab")`).
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
    )
    UrdfConverter(cfg).convert_in_place()


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)
    args = _parse_args(argv)
    logger.info("urdf_to_usd_started urdf=%s output=%s", args.urdf, args.output)
    convert(
        urdf_path=args.urdf,
        output_path=args.output,
        fix_base=args.fix_base,
        merge_fixed_joints=args.merge_fixed_joints,
        convex_decompose_mesh=args.convex_decompose_mesh,
        self_collision=args.self_collision,
    )
    logger.info("urdf_to_usd_finished output=%s bytes=%d", args.output, args.output.stat().st_size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
