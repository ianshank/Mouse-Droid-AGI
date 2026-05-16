"""Smoke test for ``scripts/convert_urdf_to_usd.py``.

Test surface:

1. **Cheap tests run on every host** (no Isaac Sim required). They
   exercise the script's argparse defaults, module loadability, and
   the no-magic-numbers contract on conversion parameters. They do NOT
   call ``pytest.importorskip("isaaclab")`` because the production code
   path lazy-imports Isaac Lab inside :func:`convert` — module load
   itself never touches Omniverse.
2. **Slow end-to-end test** (``@pytest.mark.slow``) calls
   ``pytest.importorskip("isaaclab")`` and ``pytest.importorskip("pxr")``
   at the **start of the test method body** (not at module scope), then
   runs the full conversion against the committed URDF and asserts the
   produced ``.usd`` opens via ``pxr.Usd.Stage.Open`` and retains the
   wheel + sensor prims downstream Phase B wiring depends on.

The importorskip placement is method-scope deliberately: module-scope
``importorskip`` would short-circuit ALL tests in this file, including
the cheap ones that don't need Isaac Sim. CI runners without Isaac
Lab need the cheap tests to still execute (they catch regressions in
the argparse + docstring + module-load contract).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]
_URDF_DEFAULT_RELATIVE = Path("assets/rover/mse6_4wd.urdf")
_USD_DEFAULT_RELATIVE = Path("assets/rover/mse6_4wd.usd")
_URDF_PATH = _REPO_ROOT / _URDF_DEFAULT_RELATIVE


def _load_convert_module() -> Any:
    """Import ``scripts/convert_urdf_to_usd.py`` as a module."""
    script_path = _REPO_ROOT / "scripts" / "convert_urdf_to_usd.py"
    spec = importlib.util.spec_from_file_location("convert_urdf_to_usd", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["convert_urdf_to_usd"] = module
    spec.loader.exec_module(module)
    return module


class TestConvertScriptLoadability:
    """Cheap tests that run on every host (no isaaclab needed)."""

    def test_script_module_loads(self) -> None:
        """The script's module-level imports do not require isaaclab.

        Catches the regression where ``from isaaclab.sim.converters
        import UrdfConverter`` accidentally moves to module scope —
        that would break every CI runner without Isaac Sim.
        """
        module = _load_convert_module()
        assert hasattr(module, "convert")
        assert hasattr(module, "_parse_args")
        assert hasattr(module, "main")

    def test_parse_args_defaults_align_with_rover_urdf(self) -> None:
        """Defaults point at the committed rover URDF/USD — full paths, not basenames.

        Basename-only checks would silently accept a default change like
        ``assets/some_other_dir/mse6_4wd.urdf`` even though the CLI help
        and ADR-009 documentation explicitly call out ``assets/rover/``.
        """
        module = _load_convert_module()
        args = module._parse_args([])
        # Compare path components (OS-agnostic — handles Windows \ vs Linux /).
        assert (
            Path(args.urdf) == _URDF_DEFAULT_RELATIVE
        ), f"--urdf default drifted from {_URDF_DEFAULT_RELATIVE}: got {args.urdf}"
        assert (
            Path(args.output) == _USD_DEFAULT_RELATIVE
        ), f"--output default drifted from {_USD_DEFAULT_RELATIVE}: got {args.output}"

    def test_default_conversion_params_preserve_sensor_frames(self) -> None:
        """Defaults must NOT merge fixed joints — sensor frames would dissolve.

        The MSE-6 URDF wires imu_link/lidar_link/camera_link as
        fixed-jointed children of base_link. The Isaac Lab default
        UrdfConverterCfg.merge_fixed_joints=True would collapse those
        into base_link, removing the sensor prims that
        :data:`ROVER_SENSOR_LINK_NAMES` references for B3 Story 2
        sensor attachment.

        This regression net pins the safe defaults — a future drive-by
        flip to ``merge_fixed_joints=True`` will trip the test and
        force the author to update either the constants module's sensor
        attachment contract or the script default in lockstep.
        """
        module = _load_convert_module()
        args = module._parse_args([])
        assert args.fix_base is False
        # CRITICAL: must remain False to preserve sensor link prims.
        assert args.merge_fixed_joints is False, (
            "Default --merge-fixed-joints=False is required to preserve "
            "imu_link/lidar_link/camera_link in the converted USD. See "
            "ADR-009 sensor-attachment contract."
        )
        assert args.convex_decompose_mesh is True
        assert args.self_collision is False


@pytest.mark.slow
class TestConvertScriptEndToEnd:
    """Actual URDF -> USD conversion — requires isaaclab + Isaac Sim.

    Each test calls ``pytest.importorskip`` inside the method body so
    the slow tests skip cleanly on hosts without Isaac Sim while the
    cheap tests in :class:`TestConvertScriptLoadability` still execute.
    """

    def test_conversion_produces_valid_usd(self, tmp_path: Path) -> None:
        """Run the full conversion and assert the .usd opens via pxr."""
        pytest.importorskip("isaaclab")
        pytest.importorskip("pxr")
        from pxr import Usd

        module = _load_convert_module()
        out = tmp_path / "mse6_4wd.usd"
        produced = module.convert(urdf_path=_URDF_PATH, output_path=out)

        # Use the converter's reported path (may differ from `out` if
        # Isaac Lab normalises directory separators or strips extensions).
        assert Path(produced).is_file()
        stage = Usd.Stage.Open(str(produced))
        assert stage is not None
        # Verify the rover root prim is present.
        rover_prim = stage.GetPrimAtPath("/mse6_4wd")
        assert rover_prim.IsValid()

    def test_converted_usd_retains_wheel_and_sensor_prims(self, tmp_path: Path) -> None:
        """The default conversion must preserve the prims Phase B sensors attach to.

        Catches the failure mode where a future default change merges
        fixed sensor joints into base_link, dissolving the
        imu_link/lidar_link/camera_link prims downstream Isaac Lab
        wiring depends on. Without this assertion, the smoke test stays
        green on a converter that produces a structurally-correct USD
        but with no sensor frames to attach IMUSensorCfg/RayCasterCfg/
        CameraSensorCfg to.
        """
        pytest.importorskip("isaaclab")
        pytest.importorskip("pxr")
        from pxr import Usd

        from mousedroid.sim.isaaclab.constants import (
            ROVER_SENSOR_LINK_NAMES,
            ROVER_WHEEL_JOINT_NAMES,
        )

        module = _load_convert_module()
        out = tmp_path / "mse6_4wd.usd"
        produced = module.convert(urdf_path=_URDF_PATH, output_path=out)
        stage = Usd.Stage.Open(str(produced))
        assert stage is not None

        # Wheel joints: must exist somewhere under the rover prim's
        # subtree. Isaac Lab places URDF continuous joints under the
        # articulation root; exact PrimPath depends on converter
        # version, so we scan the stage for matching prim names.
        all_prim_names = {prim.GetName() for prim in stage.Traverse()}
        for joint_name in ROVER_WHEEL_JOINT_NAMES:
            assert joint_name in all_prim_names, (
                f"Converted USD missing wheel-joint prim {joint_name!r}. "
                f"Found names: {sorted(all_prim_names)}"
            )

        # Sensor link prims: required for B3 Story 2 sensor attachment.
        # The fix here is the default --merge-fixed-joints=False; if a
        # future change re-enables merging, these prims vanish and this
        # test fails loudly.
        for link_name in ROVER_SENSOR_LINK_NAMES:
            assert link_name in all_prim_names, (
                f"Converted USD missing sensor-link prim {link_name!r}. "
                f"Default --merge-fixed-joints must be False to preserve "
                f"the fixed-jointed sensor frames. Found names: "
                f"{sorted(all_prim_names)}"
            )
