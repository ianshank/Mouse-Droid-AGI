"""Smoke test for ``scripts/convert_urdf_to_usd.py``.

The conversion itself requires Isaac Lab + Isaac Sim — those are gated
on ``pytest.importorskip("isaaclab")``. This file's purpose is twofold:

1. **Without Isaac Lab installed** (the default CI runner): verify the
   script's argparse / docstring / module load is clean. The
   ``importorskip`` short-circuits at module scope so the test cleanly
   skips end-to-end conversion.
2. **With Isaac Lab installed** (operator workstation): run the actual
   conversion against the committed URDF and assert the produced
   ``.usd`` opens via ``pxr.Usd.Stage.Open``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]
_URDF_PATH = _REPO_ROOT / "assets" / "rover" / "mse6_4wd.urdf"


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
        module = _load_convert_module()
        args = module._parse_args([])
        assert Path(args.urdf).name == "mse6_4wd.urdf"
        assert Path(args.output).name == "mse6_4wd.usd"

    def test_default_conversion_params_match_documented_values(self) -> None:
        """Defaults match the Isaac Lab tutorial for wheeled mobile robots.

        Regression net — a future drive-by change to defaults will trip
        this test, forcing the author to update the docstring + ADR-009
        and the operator runbook in lockstep.
        """
        module = _load_convert_module()
        args = module._parse_args([])
        assert args.fix_base is False
        assert args.merge_fixed_joints is True
        assert args.convex_decompose_mesh is True
        assert args.self_collision is False


@pytest.mark.slow
class TestConvertScriptEndToEnd:
    """Actual URDF -> USD conversion — requires isaaclab + Isaac Sim."""

    def test_conversion_produces_valid_usd(self, tmp_path: Path) -> None:
        """Run the full conversion and assert the .usd opens via pxr.

        Skipped on hosts without ``isaaclab`` (the converter dep) or
        ``pxr`` (USD bindings). Both are part of Isaac Sim.
        """
        pytest.importorskip("isaaclab")
        pytest.importorskip("pxr")
        from pxr import Usd

        module = _load_convert_module()
        out = tmp_path / "mse6_4wd.usd"
        module.convert(urdf_path=_URDF_PATH, output_path=out)

        assert out.is_file()
        stage = Usd.Stage.Open(str(out))
        assert stage is not None
        # Verify the rover root prim is present.
        rover_prim = stage.GetPrimAtPath("/mse6_4wd")
        assert rover_prim.IsValid()
