"""AQA — F-045 Isaac apply_domain_params seam."""

from __future__ import annotations

import inspect
from pathlib import Path

from mousedroid.sim.isaaclab.randomization import apply_isaac_domain_params
from mousedroid.sim.isaaclab.rover_env import RoverIsaacLabEnv

_RANDOMIZATION = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "mousedroid"
    / "sim"
    / "isaaclab"
    / "randomization.py"
)


def test_apply_domain_params_signature_matches_mujoco() -> None:
    sig = inspect.signature(RoverIsaacLabEnv.apply_domain_params)
    assert "friction" in sig.parameters
    assert "slip" in sig.parameters
    assert "mass_kg" in sig.parameters
    assert "motor_gain" in sig.parameters


def test_translator_is_documented() -> None:
    doc = inspect.getdoc(apply_isaac_domain_params)
    assert doc
    assert len(doc) > 20


def test_replicator_helper_is_present_check_only() -> None:
    src = _RANDOMIZATION.read_text(encoding="utf-8")
    helper = src[src.index("def _try_replicator_write") :]
    assert "modify_pose" in helper
    assert "does not write" in helper
    assert "writer(" not in helper
