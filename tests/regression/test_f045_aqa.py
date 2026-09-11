"""AQA — F-045 Isaac apply_domain_params seam."""

from __future__ import annotations

import inspect

from mousedroid.sim.isaaclab.randomization import apply_isaac_domain_params
from mousedroid.sim.isaaclab.rover_env import RoverIsaacLabEnv


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
