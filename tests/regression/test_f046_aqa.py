"""AQA — F-046 RSSM physics-backend gate."""

from __future__ import annotations

import inspect

from mousedroid.sim.protocols import ROVER_RSSM_PHYSICS_BACKENDS
from mousedroid.training.pipeline_orchestrator import PipelineOrchestrator


def test_physics_backends_are_mujoco_and_isaac_lab() -> None:
    assert frozenset({"mujoco", "isaac_lab"}) == ROVER_RSSM_PHYSICS_BACKENDS
    assert "mock" not in ROVER_RSSM_PHYSICS_BACKENDS


def test_vision_finetune_source_stays_mujoco_only() -> None:
    src = inspect.getsource(PipelineOrchestrator._run_vision_finetune)
    assert 'backend != "mujoco"' in src
    assert "isaac_lab" not in src
