"""Sub-second import + module-presence smoke for Phase 5."""

from __future__ import annotations

import importlib


def test_sim_module_imports_without_mujoco() -> None:
    # Importing the module must NOT require the mujoco engine (lazy import).
    mod = importlib.import_module("mousedroid.sim.mujoco_rover_env")
    assert hasattr(mod, "RoverMuJoCoEnv")


def test_factory_exposes_rssm_trainable_and_rover_env() -> None:
    from mousedroid import factory

    assert hasattr(factory, "build_rssm_trainable")
    assert hasattr(factory, "build_rover_env")


def test_factory_exposes_vision_finetune_surface() -> None:
    from mousedroid import factory

    assert hasattr(factory, "build_vision_feature_extractor")
    assert hasattr(factory, "build_rssm_vision_finetune")
