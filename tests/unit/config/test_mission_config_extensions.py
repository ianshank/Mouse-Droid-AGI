"""Tier C2.3: new MissionConfig fields gating VLM head + LLM replanner."""

from __future__ import annotations

import pytest

from mousedroid.config.schema import MissionConfig, MissionReplannerConfig


def test_vlm_progress_enabled_defaults_false_for_backcompat() -> None:
    assert MissionConfig().vlm_progress_enabled is False


def test_vlm_mock_progress_value_default_above_success_threshold() -> None:
    cfg = MissionConfig()
    # Mock value must be >= success_threshold so a smoke-mode mission
    # transitions to SUCCEEDED on the first scored tick (operator quickstart).
    assert cfg.vlm_mock_progress_value >= cfg.success_threshold


def test_llm_replanner_enabled_defaults_false_for_backcompat() -> None:
    assert MissionConfig().llm_replanner_enabled is False


def test_replanner_subconfig_loads_with_defaults() -> None:
    cfg = MissionConfig()
    assert isinstance(cfg.replanner, MissionReplannerConfig)
    assert cfg.replanner.max_prompt_chars == 512
    assert cfg.replanner.include_progress_in_prompt is True


def test_replanner_max_prompt_chars_rejects_zero() -> None:
    with pytest.raises(ValueError, match="max_prompt_chars"):
        MissionReplannerConfig(max_prompt_chars=0)


def test_mission_config_loads_unchanged_for_existing_yaml() -> None:
    """Existing YAML with only ``replan_enabled=true`` still builds cleanly."""
    cfg = MissionConfig(replan_enabled=True)
    assert cfg.replan_enabled is True
    assert cfg.vlm_progress_enabled is False
    assert cfg.llm_replanner_enabled is False
