"""Regression: BaselinesConfig and OpenClawPolicyConfig defaults are pinned.

These tests ensure that new fields added to the config schema always carry
backwards-compatible defaults. If a field's default changes, these tests
will fail and force explicit acknowledgement.
"""

from __future__ import annotations

from mousedroid.config.schema import BaselinesConfig, OpenClawPolicyConfig


def test_baselines_config_default_max_memory_query_latency_ms() -> None:
    """max_memory_query_latency_ms default is 150.0."""
    cfg = BaselinesConfig()
    assert cfg.max_memory_query_latency_ms == 150.0


def test_openclaw_policy_config_default_max_skills_per_mission() -> None:
    """max_skills_per_mission default is 5."""
    cfg = OpenClawPolicyConfig()
    assert cfg.max_skills_per_mission == 5


def test_openclaw_policy_config_default_allow_actuation() -> None:
    """allow_actuation defaults to True."""
    cfg = OpenClawPolicyConfig()
    assert cfg.allow_actuation is True


def test_openclaw_policy_config_default_actuation_skill_names() -> None:
    """actuation_skill_names defaults to the 4-element tuple."""
    cfg = OpenClawPolicyConfig()
    assert cfg.actuation_skill_names == ("move", "arm", "drive", "actuate")


def test_openclaw_policy_config_default_max_tracked_missions() -> None:
    """max_tracked_missions defaults to 1000."""
    cfg = OpenClawPolicyConfig()
    assert cfg.max_tracked_missions == 1000


def test_openclaw_policy_config_default_openshell_policy_path() -> None:
    """openshell_policy_path defaults to None."""
    cfg = OpenClawPolicyConfig()
    assert cfg.openshell_policy_path is None
