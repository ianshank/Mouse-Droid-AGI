"""Regression: the F-023 config surface is purely additive.

Guards the CLAUDE.md invariant "New config fields MUST have defaults; existing
YAML files must load unchanged" for the AlayaWorld-adaptation blocks
(``world_model_memory`` on ``Settings``, ``training.drift`` on
``TrainingConfig``). A Settings built without either block must leave them
``None`` (default-off, byte-identical tick path), a legacy YAML without them
must validate, both blocks must round-trip with defaults, the env-var mapping
must reach them, and the F-023 catalog entry must validate against the harness
schema.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import jsonschema
import pytest
import yaml
from pydantic import ValidationError

from mousedroid.config.schema import (
    DriftTrainingConfig,
    Settings,
    WorldModelMemoryConfig,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_settings_without_blocks_is_none() -> None:
    s = Settings.model_validate({"mock_hardware": True})
    assert s.world_model_memory is None
    assert s.training.drift is None


def test_legacy_yaml_loads_without_blocks() -> None:
    legacy_yaml = """
    mock_hardware: true
    platform: mouse_droid
    training:
      batch_size: 16
    """
    s = Settings.model_validate(yaml.safe_load(legacy_yaml))
    assert s.world_model_memory is None
    assert s.training.drift is None
    assert s.training.batch_size == 16


def test_memory_block_round_trips_with_defaults() -> None:
    s = Settings.model_validate({"mock_hardware": True, "world_model_memory": {"enabled": True}})
    assert s.world_model_memory is not None
    assert s.world_model_memory.enabled is True
    assert s.world_model_memory.recent_size == 16
    assert s.world_model_memory.stride == 8
    assert s.world_model_memory.long_ema_alpha == 0.05
    assert s.world_model_memory.blend_weight == 0.1
    assert s.world_model_memory.sink_warmup_ticks == 30
    assert s.world_model_memory.recapture_on_mission is True


def test_drift_block_round_trips_with_defaults() -> None:
    s = Settings.model_validate({"mock_hardware": True, "training": {"drift": {"enabled": True}}})
    assert s.training.drift is not None
    assert s.training.drift.enabled is True
    assert s.training.drift.corruption_prob == 0.5
    assert s.training.drift.max_prefix_frac == 0.5
    assert s.training.drift.recovery_weight == 1.0
    assert s.training.drift.residual_head is True
    assert s.training.drift.eval_context_steps == 8
    assert s.training.drift.eval_horizon == 24
    assert s.training.drift.seed == 42


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("recent_size", 0),
        ("stride", 0),
        ("long_ema_alpha", 0.0),
        ("long_ema_alpha", 1.5),
        ("blend_weight", -0.1),
        ("blend_weight", 1.5),
        ("sink_warmup_ticks", -1),
    ],
)
def test_memory_field_bounds_rejected(field: str, bad: float) -> None:
    with pytest.raises(ValidationError):
        WorldModelMemoryConfig.model_validate({field: bad})


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("corruption_prob", -0.1),
        ("corruption_prob", 1.1),
        ("max_prefix_frac", 0.0),
        ("max_prefix_frac", 1.1),
        ("recovery_weight", -1.0),
        ("eval_context_steps", 0),
        ("eval_horizon", 0),
    ],
)
def test_drift_field_bounds_rejected(field: str, bad: float) -> None:
    with pytest.raises(ValidationError):
        DriftTrainingConfig.model_validate({field: bad})


def test_env_var_mapping_reaches_both_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MOUSEDROID_WORLD_MODEL_MEMORY__ENABLED", "true")
    monkeypatch.setenv("MOUSEDROID_TRAINING__DRIFT__ENABLED", "true")
    s = Settings()
    assert s.world_model_memory is not None
    assert s.world_model_memory.enabled is True
    assert s.training.drift is not None
    assert s.training.drift.enabled is True


def test_shipped_default_yaml_still_parses() -> None:
    """The real shipped default config loads with both new fields defaulting None."""
    data = yaml.safe_load((_REPO_ROOT / "config" / "default.yaml").read_text())
    s = Settings.model_validate(data)
    assert s.world_model_memory is None
    assert s.training.drift is None


def test_f023_catalog_entry_validates_against_harness_schema() -> None:
    catalog = yaml.safe_load((_REPO_ROOT / "features.yaml").read_text())
    schema = json.loads((_REPO_ROOT / "features.schema.json").read_text())
    jsonschema.validate(catalog, schema)
    ids = [f["id"] for f in catalog["features"]]
    assert "F-023" in ids
    f023 = next(f for f in catalog["features"] if f["id"] == "F-023")
    assert f023["depends_on"] == ["F-001"]
    # Status is a lifecycle value, not a schema property -- pinning a literal
    # here made this test fail the moment F-023 closed out. Pin the durable
    # invariant instead: the status is in-vocabulary, and a `done` entry
    # carries a resolvable hex `implemented_in` (never a branch name -- see
    # .claude/skills/openspec-change/SKILL.md and HARNESS_SPEC.md).
    assert f023["status"] in {"todo", "in_progress", "done", "blocked", "deferred"}
    if f023["status"] == "done":
        assert re.fullmatch(r"[0-9a-f]{40}", f023["implemented_in"] or "")
        assert f023["validation_command"].strip()
