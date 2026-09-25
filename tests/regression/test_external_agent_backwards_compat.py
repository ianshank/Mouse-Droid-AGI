"""Regression tests — external agent integration backwards compatibility.

Pins the default values of extended dataclasses and new config sections
so that existing YAML loads byte-identical after a ``git pull``.
"""

from __future__ import annotations

from pydantic import SecretStr

from mousedroid.config.schema.agents import AgentsConfig, HonchoConfig
from mousedroid.config.schema.root import Settings
from mousedroid.harness.journal.protocol import JournalEntry
from mousedroid.llm_gateway.mission_parser import IntentType, MissionIntent


def test_mission_intent_existing_fields_unchanged() -> None:
    """MissionIntent() with no args preserves all pre-change defaults."""
    intent = MissionIntent()
    assert intent.intent_type is IntentType.UNKNOWN
    assert intent.confidence == 0.0
    assert intent.raw_command == ""
    assert intent.parameters == {}


def test_mission_intent_new_fields_have_defaults() -> None:
    """New ADK-integration fields default to safe no-op values."""
    intent = MissionIntent()
    assert intent.sub_tasks == ()
    assert intent.agent_source == "local"


def test_journal_entry_existing_fields_unchanged() -> None:
    """JournalEntry() with no args preserves all pre-change defaults."""
    entry = JournalEntry()
    assert entry.task_id is None
    assert entry.phase == ""
    assert entry.event == ""
    assert entry.payload == {}
    assert entry.agent_id is None


def test_journal_entry_new_fields_have_defaults() -> None:
    """New journal fields default to backwards-compatible values."""
    entry = JournalEntry()
    assert entry.category == "harness"
    assert entry.severity == "info"


def test_agents_config_defaults() -> None:
    """AgentsConfig() defaults all frameworks to disabled."""
    cfg = AgentsConfig()
    assert cfg.adk.enabled is False
    assert cfg.honcho.enabled is False
    assert cfg.composio.enabled is False


def test_settings_loads_without_agents_section() -> None:
    """Settings(mock_hardware=True) loads with no agents section."""
    settings = Settings(mock_hardware=True)
    assert settings.agents is None  # None default


def test_honcho_api_key_is_secret_str() -> None:
    """HonchoConfig.api_key is SecretStr; repr never leaks the value."""
    cfg = HonchoConfig(api_key="super_secret_key")
    assert isinstance(cfg.api_key, SecretStr)
    assert "super_secret_key" not in repr(cfg)
