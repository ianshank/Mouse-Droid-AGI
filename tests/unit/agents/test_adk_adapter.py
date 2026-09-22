"""Unit tests — ADKMissionAdapter.

Tests the Google ADK mission decomposer adapter with mocked SDK,
verifying lazy import, degraded behaviour, injection filtering, and
correct MissionIntent construction.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from mousedroid.config.schema.agents import ADKConfig
from mousedroid.llm_gateway.mission_parser import MissionIntent


@pytest.fixture
def adk_config() -> ADKConfig:
    """ADK config with enabled=True for testing."""
    return ADKConfig(enabled=True, model_name="test-model", timeout_s=5.0)


@pytest.fixture
def mock_injection_filter() -> MagicMock:
    """Injection filter mock that passes through unchanged."""
    filt = MagicMock()
    filt.sanitize.side_effect = lambda x: x
    return filt


@pytest.fixture
def adapter(adk_config: ADKConfig, mock_injection_filter: MagicMock) -> Any:
    """ADKMissionAdapter instance with mocked injection filter."""
    from mousedroid.agents.adk_adapter import ADKMissionAdapter

    return ADKMissionAdapter(adk_config, injection_filter=mock_injection_filter)


@pytest.mark.asyncio
async def test_start_missing_sdk_degrades(adk_config: ADKConfig) -> None:
    """When google.adk is not installed, start() degrades without crash."""
    from unittest.mock import patch

    from mousedroid.agents.adk_adapter import ADKMissionAdapter

    with patch.dict("sys.modules", {"google.adk": None}):  # type: ignore[dict-item]
        adapter = ADKMissionAdapter(adk_config)
        # The SDK is not installed in test env, so start() should degrade
        await adapter.start()
        assert adapter.is_degraded is True
        assert adapter.is_ready is False


@pytest.mark.asyncio
async def test_start_missing_agent_class_degrades(adk_config: ADKConfig) -> None:
    """When google.adk.Agent is unavailable, start() degrades without crash."""
    from unittest.mock import patch

    from mousedroid.agents.adk_adapter import ADKMissionAdapter

    with patch("importlib.import_module", return_value=object()):
        adapter = ADKMissionAdapter(adk_config)
        await adapter.start()
        assert adapter.is_degraded is True
        assert adapter.is_ready is False


@pytest.mark.asyncio
async def test_decompose_degraded_returns_neutral(
    adk_config: ADKConfig,
) -> None:
    """When adapter is degraded, decompose returns a neutral MissionIntent."""
    from mousedroid.agents.adk_adapter import ADKMissionAdapter

    adapter = ADKMissionAdapter(adk_config)
    adapter._degraded = True

    intents = await adapter.decompose("go forward")
    assert len(intents) == 1
    assert isinstance(intents[0], MissionIntent)
    assert intents[0].agent_source == "adk"


@pytest.mark.asyncio
async def test_decompose_empty_command_returns_neutral(
    adapter: Any,
) -> None:
    """Empty command returns neutral MissionIntent (not an error)."""
    intents = await adapter.decompose("")
    assert len(intents) == 1
    assert isinstance(intents[0], MissionIntent)
    assert intents[0].agent_source == "adk"


@pytest.mark.asyncio
async def test_injection_filter_called_before_adk(
    adapter: Any,
    mock_injection_filter: MagicMock,
) -> None:
    """Injection filter sanitize() is called before ADK dispatch."""
    adapter._ready = True
    adapter._degraded = False
    adapter._agent = MagicMock()
    adapter._agent.decompose.return_value = []

    await adapter.decompose("dirty command")
    mock_injection_filter.sanitize.assert_called_once_with("dirty command")


@pytest.mark.asyncio
async def test_is_ready_false_initially(adapter: Any) -> None:
    """Adapter is not ready before start() is called."""
    assert adapter.is_ready is False


@pytest.mark.asyncio
async def test_is_degraded_false_initially(adapter: Any) -> None:
    """Adapter is not degraded before start() is called."""
    assert adapter.is_degraded is False


@pytest.mark.asyncio
async def test_agent_source_field(adapter: Any) -> None:
    """Returned MissionIntents always carry agent_source='adk'."""
    # Force degraded to verify even the fallback path sets agent_source
    adapter._degraded = True
    intents = await adapter.decompose("test command")
    assert all(i.agent_source == "adk" for i in intents)


@pytest.mark.asyncio
async def test_start_success_and_decompose() -> None:
    from unittest.mock import MagicMock, patch

    from mousedroid.agents.adk_adapter import ADKMissionAdapter
    from mousedroid.config.schema.agents import ADKConfig

    mock_adk = MagicMock()
    mock_agent_instance = MagicMock()
    # Mocking what ADK agent returns
    mock_agent_instance.decompose.return_value = [{"intent": "navigate", "parameters": {}}]
    mock_adk.Agent.return_value = mock_agent_instance

    with patch.dict("sys.modules", {"google": MagicMock(), "google.adk": mock_adk}):
        cfg = ADKConfig(enabled=True, model_name="test")
        adapter = ADKMissionAdapter(cfg)

        await adapter.start()
        assert adapter.is_ready is True
        assert adapter.is_degraded is False

        # Test decompose success path
        intents = await adapter.decompose("test command")
        assert len(intents) == 1
        assert intents[0].agent_source == "adk"
        mock_agent_instance.decompose.assert_called_once_with(
            "test command",
            max_sub_tasks=cfg.max_sub_tasks,
            model_name=cfg.model_name,
        )

        # Test stop
        await adapter.stop()
        assert adapter.is_ready is False


@pytest.mark.asyncio
async def test_start_missing_agent_attr_or_none_degrades() -> None:
    """start() gracefully degrades when Agent attr is missing or returns None."""
    from unittest.mock import MagicMock, patch

    from mousedroid.agents.adk_adapter import ADKMissionAdapter
    from mousedroid.config.schema.agents import ADKConfig

    cfg = ADKConfig(enabled=True)

    # Sub-case A: adk module has no Agent attribute
    mock_adk_no_agent = MagicMock(spec=[])
    with patch.dict("sys.modules", {"google": MagicMock(), "google.adk": mock_adk_no_agent}):
        adapter = ADKMissionAdapter(cfg)
        await adapter.start()
        assert adapter.is_ready is False
        assert adapter.is_degraded is True

    # Sub-case B: adk.Agent(...) returns None
    mock_adk = MagicMock()
    mock_adk.Agent.return_value = None
    with patch.dict("sys.modules", {"google": MagicMock(), "google.adk": mock_adk}):
        adapter = ADKMissionAdapter(cfg)
        await adapter.start()
        assert adapter.is_ready is False
        assert adapter.is_degraded is True


@pytest.mark.asyncio
async def test_decompose_validates_and_filters_dict_and_non_dict_items() -> None:
    """decompose() filters unexpected fields, handles enum conversion,
    strings, and MissionIntent.
    """
    from unittest.mock import MagicMock, patch

    from mousedroid.agents.adk_adapter import ADKMissionAdapter
    from mousedroid.config.schema.agents import ADKConfig
    from mousedroid.llm_gateway.mission_parser import IntentType, MissionIntent

    mock_adk = MagicMock()
    mock_agent_instance = MagicMock()
    mock_agent_instance.decompose.return_value = [
        # 1. Dict with extra unexpected fields + valid string intent_type
        {
            "intent_type": "patrol",
            "confidence": 0.9,
            "sub_tasks": ["sweep_left", "sweep_right"],
            "unexpected_field": "should_be_stripped",
        },
        # 2. Dict with invalid intent_type string -> should fallback to IntentType.UNKNOWN
        {
            "intent_type": "invalid_intent_xyz",
            "confidence": 0.5,
        },
        # 3. Non-dict string item -> wrapped as MissionIntent with sub_tasks=(str(item),)
        "navigate to waypoint 3",
        # 4. Existing MissionIntent instance
        MissionIntent(agent_source="adk", raw_command="existing intent"),
    ]
    mock_adk.Agent.return_value = mock_agent_instance

    with patch.dict("sys.modules", {"google": MagicMock(), "google.adk": mock_adk}):
        cfg = ADKConfig(enabled=True, model_name="test")
        adapter = ADKMissionAdapter(cfg)
        await adapter.start()

        intents = await adapter.decompose("patrol the perimeter")
        assert len(intents) == 4

        # 1.
        assert intents[0].intent_type == IntentType.PATROL
        assert intents[0].confidence == 0.9
        assert intents[0].sub_tasks == ("sweep_left", "sweep_right")
        assert intents[0].raw_command == "patrol the perimeter"
        assert not hasattr(intents[0], "unexpected_field")

        # 2.
        assert intents[1].intent_type == IntentType.UNKNOWN
        assert intents[1].confidence == 0.5

        # 3.
        assert intents[2].sub_tasks == ("navigate to waypoint 3",)
        assert intents[2].raw_command == "patrol the perimeter"

        # 4.
        assert intents[3].raw_command == "existing intent"

        await adapter.stop()
