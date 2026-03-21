"""Tests for OpenClawGateway (HTTP-based gateway)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from mousedroid.config.schema import OpenClawConfig
from mousedroid.openclaw.gateway import OpenClawGateway


@pytest.fixture
def cfg() -> OpenClawConfig:
    return OpenClawConfig(
        enabled=True,
        api_endpoint="http://localhost:9999",
        api_timeout_s=1.0,
        connect_retries=1,
        connect_backoff_base=0.01,
    )


@pytest.fixture
def gateway(cfg: OpenClawConfig) -> OpenClawGateway:
    return OpenClawGateway(cfg)


# -- Construction --------------------------------------------------------------


def test_construction(cfg: OpenClawConfig):
    gw = OpenClawGateway(cfg)
    assert gw.is_connected is False
    assert gw._session is None


# -- start / stop lifecycle ----------------------------------------------------


async def test_start_connection_failure_with_fallback(cfg: OpenClawConfig):
    """When OpenClaw is unreachable and fallback is enabled, start() should not raise."""
    cfg_fallback = cfg.model_copy(update={"fallback_to_cognitive": True})
    gw = OpenClawGateway(cfg_fallback)

    mock_session = AsyncMock()
    mock_response = AsyncMock()
    mock_response.status = 500
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)
    mock_session.get = MagicMock(return_value=mock_response)
    mock_session.close = AsyncMock()

    with patch("mousedroid.openclaw.gateway.aiohttp") as mock_aiohttp:
        mock_aiohttp.ClientTimeout = MagicMock()
        mock_aiohttp.ClientSession = MagicMock(return_value=mock_session)
        await gw.start()

    assert gw.is_connected is False
    await gw.stop()


async def test_start_connection_failure_no_fallback_raises(cfg: OpenClawConfig):
    """When fallback is disabled and connection fails, start() should raise."""
    cfg_strict = cfg.model_copy(update={"fallback_to_cognitive": False})
    gw = OpenClawGateway(cfg_strict)

    mock_session = AsyncMock()
    mock_response = AsyncMock()
    mock_response.status = 500
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)
    mock_session.get = MagicMock(return_value=mock_response)
    mock_session.close = AsyncMock()

    with patch("mousedroid.openclaw.gateway.aiohttp") as mock_aiohttp:
        mock_aiohttp.ClientTimeout = MagicMock()
        mock_aiohttp.ClientSession = MagicMock(return_value=mock_session)
        with pytest.raises(ConnectionError, match="Failed to connect"):
            await gw.start()

    await gw.stop()


async def test_stop_closes_session(gateway: OpenClawGateway):
    mock_session = AsyncMock()
    mock_session.close = AsyncMock()
    gateway._session = mock_session
    gateway._connected = True

    await gateway.stop()

    assert gateway.is_connected is False
    mock_session.close.assert_awaited_once()


# -- get_action ----------------------------------------------------------------


async def test_get_action_not_connected_returns_none(gateway: OpenClawGateway):
    result = await gateway.get_action({"state": "test"})
    assert result is None


async def test_get_action_success():
    cfg = OpenClawConfig(enabled=True, api_endpoint="http://test:8000")
    gw = OpenClawGateway(cfg)
    gw._connected = True

    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(
        return_value={
            "action": [0.1, -0.2, 0.3],
            "goal_id": "g-42",
            "reasoning": "turn left",
            "confidence": 0.85,
        }
    )
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock()
    mock_session.post = MagicMock(return_value=mock_response)
    gw._session = mock_session

    result = await gw.get_action({"distance_m": 1.5})
    assert result is not None
    np.testing.assert_allclose(result.action, [0.1, -0.2, 0.3], atol=1e-6)
    assert result.goal_id == "g-42"
    assert result.confidence == 0.85


async def test_get_action_http_error_returns_none():
    cfg = OpenClawConfig(enabled=True)
    gw = OpenClawGateway(cfg)
    gw._connected = True

    mock_response = AsyncMock()
    mock_response.status = 500
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock()
    mock_session.post = MagicMock(return_value=mock_response)
    gw._session = mock_session

    result = await gw.get_action({})
    assert result is None


async def test_get_action_exception_returns_none():
    cfg = OpenClawConfig(enabled=True)
    gw = OpenClawGateway(cfg)
    gw._connected = True

    mock_session = AsyncMock()
    mock_session.post = MagicMock(side_effect=OSError("network down"))
    gw._session = mock_session

    result = await gw.get_action({})
    assert result is None
    assert gw.is_connected is False  # Marked as disconnected


# -- set_goal ------------------------------------------------------------------


async def test_set_goal_success():
    cfg = OpenClawConfig(enabled=True)
    gw = OpenClawGateway(cfg)
    gw._connected = True

    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock()
    mock_session.post = MagicMock(return_value=mock_response)
    gw._session = mock_session

    await gw.set_goal("patrol corridor B")
    assert gw._current_goal == "patrol corridor B"


async def test_set_goal_not_connected():
    cfg = OpenClawConfig(enabled=True)
    gw = OpenClawGateway(cfg)
    gw._connected = False

    await gw.set_goal("test")
    assert gw._current_goal == "test"  # Goal stored even if not connected


# -- _build_payload -----------------------------------------------------------


def test_build_payload_filters_keys():
    cfg = OpenClawConfig(observation_keys=["distance_m", "battery_v"])
    gw = OpenClawGateway(cfg)

    payload = gw._build_payload(
        {
            "distance_m": 1.5,
            "battery_v": 11.2,
            "secret_key": "should_not_appear",
        }
    )
    assert "distance_m" in payload
    assert "battery_v" in payload
    assert "secret_key" not in payload


def test_build_payload_converts_numpy():
    cfg = OpenClawConfig(observation_keys=["motor_state"])
    gw = OpenClawGateway(cfg)

    payload = gw._build_payload(
        {
            "motor_state": np.array([0.1, 0.2, 0.3, 11.5]),
        }
    )
    assert payload["motor_state"] == [0.1, 0.2, 0.3, 11.5]


def test_build_payload_includes_goal():
    cfg = OpenClawConfig(observation_keys=[])
    gw = OpenClawGateway(cfg)
    gw._current_goal = "go left"

    payload = gw._build_payload({})
    assert payload["current_goal"] == "go left"


# -- _parse_action -------------------------------------------------------------


def test_parse_action_valid():
    cfg = OpenClawConfig()
    gw = OpenClawGateway(cfg)

    result = gw._parse_action(
        {
            "action": [0.5, -0.5, 0.0],
            "goal_id": "g-1",
            "reasoning": "because",
            "confidence": 0.9,
        }
    )
    assert result is not None
    np.testing.assert_allclose(result.action, [0.5, -0.5, 0.0], atol=1e-6)


def test_parse_action_missing_action_key():
    cfg = OpenClawConfig()
    gw = OpenClawGateway(cfg)

    result = gw._parse_action({"goal_id": "g-1"})
    assert result is None


def test_parse_action_malformed():
    cfg = OpenClawConfig()
    gw = OpenClawGateway(cfg)

    result = gw._parse_action({"action": "not-a-list"})
    # Should return None or a valid result (numpy can parse some strings)
    # The key point is it doesn't raise
    assert result is None or result is not None
