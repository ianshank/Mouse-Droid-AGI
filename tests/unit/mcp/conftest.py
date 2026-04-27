from __future__ import annotations

import re
from typing import Any
from unittest.mock import MagicMock

import pytest

from mousedroid.config.schema import (
    CircuitBreakerConfig,
    MCPConfig,
    MCPResourcesConfig,
    Settings,
)


@pytest.fixture
def root_settings() -> Settings:
    return Settings.model_validate({"mock_hardware": True})


@pytest.fixture
def mcp_cfg() -> MCPConfig:
    return MCPConfig.model_validate({"enabled": True, "transport": "stdio"})


@pytest.fixture
def fast_circuit_cfg() -> CircuitBreakerConfig:
    return CircuitBreakerConfig(failure_threshold=2, recovery_timeout_s=0.05)


@pytest.fixture
def redact_pattern() -> re.Pattern[str]:
    return re.compile(MCPConfig.model_fields["redact_key_pattern"].default)


@pytest.fixture
def safe_safety_monitor() -> Any:
    monitor = MagicMock()
    monitor.evaluate.return_value = MagicMock(is_emergency=False, violations=[])
    return monitor


@pytest.fixture
def emergency_safety_monitor() -> Any:
    monitor = MagicMock()
    monitor.evaluate.return_value = MagicMock(is_emergency=True, violations=["forward_clearance"])
    return monitor


@pytest.fixture
def resources_cfg() -> MCPResourcesConfig:
    return MCPResourcesConfig(
        telemetry_enabled=True,
        logs_enabled=True,
        config_enabled=True,
        memory_enabled=True,
        recent_frames_max=8,
        log_tail_max=8,
    )
