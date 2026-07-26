"""Tests for configuration injection via environment variables."""

from __future__ import annotations

import os

import pytest

from mousedroid.config.schema import Settings


@pytest.fixture
def env_with_shell_chars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOUSEDROID_TELEMETRY__HOST", "$(echo inject) & whoami")
    monkeypatch.setenv("MOUSEDROID_LLM__SYSTEM_PROMPT", "normal prompt; rm -rf /")


def test_config_safely_loads_shell_metacharacters(env_with_shell_chars: None) -> None:
    """Env vars with shell metacharacters must be treated as literals, not evaluated."""
    settings = Settings()

    # Check that they are literal strings
    assert settings.telemetry.host == "$(echo inject) & whoami"
    # Even if dangerous, Pydantic parses them as literals. The main issue is if they are used in os.system or similar.
    # We assert the literal value is kept.
    assert settings.llm.system_prompt == "normal prompt; rm -rf /"


def test_config_rejects_command_injection_in_restricted_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Some fields like ports should reject non-integers, thus preventing injection."""
    monkeypatch.setenv("MOUSEDROID_TELEMETRY__PORT", "8080; reboot")
    with pytest.raises(Exception):
        Settings()
