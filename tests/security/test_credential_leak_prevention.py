"""Tests to ensure credentials do not leak in logs or output."""

from __future__ import annotations

import json
from pydantic import SecretStr
from mousedroid.config.schema import Settings


def test_api_keys_masked_in_str_repr(mock_settings: Settings) -> None:
    """Test that SecretStr fields are masked in str() and repr()."""
    cfg = mock_settings.model_copy(deep=True)
    cfg.llm.api_key = SecretStr("secret-key-12345")

    settings_str = str(cfg)
    settings_repr = repr(cfg)

    assert "secret-key-12345" not in settings_str
    assert "secret-key-12345" not in settings_repr
    assert "**********" in settings_str or "SecretStr" in settings_str


def test_api_keys_masked_in_json_dump(mock_settings: Settings) -> None:
    """Test that API keys are masked when config is serialized."""
    cfg = mock_settings.model_copy(deep=True)
    cfg.llm.api_key = SecretStr("secret-key-12345")
    dumped = cfg.model_dump_json()
    assert "secret-key-12345" not in dumped


def test_exception_messages_do_not_leak_secrets(mock_settings: Settings) -> None:
    """Test that exception messages don't leak secrets."""
    cfg = mock_settings.model_copy(deep=True)
    cfg.llm.api_key = SecretStr("secret-key-12345")
    try:
        raise ValueError(f"Invalid config: {cfg.llm}")
    except ValueError as e:
        assert "secret-key-12345" not in str(e)
