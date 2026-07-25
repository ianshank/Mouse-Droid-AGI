# tests/unit/tools/claude_hooks/test_envflags.py
"""Unit tests for environment-flag truthiness.

The freeze-gate override resolves through here, so "0"/"false" reading as ON
would open a safety gate on exactly the value written to keep it shut.
"""

from __future__ import annotations

import pytest
from tools.claude_hooks.envflags import env_flag, is_truthy


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "True", "yes", "on", " on ", "ON"])
def test_affirmative_values_are_truthy(value: str) -> None:
    assert is_truthy(value) is True


@pytest.mark.parametrize(
    "value",
    ["0", "false", "FALSE", "no", "off", "", "   ", "maybe", "2", "-1", "null"],
)
def test_everything_else_is_falsy(value: str) -> None:
    # Load-bearing: a presence check would call every one of these "on".
    assert is_truthy(value) is False


def test_none_is_falsy() -> None:
    assert is_truthy(None) is False


def test_env_flag_reads_the_named_variable() -> None:
    assert env_flag({"FLAG": "1"}, "FLAG") is True
    assert env_flag({"FLAG": "0"}, "FLAG") is False
    assert env_flag({}, "FLAG") is False
