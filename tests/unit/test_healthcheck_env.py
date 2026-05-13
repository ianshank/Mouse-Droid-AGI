"""Unit tests for :func:`mousedroid.health.healthcheck_env.derive_healthcheck_env`."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mousedroid.config.schema import Settings
from mousedroid.health.healthcheck_env import (
    _SAFE_PATH_RE,
    _validate_path,
    derive_healthcheck_env,
)


def _settings(**loop_overrides: object) -> Settings:
    """Build a minimal ``Settings`` with ``loop`` overrides."""
    return Settings.model_validate({"loop": loop_overrides, "mock_hardware": True})


def test_derive_returns_exact_required_keys() -> None:
    """Contract: env mapping has exactly the 4 keys the shell script reads."""
    env = derive_healthcheck_env(_settings())
    assert set(env) == {
        "MOUSEDROID_HEARTBEAT_PATH",
        "MOUSEDROID_HEARTBEAT_STALE_S",
        "MOUSEDROID_START_GRACE_S",
        "MOUSEDROID_START_GRACE_FILE",
    }


def test_stale_threshold_is_interval_times_tolerance() -> None:
    """Threshold derivation is ``interval * tolerance_factor``, no constants."""
    env = derive_healthcheck_env(
        _settings(watchdog_interval_s=5.0, watchdog_tolerance_factor=4.0),
    )
    assert float(env["MOUSEDROID_HEARTBEAT_STALE_S"]) == pytest.approx(20.0)


def test_path_passes_through_unchanged() -> None:
    """Heartbeat path is not normalized or rewritten by the helper."""
    env = derive_healthcheck_env(_settings(watchdog_heartbeat_path="/var/run/hb"))
    assert env["MOUSEDROID_HEARTBEAT_PATH"] == "/var/run/hb"


def test_all_values_are_non_empty_strings() -> None:
    """Shell cannot distinguish missing vs empty — all values must be set."""
    env = derive_healthcheck_env(_settings())
    for key, value in env.items():
        assert isinstance(value, str), key
        assert value, key


def test_zero_grace_is_accepted() -> None:
    """``start_grace_s=0`` is valid (no grace) — must not raise or coerce."""
    env = derive_healthcheck_env(_settings(start_grace_s=0.0))
    assert float(env["MOUSEDROID_START_GRACE_S"]) == 0.0


def test_start_grace_file_is_settings_driven() -> None:
    """``start_grace_file`` flows from Settings, not a Python literal."""
    env = derive_healthcheck_env(_settings(start_grace_file="/var/lib/mousedroid.start"))
    assert env["MOUSEDROID_START_GRACE_FILE"] == "/var/lib/mousedroid.start"


def test_shell_unsafe_heartbeat_path_rejected_at_load() -> None:
    """Pydantic validator rejects shell-metacharacter paths at YAML load."""
    with pytest.raises(ValidationError, match="shell-unsafe"):
        Settings.model_validate(
            {
                "loop": {"watchdog_heartbeat_path": "/tmp/x'$(rm -rf /)'/hb"},
                "mock_hardware": True,
            },
        )


def test_shell_unsafe_start_grace_file_rejected_at_load() -> None:
    """Same Pydantic validator applies to ``start_grace_file``."""
    with pytest.raises(ValidationError, match="shell-unsafe"):
        Settings.model_validate(
            {
                "loop": {"start_grace_file": "/run/x;evil"},
                "mock_hardware": True,
            },
        )


@pytest.mark.parametrize(
    "value",
    [
        "/tmp/mousedroid_heartbeat",
        "/var/run/mousedroid:1",
        "/run/mousedroid.start",
        "C:/Users/data/hb",
    ],
)
def test_safe_path_whitelist_accepts_real_paths(value: str) -> None:
    """The whitelist regex permits paths actually used in production."""
    assert _SAFE_PATH_RE.fullmatch(value) is not None


@pytest.mark.parametrize(
    "value",
    [
        "/tmp/x;rm -rf /",
        "/tmp/x'\"",
        "/tmp/x`whoami`",
        "/tmp/x$(id)",
        "/tmp/x with spaces",
        "/tmp/x|true",
        "/tmp/x&&true",
    ],
)
def test_validate_path_rejects_attack_payloads(value: str) -> None:
    """Defense-in-depth: ``_validate_path`` rejects every common shell injection."""
    with pytest.raises(ValueError, match="unsafe"):
        _validate_path(value, "test_field")


def test_validate_path_error_includes_field_name() -> None:
    """Error message names the offending field so operators can find the YAML line."""
    with pytest.raises(ValueError, match="my_field"):
        _validate_path("/tmp/x'evil", "my_field")
