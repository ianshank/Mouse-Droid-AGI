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

_BASE_KEYS = {
    "MOUSEDROID_HEARTBEAT_PATH",
    "MOUSEDROID_HEARTBEAT_STALE_S",
    "MOUSEDROID_START_GRACE_S",
    "MOUSEDROID_START_GRACE_FILE",
}
_PORT_KEY = "MOUSEDROID_RESOLVED_TELEMETRY_PORT"
_PATH_KEY = "MOUSEDROID_RESOLVED_HEALTH_PATH"


def _settings(**loop_overrides: object) -> Settings:
    """Build a minimal ``Settings`` with ``loop`` overrides."""
    return Settings.model_validate({"loop": loop_overrides, "mock_hardware": True})


def test_derive_returns_exact_required_keys() -> None:
    """Contract: the exact key set, so an addition is a deliberate edit here.

    The four heartbeat/grace keys are read by ``mousedroid_healthcheck.sh``; the
    two ``RESOLVED_`` keys are read by ``docker_deploy.sh`` (F-052 task 5.3), so
    the deploy probe follows the port the rover actually serves.
    """
    env = derive_healthcheck_env(_settings())
    assert set(env) == {
        "MOUSEDROID_HEARTBEAT_PATH",
        "MOUSEDROID_HEARTBEAT_STALE_S",
        "MOUSEDROID_START_GRACE_S",
        "MOUSEDROID_START_GRACE_FILE",
        "MOUSEDROID_RESOLVED_TELEMETRY_PORT",
        "MOUSEDROID_RESOLVED_HEALTH_PATH",
    }


def test_resolved_port_follows_the_telemetry_config() -> None:
    """The port the rover will serve, from the same Settings — not a default.

    The deploy probe used to read ``MOUSEDROID_TELEMETRY_PORT``, which is not a
    settings key, so moving ``telemetry.port`` left it probing the old port.
    """
    cfg = Settings.model_validate({"mock_hardware": True, "telemetry": {"port": 9191}})

    assert derive_healthcheck_env(cfg)["MOUSEDROID_RESOLVED_TELEMETRY_PORT"] == "9191"


def test_resolved_health_path_follows_the_api_prefix() -> None:
    cfg = Settings.model_validate({"mock_hardware": True, "telemetry": {"api_prefix": "/rover/v2"}})

    assert derive_healthcheck_env(cfg)["MOUSEDROID_RESOLVED_HEALTH_PATH"] == "/rover/v2/health"


@pytest.mark.parametrize(
    "api_prefix",
    [
        "/api'; touch /tmp/x; '",  # would break out of the env file's quotes
        "/api$(id)",
        "api/v1",  # no leading slash
        "/api v1",
        "/api\nX=1",
        "/ap\u00ef",  # non-ASCII
        "/a?b=c",
        # Passes the character rule; HTTP clients drop the ".." before sending.
        "/api/..",
        "/api/.",
    ],
)
def test_a_prefix_the_probe_cannot_use_is_omitted_never_raised(api_prefix: str) -> None:
    """``api_prefix`` is a free-form ``str`` in the schema; every value above loads.

    The entrypoint writes this file under ``set -eu`` and only then execs the
    rover, so a raise here turns a prefix the rover accepts into a container
    crash loop. The path is published EMPTY instead -- "cannot vouch", which
    ``docker_deploy.sh`` reports -- and everything else is still written.
    """
    cfg = Settings.model_validate({"mock_hardware": True, "telemetry": {"api_prefix": api_prefix}})

    env = derive_healthcheck_env(cfg)

    assert env[_PATH_KEY] == ""
    assert env[_PORT_KEY] == "8080"
    assert all(env[key] for key in _BASE_KEYS)


def test_a_prefix_outside_the_old_whitelist_but_url_safe_is_published() -> None:
    """``~`` is an RFC 3986 unreserved character and the probe accepts it.

    The first version reused the heartbeat-path whitelist, which has no ``~``,
    and so crashed the entrypoint on a prefix like this one.
    """
    cfg = Settings.model_validate({"mock_hardware": True, "telemetry": {"api_prefix": "/~rover"}})

    assert derive_healthcheck_env(cfg)[_PATH_KEY] == "/~rover/health"


@pytest.mark.parametrize("strategy", ["fallback_range", "kernel_assigned"])
def test_the_port_is_omitted_when_the_rover_picks_it_at_startup(strategy: str) -> None:
    """Only ``fixed`` binds ``telemetry.port`` itself.

    ``fallback_range`` may end on a later port and ``kernel_assigned`` on any
    port; publishing the configured one would hand the probe a confident wrong
    answer.
    """
    cfg = Settings.model_validate(
        {
            "mock_hardware": True,
            "telemetry": {"port": 9191, "port_discovery_strategy": strategy},
        }
    )

    env = derive_healthcheck_env(cfg)

    assert env[_PORT_KEY] == ""
    assert env[_PATH_KEY] == "/api/v1/health"


class _ShoutingInt(int):
    """An ``int`` whose ``str()`` is not its digits."""

    def __str__(self) -> str:
        return "1'; echo INJECTED; '"

    __repr__ = __str__


@pytest.mark.parametrize(
    ("port", "expected"),
    [
        pytest.param("1'; echo INJECTED; '", "", id="str-bypassing-validation"),
        pytest.param(True, "", id="bool"),  # an int subclass; never a port
        pytest.param(0, "", id="zero"),
        pytest.param(65536, "", id="above-max"),
        # Written back through int(), so only its digits reach the file.
        pytest.param(_ShoutingInt(8080), "8080", id="int-with-hostile-str"),
    ],
)
def test_a_port_that_bypassed_validation_can_only_write_digits(port: object, expected: str) -> None:
    """Defence in depth, as the module docstring promises for every value.

    ``model_copy(update=...)`` skips validation -- the same route
    ``model_construct`` or a test double takes.
    """
    cfg = Settings.model_validate({"mock_hardware": True})
    cfg = cfg.model_copy(update={"telemetry": cfg.telemetry.model_copy(update={"port": port})})

    env = derive_healthcheck_env(cfg)

    assert env[_PORT_KEY] == expected


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
    """Shell cannot distinguish missing vs empty — all values must be set.

    Holds for every key under a default config. A ``RESOLVED_`` key may be
    empty, but only when the rover cannot vouch for it (tested above), and
    the four healthcheck keys never are.
    """
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
