"""Smoke-pass: the graceful-shutdown surface imports and parses (S-1).

Sub-second checks that the pieces exist and agree with each other on any
host, including one with no hardware and no POSIX signals. Behavioural
coverage lives in the unit/integration tiers.

The deployment checks belong here rather than in a unit test: they pin the
relationship between a Python default and a container setting, which no
single module owns.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.smoke

_REPO_ROOT = Path(__file__).resolve().parents[2]
_COMPOSE = _REPO_ROOT / "docker-compose.jetson.yml"
_ENTRYPOINT = _REPO_ROOT / "scripts" / "mousedroid_entrypoint.sh"


def test_signals_module_imports() -> None:
    """The module must import with no optional extras present."""
    from mousedroid.common.signals import (
        DEFAULT_SHUTDOWN_SIGNALS,
        install_shutdown_handlers,
    )

    assert callable(install_shutdown_handlers)
    assert len(DEFAULT_SHUTDOWN_SIGNALS) == 2


def test_shutdown_grace_parses_from_default_config() -> None:
    """``loop.shutdown_grace_s`` resolves through the real loader."""
    from mousedroid.config.loader import load_settings

    assert load_settings().loop.shutdown_grace_s > 0


def test_entrypoint_still_execs_the_python_process() -> None:
    """``exec`` is what makes the container's PID 1 receive SIGTERM at all.

    Without it the shell is PID 1, SIGTERM goes to the shell, and the
    Python handler installed by this work never sees the signal.
    """
    source = _ENTRYPOINT.read_text(encoding="utf-8")
    assert re.search(r"^exec python3 -m mousedroid\.main", source, re.MULTILINE)


def test_compose_stop_grace_exceeds_the_loop_grace() -> None:
    """The container must outlive the loop's grace, or SIGKILL pre-empts the halt.

    ``stop_grace_period`` bounds the whole teardown: the loop winding down
    (``shutdown_grace_s``) *and* ``stop()`` draining background tasks,
    flushing the cloud sink and closing transports afterwards.
    """
    from mousedroid.config.loader import load_settings

    compose = yaml.safe_load(_COMPOSE.read_text(encoding="utf-8"))
    raw = compose["services"]["mousedroid"]["stop_grace_period"]

    match = re.fullmatch(r"(\d+)s", str(raw))
    assert match, f"expected a plain-seconds duration, got {raw!r}"
    stop_grace_s = int(match.group(1))

    loop_grace_s = load_settings().loop.shutdown_grace_s
    assert stop_grace_s > loop_grace_s
