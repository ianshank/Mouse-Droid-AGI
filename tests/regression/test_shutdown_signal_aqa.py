"""Automated Quality Assurance (AQA) — schema + surface hygiene for S-1.

Covers the graceful-shutdown work: ``LoopConfig.shutdown_grace_s``, the
``mousedroid.common.signals`` entry point, and the orchestrator methods
``main.py`` now depends on.

Field hygiene is checked on ``model_fields`` (the ``FieldInfo``) rather
than by instantiating, so a refactor that replaces ``Field(...)`` with a
plain class attribute or a property override is still caught — an instance
only proves the default is legal, not that it is declared.
"""

from __future__ import annotations

import inspect

import pytest
from pydantic import ValidationError
from pydantic.fields import FieldInfo

from mousedroid.common.signals import DEFAULT_SHUTDOWN_SIGNALS, install_shutdown_handlers
from mousedroid.config.schema import ESP32Config
from mousedroid.config.schema.misc import LoopConfig

# ---------------------------------------------------------------------------
# LoopConfig.shutdown_grace_s
# ---------------------------------------------------------------------------


def test_shutdown_grace_s_has_description() -> None:
    """The field carries a non-empty, explanatory description."""
    info: FieldInfo = LoopConfig.model_fields["shutdown_grace_s"]
    assert info.description
    assert len(info.description) > 20, info.description


def test_shutdown_grace_s_default_is_two_seconds() -> None:
    """Pinned off FieldInfo, not a live instance — see module docstring."""
    info: FieldInfo = LoopConfig.model_fields["shutdown_grace_s"]
    assert info.default == 2.0


def test_shutdown_grace_s_clears_worst_case_cooperative_exit() -> None:
    """The grace must exceed one tick timeout or it would pre-empt a healthy exit.

    A grace shorter than ``tick_timeout_s`` would cancel a loop that was
    about to wind down on its own, turning the backstop into the common
    case.
    """
    grace = LoopConfig.model_fields["shutdown_grace_s"].default
    tick_timeout = LoopConfig.model_fields["tick_timeout_s"].default
    assert grace > tick_timeout


def test_zero_shutdown_grace_is_rejected_at_load() -> None:
    """A misconfigured grace is rejected at YAML-load time, not silently accepted."""
    with pytest.raises(ValidationError, match=r"shutdown_grace_s"):
        LoopConfig(shutdown_grace_s=0.0)


def test_negative_shutdown_grace_is_rejected_at_load() -> None:
    """Negative would mean "cancel immediately" — reject rather than reinterpret."""
    with pytest.raises(ValidationError, match=r"shutdown_grace_s"):
        LoopConfig(shutdown_grace_s=-1.0)


def test_positive_shutdown_grace_loads_cleanly() -> None:
    """The same validator, satisfied — proves it isn't just always-raise."""
    cfg = LoopConfig(shutdown_grace_s=7.5)
    assert cfg.shutdown_grace_s == 7.5


# ---------------------------------------------------------------------------
# mousedroid.common.signals
# ---------------------------------------------------------------------------


def test_default_signals_cover_sigterm_and_sigint() -> None:
    """SIGTERM is what docker/systemd send; SIGINT keeps Ctrl-C on one path."""
    names = {s.name for s in DEFAULT_SHUTDOWN_SIGNALS}
    assert names == {"SIGTERM", "SIGINT"}


def test_install_shutdown_handlers_signature_is_stable() -> None:
    """Bare presence is not enough — pin the parameter shape callers rely on."""
    sig = inspect.signature(install_shutdown_handlers)
    assert list(sig.parameters) == ["on_shutdown", "signals"]
    assert sig.parameters["signals"].kind is inspect.Parameter.KEYWORD_ONLY
    assert sig.parameters["signals"].default is DEFAULT_SHUTDOWN_SIGNALS


# ---------------------------------------------------------------------------
# Orchestrator surface main.py depends on
# ---------------------------------------------------------------------------


def test_orchestrator_exposes_the_shutdown_surface() -> None:
    """``main.py`` calls these by name; a rename must fail here, not on the rover."""
    from mousedroid.orchestrator.orchestrator import MouseDroidOrchestrator

    assert inspect.iscoroutinefunction(MouseDroidOrchestrator.serve)
    # Deliberately NOT a coroutine: it is called from a signal handler,
    # which cannot await.
    assert not inspect.iscoroutinefunction(MouseDroidOrchestrator.request_shutdown)

    sig = inspect.signature(MouseDroidOrchestrator.request_shutdown)
    assert list(sig.parameters) == ["self", "reason"]


def test_main_drives_the_loop_through_serve() -> None:
    """The one-line wiring that makes the whole fix reachable."""
    import mousedroid.main as main_module

    source = inspect.getsource(main_module._run)
    assert "serve()" in source
    assert "await orch_obj.run()" not in source


# ---------------------------------------------------------------------------
# Heartbeat visibility (the failsafe the signal path cannot cover)
# ---------------------------------------------------------------------------


def test_heartbeat_enabled_still_defaults_true() -> None:
    """Unchanged by this work — pinned because the new warning keys off it."""
    info: FieldInfo = ESP32Config.model_fields["heartbeat_enabled"]
    assert info.default is True


def test_arm_command_set_warns_when_heartbeat_cannot_arm() -> None:
    """The legacy default cannot arm a chassis failsafe; silence hid that."""
    from mousedroid.comms.base_driver import BaseESP32Driver

    source = inspect.getsource(BaseESP32Driver._arm_command_set)
    assert "esp32_heartbeat_unavailable" in source
