"""Tests for watchdog notifier implementations."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from mousedroid.health.watchdog import (
    FileHeartbeatNotifier,
    NullNotifier,
    SystemdNotifier,
    WatchdogProtocol,
)

# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


def test_null_notifier_satisfies_protocol() -> None:
    """NullNotifier implements WatchdogProtocol."""
    notifier = NullNotifier()
    assert isinstance(notifier, WatchdogProtocol)


def test_file_heartbeat_satisfies_protocol(tmp_path: Path) -> None:
    """FileHeartbeatNotifier implements WatchdogProtocol."""
    notifier = FileHeartbeatNotifier(path=tmp_path / "heartbeat")
    assert isinstance(notifier, WatchdogProtocol)


def test_systemd_notifier_satisfies_protocol() -> None:
    """SystemdNotifier implements WatchdogProtocol."""
    with patch.object(SystemdNotifier, "_build_notifier", return_value=None):
        notifier = SystemdNotifier(ready_on_init=False)
    assert isinstance(notifier, WatchdogProtocol)


# ---------------------------------------------------------------------------
# NullNotifier
# ---------------------------------------------------------------------------


def test_null_notifier_notify_is_noop() -> None:
    """NullNotifier.notify() does nothing and doesn't raise."""
    notifier = NullNotifier()
    notifier.notify()  # Should not raise


# ---------------------------------------------------------------------------
# FileHeartbeatNotifier
# ---------------------------------------------------------------------------


def test_file_heartbeat_creates_parent_dirs(tmp_path: Path) -> None:
    """FileHeartbeatNotifier creates parent directories on construction."""
    deep_path = tmp_path / "a" / "b" / "c" / "heartbeat"
    FileHeartbeatNotifier(path=deep_path)
    assert deep_path.parent.exists()


def test_file_heartbeat_notify_writes_file(tmp_path: Path) -> None:
    """notify() writes a timestamp to the heartbeat file."""
    path = tmp_path / "heartbeat"
    notifier = FileHeartbeatNotifier(path=path)
    notifier.notify()
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    # Should be a float (monotonic timestamp)
    float(content)


def test_file_heartbeat_notify_updates_timestamp(tmp_path: Path) -> None:
    """Successive notify() calls update the heartbeat file content."""
    path = tmp_path / "heartbeat"
    notifier = FileHeartbeatNotifier(path=path)

    notifier.notify()
    first = path.read_text(encoding="utf-8")

    time.sleep(0.05)  # Generous sleep to avoid flakiness under load
    notifier.notify()
    second = path.read_text(encoding="utf-8")

    assert float(second) > float(first)


def test_file_heartbeat_path_property(tmp_path: Path) -> None:
    """path property returns the heartbeat file path."""
    path = tmp_path / "heartbeat"
    notifier = FileHeartbeatNotifier(path=path)
    assert notifier.path == path


# ---------------------------------------------------------------------------
# SystemdNotifier
# ---------------------------------------------------------------------------


def test_systemd_notifier_with_sdnotify_package() -> None:
    """SystemdNotifier uses sdnotify package when available."""
    mock_sd = MagicMock()
    mock_notifier_instance = MagicMock()
    mock_sd.SystemdNotifier.return_value = mock_notifier_instance

    with patch.dict("sys.modules", {"sdnotify": mock_sd}):
        notifier = SystemdNotifier(ready_on_init=True)

    # Should have sent READY=1 on init
    mock_notifier_instance.notify.assert_called_with("READY=1")

    # notify() should send WATCHDOG=1
    notifier.notify()
    mock_notifier_instance.notify.assert_called_with("WATCHDOG=1")


def test_systemd_notifier_without_sdnotify() -> None:
    """SystemdNotifier works without sdnotify (fallback to subprocess)."""
    with patch.object(SystemdNotifier, "_build_notifier", return_value=None):
        notifier = SystemdNotifier(ready_on_init=False)

    # notify() should not raise even without sdnotify
    with patch.dict("os.environ", {}, clear=False):
        notifier.notify()  # No NOTIFY_SOCKET — should be silent no-op


def test_systemd_notifier_subprocess_fallback() -> None:
    """SystemdNotifier falls back to subprocess when sdnotify is unavailable."""
    with patch.object(SystemdNotifier, "_build_notifier", return_value=None):
        notifier = SystemdNotifier(ready_on_init=False)

    with (
        patch.dict("os.environ", {"NOTIFY_SOCKET": "/tmp/test.sock"}),
        patch("subprocess.run") as mock_run,
    ):
        notifier.notify()

    mock_run.assert_called_once()
    args = mock_run.call_args
    assert "systemd-notify" in args[0][0][0]
    assert "WATCHDOG=1" in args[0][0][-1]


def test_systemd_notifier_ready_on_init_false() -> None:
    """ready_on_init=False skips READY=1 on construction."""
    mock_sd = MagicMock()
    mock_notifier_instance = MagicMock()
    mock_sd.SystemdNotifier.return_value = mock_notifier_instance

    with patch.dict("sys.modules", {"sdnotify": mock_sd}):
        SystemdNotifier(ready_on_init=False)

    # Should NOT have sent READY=1
    mock_notifier_instance.notify.assert_not_called()


def test_build_notifier_returns_none_when_sdnotify_missing() -> None:
    """_build_notifier() returns None when sdnotify is not installed."""
    # Force sdnotify out of sys.modules so ImportError path executes.
    # Track presence separately: sys.modules may legitimately hold None
    # (import-blocked marker), which pop(..., None) alone cannot distinguish.
    import sys

    had_key = "sdnotify" in sys.modules
    orig = sys.modules.pop("sdnotify", None)

    try:
        result = SystemdNotifier._build_notifier()
        assert result is None
    finally:
        # Restore the exact previous state
        if had_key:
            sys.modules["sdnotify"] = orig  # type: ignore[assignment]
        else:
            sys.modules.pop("sdnotify", None)


def test_file_heartbeat_path_configurable(tmp_path: Path) -> None:
    """FileHeartbeatNotifier accepts a configurable path (not hardcoded)."""
    custom_path = tmp_path / "custom" / "heartbeat.txt"
    notifier = FileHeartbeatNotifier(path=custom_path)
    notifier.notify()
    assert custom_path.exists()
    assert notifier.path == custom_path
