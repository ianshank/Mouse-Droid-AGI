"""Unit tests for the watchdog notifier module.

Validates all three implementations (NullNotifier, FileHeartbeatNotifier,
SystemdNotifier) and the ``build_watchdog`` factory function.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from mousedroid.health.watchdog import (
    FileHeartbeatNotifier,
    NullNotifier,
    SystemdNotifier,
    WatchdogProtocol,
)

# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestWatchdogProtocol:
    """All notifiers must satisfy the WatchdogProtocol."""

    def test_null_satisfies_protocol(self) -> None:
        assert isinstance(NullNotifier(), WatchdogProtocol)

    def test_file_satisfies_protocol(self, tmp_path: Path) -> None:
        notifier = FileHeartbeatNotifier(tmp_path / "heartbeat")
        assert isinstance(notifier, WatchdogProtocol)

    def test_systemd_satisfies_protocol(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NOTIFY_SOCKET", raising=False)
        assert isinstance(SystemdNotifier(), WatchdogProtocol)


# ---------------------------------------------------------------------------
# NullNotifier
# ---------------------------------------------------------------------------


class TestNullNotifier:
    """NullNotifier must be a silent no-op."""

    def test_notify_does_not_raise(self) -> None:
        NullNotifier().notify()

    def test_multiple_notify_calls(self) -> None:
        notifier = NullNotifier()
        for _ in range(100):
            notifier.notify()


# ---------------------------------------------------------------------------
# FileHeartbeatNotifier
# ---------------------------------------------------------------------------


class TestFileHeartbeatNotifier:
    """FileHeartbeatNotifier writes a monotonic timestamp on each call."""

    def test_creates_heartbeat_file(self, tmp_path: Path) -> None:
        hb = tmp_path / "heartbeat"
        notifier = FileHeartbeatNotifier(hb)
        notifier.notify()
        assert hb.exists()

    def test_heartbeat_file_contains_timestamp(self, tmp_path: Path) -> None:
        hb = tmp_path / "heartbeat"
        notifier = FileHeartbeatNotifier(hb)
        notifier.notify()
        content = hb.read_text()
        assert float(content) > 0

    def test_heartbeat_updates_on_each_call(self, tmp_path: Path) -> None:
        hb = tmp_path / "heartbeat"
        notifier = FileHeartbeatNotifier(hb)
        notifier.notify()
        first = float(hb.read_text())
        notifier.notify()
        second = float(hb.read_text())
        assert second >= first

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        hb = tmp_path / "deep" / "nested" / "dir" / "heartbeat"
        notifier = FileHeartbeatNotifier(hb)
        notifier.notify()
        assert hb.exists()


# ---------------------------------------------------------------------------
# SystemdNotifier
# ---------------------------------------------------------------------------


class TestSystemdNotifier:
    """SystemdNotifier sends WATCHDOG=1 via NOTIFY_SOCKET."""

    def test_no_op_without_notify_socket(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NOTIFY_SOCKET", raising=False)
        notifier = SystemdNotifier()
        # Should not raise
        notifier.notify()

    def test_sends_watchdog_with_socket(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NOTIFY_SOCKET", "/run/systemd/notify")
        notifier = SystemdNotifier()
        # Mock sys.platform to "linux" so the Windows guard doesn't skip
        with (
            patch("mousedroid.health.watchdog.sys") as mock_sys,
            patch("socket.socket") as mock_socket_cls,
        ):
            mock_sys.platform = "linux"
            mock_sock = mock_socket_cls.return_value
            notifier.notify()
            mock_sock.sendto.assert_called_once_with(b"WATCHDOG=1", "/run/systemd/notify")
            mock_sock.close.assert_called_once()

    def test_abstract_socket_addr(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NOTIFY_SOCKET", "@/run/systemd/notify")
        notifier = SystemdNotifier()
        with (
            patch("mousedroid.health.watchdog.sys") as mock_sys,
            patch("socket.socket") as mock_socket_cls,
        ):
            mock_sys.platform = "linux"
            mock_sock = mock_socket_cls.return_value
            notifier.notify()
            expected_addr = "\0/run/systemd/notify"
            mock_sock.sendto.assert_called_once_with(b"WATCHDOG=1", expected_addr)

    def test_notify_suppresses_socket_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NOTIFY_SOCKET", "/run/systemd/notify")
        notifier = SystemdNotifier()
        with (
            patch("mousedroid.health.watchdog.sys") as mock_sys,
            patch("socket.socket", side_effect=OSError("socket error")),
        ):
            mock_sys.platform = "linux"
            # Must not propagate the error
            notifier.notify()


# ---------------------------------------------------------------------------
# build_watchdog factory
# ---------------------------------------------------------------------------


class TestBuildWatchdog:
    """Tests for the factory ``build_watchdog`` function."""

    def test_disabled_returns_null(self) -> None:
        from mousedroid.config.schema import LoopConfig, Settings

        cfg = Settings(mock_hardware=True, loop=LoopConfig(watchdog_enabled=False))
        from mousedroid.factory import build_watchdog

        result = build_watchdog(cfg)
        assert isinstance(result, NullNotifier)

    def test_auto_mode_no_socket_returns_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from mousedroid.config.schema import LoopConfig, Settings

        monkeypatch.delenv("NOTIFY_SOCKET", raising=False)
        hb_path = str(tmp_path / "heartbeat")
        cfg = Settings(
            mock_hardware=True,
            loop=LoopConfig(watchdog_enabled=True, watchdog_mode="auto", heartbeat_path=hb_path),
        )
        from mousedroid.factory import build_watchdog

        result = build_watchdog(cfg)
        assert isinstance(result, FileHeartbeatNotifier)

    def test_systemd_mode_returns_systemd(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from mousedroid.config.schema import LoopConfig, Settings

        monkeypatch.delenv("NOTIFY_SOCKET", raising=False)
        cfg = Settings(
            mock_hardware=True,
            loop=LoopConfig(watchdog_enabled=True, watchdog_mode="systemd"),
        )
        from mousedroid.factory import build_watchdog

        result = build_watchdog(cfg)
        assert isinstance(result, SystemdNotifier)

    def test_file_mode_returns_file(self, tmp_path: Path) -> None:
        from mousedroid.config.schema import LoopConfig, Settings

        hb_path = str(tmp_path / "heartbeat")
        cfg = Settings(
            mock_hardware=True,
            loop=LoopConfig(watchdog_enabled=True, watchdog_mode="file", heartbeat_path=hb_path),
        )
        from mousedroid.factory import build_watchdog

        result = build_watchdog(cfg)
        assert isinstance(result, FileHeartbeatNotifier)

    def test_unknown_mode_rejected_by_pydantic(self) -> None:
        from mousedroid.config.schema import LoopConfig

        with pytest.raises(Exception, match=r"literal_error|Input should be"):
            LoopConfig(watchdog_mode="bogus")

    def test_none_mode_returns_null(self) -> None:
        from mousedroid.config.schema import LoopConfig, Settings

        cfg = Settings(
            mock_hardware=True,
            loop=LoopConfig(watchdog_enabled=True, watchdog_mode="none"),
        )
        from mousedroid.factory import build_watchdog

        result = build_watchdog(cfg)
        assert isinstance(result, NullNotifier)
