"""Tests for ``build_face_display`` and ``build_face_controller`` factories."""

from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import MagicMock

import pytest

from mousedroid.config.schema import FaceDisplayConfig, Settings
from mousedroid.factory import build_face_controller, build_face_display
from mousedroid.hardware.display.mock_face_driver import MockFaceDriver
from mousedroid.orchestrator.face_controller import FaceController


def _settings(face_cfg: FaceDisplayConfig | None, *, mock_hardware: bool = True) -> Settings:
    """Build a Settings object with the minimum sensors the schema requires."""
    payload: dict[str, Any] = {"mock_hardware": mock_hardware}
    if not mock_hardware:
        # Real-hardware mode requires at least one distance sensor; the
        # ultrasonic mock satisfies the validator without affecting the
        # face-display test path.
        payload["ultrasonic"] = {"trigger_pin": 0, "echo_pin": 0}
    if face_cfg is not None:
        payload["face_display"] = face_cfg.model_dump()
    return Settings.model_validate(payload)


def test_build_face_display_returns_none_when_section_omitted() -> None:
    cfg = Settings.model_validate({"mock_hardware": True})
    assert build_face_display(cfg) is None


def test_build_face_display_returns_none_when_disabled() -> None:
    cfg = _settings(FaceDisplayConfig(enabled=False))
    assert build_face_display(cfg) is None


def test_build_face_display_returns_mock_under_mock_hardware() -> None:
    cfg = _settings(FaceDisplayConfig(enabled=True))
    drv = build_face_display(cfg)
    assert isinstance(drv, MockFaceDriver)


def test_build_face_display_falls_back_to_mock_when_probe_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Probe failure must trigger the mock fallback when the flag is set."""
    smbus2 = types.ModuleType("smbus2")

    class _ErrSMBus:
        def __init__(self, _bus: int) -> None: ...

        def __enter__(self) -> _ErrSMBus:
            return self

        def __exit__(self, *exc: Any) -> None:
            return None

        def read_byte(self, _addr: int) -> int:
            raise OSError("simulated probe failure")

    smbus2.SMBus = _ErrSMBus  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "smbus2", smbus2)

    cfg = _settings(
        FaceDisplayConfig(enabled=True, fallback_to_mock_on_error=True),
        mock_hardware=False,
    )
    drv = build_face_display(cfg)
    assert isinstance(drv, MockFaceDriver)


def test_build_face_display_propagates_probe_failure_when_fallback_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the fallback flag, the probe error must propagate."""
    smbus2 = types.ModuleType("smbus2")

    class _ErrSMBus:
        def __init__(self, _bus: int) -> None: ...

        def __enter__(self) -> _ErrSMBus:
            return self

        def __exit__(self, *exc: Any) -> None:
            return None

        def read_byte(self, _addr: int) -> int:
            raise OSError("simulated probe failure")

    smbus2.SMBus = _ErrSMBus  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "smbus2", smbus2)

    cfg = _settings(
        FaceDisplayConfig(enabled=True, fallback_to_mock_on_error=False),
        mock_hardware=False,
    )
    with pytest.raises(OSError, match="simulated probe failure"):
        build_face_display(cfg)


def test_build_face_display_does_not_swallow_unexpected_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only ImportError / OSError are caught; other exceptions must propagate.

    If an AttributeError or similar programming mistake happens inside the
    driver code, it must never be silently swallowed and replaced by a mock.
    """
    smbus2 = types.ModuleType("smbus2")

    class _AttrErrSMBus:
        def __init__(self, _bus: int) -> None: ...

        def __enter__(self) -> _AttrErrSMBus:
            return self

        def __exit__(self, *exc: Any) -> None:
            return None

        def read_byte(self, _addr: int) -> int:
            raise AttributeError("unexpected programming error")

    smbus2.SMBus = _AttrErrSMBus  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "smbus2", smbus2)

    cfg = _settings(
        FaceDisplayConfig(enabled=True, fallback_to_mock_on_error=True),
        mock_hardware=False,
    )
    # fallback_to_mock_on_error=True must NOT catch AttributeError.
    with pytest.raises(AttributeError, match="unexpected programming error"):
        build_face_display(cfg)


def test_build_face_controller_returns_none_for_none_driver() -> None:
    cfg = _settings(FaceDisplayConfig(enabled=True))
    assert build_face_controller(cfg, None) is None


def test_build_face_controller_returns_none_when_face_display_section_missing() -> None:
    cfg = Settings.model_validate({"mock_hardware": True})
    assert build_face_controller(cfg, MagicMock()) is None


def test_build_face_controller_wraps_driver() -> None:
    cfg = _settings(FaceDisplayConfig(enabled=True))
    drv = build_face_display(cfg)
    fc = build_face_controller(cfg, drv)
    assert isinstance(fc, FaceController)
