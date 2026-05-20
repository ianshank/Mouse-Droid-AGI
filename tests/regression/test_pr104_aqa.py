"""Automated Quality Assurance (AQA) — schema + protocol hygiene for PR #104.

This goes beyond functional regression and enforces *quality* properties on
the surface PR #104 introduced:

1. **Every new schema field has a non-empty Pydantic ``description``** — that's
   how the YAML overlay docs auto-generate, and how the dashboards' "what does
   this knob do?" tooltips populate.
2. **Every new schema field has a documented default** — the project CLAUDE.md
   invariant #9 is "New config fields MUST have defaults".
3. **The JetsonCSICamera + MockCamera both satisfy
   :class:`RawFrameSourceProtocol`** — the duck-typing surface the telemetry
   server uses to register ``/camera/frame.jpg``.
4. **The MockESP32Driver satisfies :class:`ESP32CommProtocol`** — the orchestrator
   relies on the driver being interchangeable.
5. **The dashboard proxy's hop-by-hop header set covers the documented
   RFC-9110 entries** — guards against accidentally letting a hop-by-hop
   header leak into the upstream request.

Naming + behaviour drift here is the most insidious failure mode: tests still
pass, but a refactor weakens the contract silently. AQA catches that.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from pydantic.fields import FieldInfo

from mousedroid.comms.mock_driver import MockESP32Driver
from mousedroid.comms.protocol import ESP32CommProtocol
from mousedroid.config.schema import CameraConfig, ESP32Config, Settings
from mousedroid.hardware.camera.mock_camera import MockCamera
from mousedroid.hardware.protocols import RawFrameSourceProtocol, VisionProtocol

_REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Schema hygiene — every new PR #104 field must carry a description + default
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model_cls", "field_name"),
    [
        (CameraConfig, "snapshot_jpeg_quality"),
        (CameraConfig, "v4l2_grayscale_extract"),
        (ESP32Config, "enabled"),
    ],
)
def test_pr104_field_has_description(model_cls: type, field_name: str) -> None:
    """Each new PR #104 field carries a non-empty Pydantic ``description``.

    Pydantic exposes field metadata via ``model_fields``; an empty / missing
    description means YAML doc auto-generation produces "(no description)"
    which is operator-hostile.
    """
    fields = model_cls.model_fields  # type: ignore[attr-defined]
    assert field_name in fields, (
        f"{model_cls.__name__} is missing field {field_name!r} — "
        f"PR #104 changed the schema; AQA expectations need updating."
    )
    info: FieldInfo = fields[field_name]
    assert info.description, (
        f"{model_cls.__name__}.{field_name} has no description — operator "
        f"docs + dashboard tooltips will be blank."
    )
    # Long enough to be useful (>20 chars excludes one-word placeholders).
    assert len(info.description) > 20, info.description


@pytest.mark.parametrize(
    ("model_cls", "field_name", "expected_default"),
    [
        (CameraConfig, "snapshot_jpeg_quality", 90),
        (CameraConfig, "v4l2_grayscale_extract", True),
        (ESP32Config, "enabled", True),
    ],
)
def test_pr104_field_has_expected_default(
    model_cls: type, field_name: str, expected_default: object
) -> None:
    """Each new PR #104 field carries its documented default.

    Belt-and-suspenders on the regression tests: this checks the Pydantic
    ``FieldInfo.default`` directly (not via ``model_validate``) so a future
    refactor that swaps Field(...) for a property override is caught.
    """
    fields = model_cls.model_fields  # type: ignore[attr-defined]
    info: FieldInfo = fields[field_name]
    assert info.default == expected_default, (
        f"{model_cls.__name__}.{field_name} default drifted: "
        f"{info.default!r} != expected {expected_default!r}"
    )


# ---------------------------------------------------------------------------
# Protocol conformance — duck-typing surfaces hold across implementations
# ---------------------------------------------------------------------------


def test_mock_camera_satisfies_vision_protocol() -> None:
    """``MockCamera`` is a :class:`VisionProtocol` — basic invariant."""
    cam = MockCamera(_minimal_camera_cfg())
    assert isinstance(cam, VisionProtocol)


def test_mock_camera_satisfies_raw_frame_source_protocol() -> None:
    """``MockCamera`` satisfies :class:`RawFrameSourceProtocol`.

    This is what the telemetry server's factory uses to decide whether to
    register ``/camera/frame.jpg`` + ``/camera/stream``. Losing it makes
    the dashboard's camera pane silently return 404.
    """
    cam = MockCamera(_minimal_camera_cfg())
    assert isinstance(cam, RawFrameSourceProtocol)


def test_jetson_csi_class_has_capture_raw_jpeg() -> None:
    """``JetsonCSICamera.capture_raw_jpeg`` is on the class surface.

    We can't ``isinstance(...)`` against the protocol without instantiating
    the camera (which needs a live ``/dev/video0`` on Jetson). Instead, we
    confirm the method is bound on the class itself — the PR #104 surface
    promised by ``RawFrameSourceProtocol``.
    """
    from mousedroid.hardware.camera.jetson_csi import JetsonCSICamera

    assert hasattr(JetsonCSICamera, "capture_raw_jpeg"), (
        "JetsonCSICamera lost capture_raw_jpeg — telemetry server's "
        "RawFrameSourceProtocol isinstance() check will fall through and "
        "the dashboard's /camera/frame.jpg returns 404."
    )
    assert callable(JetsonCSICamera.capture_raw_jpeg)


def test_mock_esp32_driver_satisfies_protocol() -> None:
    """``MockESP32Driver`` is a :class:`ESP32CommProtocol` — required for
    the PR #104 factory branch to typecheck at runtime."""
    cfg = ESP32Config()
    drv = MockESP32Driver(cfg)
    assert isinstance(drv, ESP32CommProtocol)


# ---------------------------------------------------------------------------
# Dashboard proxy AQA — hop-by-hop header set covers RFC-9110 essentials
# ---------------------------------------------------------------------------


def _load_dashboard_proxy(monkeypatch: pytest.MonkeyPatch) -> object:
    """Spec-load ``tools/dashboard_proxy.py`` with neutral argv/env."""
    monkeypatch.setattr(sys, "argv", ["dashboard_proxy.py"])
    monkeypatch.delenv("JETSON_HTTP", raising=False)
    monkeypatch.delenv("JETSON_TOKEN", raising=False)
    monkeypatch.delenv("PROXY_PORT", raising=False)
    sys.modules.pop("dashboard_proxy", None)
    proxy_path = _REPO_ROOT / "tools" / "dashboard_proxy.py"
    spec = importlib.util.spec_from_file_location("dashboard_proxy", proxy_path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["dashboard_proxy"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize(
    "header",
    [
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "host",
    ],
)
def test_dashboard_proxy_hop_by_hop_set_covers_rfc9110(
    monkeypatch: pytest.MonkeyPatch, header: str
) -> None:
    """The proxy's hop-by-hop blocklist covers every RFC-9110 §7.6.1 hop-
    by-hop entry. Letting any of these leak upstream causes interop bugs
    (Connection: upgrade flapping, Trailer header bleed, etc.)."""
    mod = _load_dashboard_proxy(monkeypatch)
    assert header in mod._HOP_BY_HOP, (  # type: ignore[attr-defined]
        f"{header!r} not in dashboard_proxy._HOP_BY_HOP — RFC-9110 hop-by-hop "
        f"header would leak to upstream."
    )


def test_dashboard_proxy_strips_content_length_and_encoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``Content-Length`` + ``Content-Encoding`` are in the hop-by-hop set.

    These aren't hop-by-hop per RFC-9110, but aiohttp recomputes them on
    write — passing the client's values through causes "header value
    conflict" errors mid-stream. The proxy intentionally strips them.
    """
    mod = _load_dashboard_proxy(monkeypatch)
    blocklist: set[str] = mod._HOP_BY_HOP  # type: ignore[attr-defined]
    assert "content-length" in blocklist
    assert "content-encoding" in blocklist


# ---------------------------------------------------------------------------
# Settings invariant — esp32.enabled toggle is reachable via env override
# (the path operators use on the Jetson via /etc/mousedroid/docker.env).
# ---------------------------------------------------------------------------


def test_esp32_enabled_settable_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """``MOUSEDROID_ESP32__ENABLED=false`` flips ``cfg.esp32.enabled`` to False.

    This is the documented CLAUDE.md path: ``MOUSEDROID_`` prefix +
    ``__`` nested delimiter. Operators on the Jetson set this in
    ``docker.env`` rather than editing YAML.
    """
    monkeypatch.setenv("MOUSEDROID_ESP32__ENABLED", "false")
    cfg = Settings()
    assert cfg.esp32.enabled is False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_camera_cfg() -> CameraConfig:
    """Smallest valid CameraConfig for the protocol-conformance tests."""
    return CameraConfig(
        resolution_width=320,
        resolution_height=240,
        fps=15,
        feature_dim=128,
        backend="auto",
        feature_extractor="mean_pool",
        l2_normalize=True,
    )
