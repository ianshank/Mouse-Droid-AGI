"""Camera capture + JPEG-encode runtime validation helpers (Task 1 domain)."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from mousedroid.config.schema import Settings


@dataclass(frozen=True)
class CameraFrameDiagnostics:
    """Diagnostics record for one ``capture_camera_frame`` call (Task 1).

    All fields default so the dataclass can be constructed with zero
    positional args. The legacy ``(frame, backend_name)`` 2-tuple return
    shape of :func:`capture_camera_frame` is preserved — the diagnostics
    instance occupies the first slot and the backend name the second so
    existing destructuring (``frame, backend_name = ...``) keeps the same
    structure post-migration.

    Attributes:
        frame: Last captured RGB frame (uint8 ndarray) — preserves the
            legacy bare-array consumer's data contract.
        frames_captured: Total number of frames captured during this call.
        mean_capture_ms: Mean per-frame elapsed time over ``frames_captured``.
        max_capture_ms: Slowest single-frame elapsed time.
        saved_to: Absolute path of the JPEG snapshot when ``save_path`` was
            requested; ``None`` otherwise (or when the backend lacks both
            ``capture_raw_jpeg`` and ``capture_raw_frame`` — defensive).
    """

    frame: NDArray[np.uint8] | None = None
    frames_captured: int = 1
    mean_capture_ms: float = 0.0
    max_capture_ms: float = 0.0
    saved_to: str | None = None


def camera_unavailable_reason(cfg: Settings, exc: Exception | None = None) -> str | None:
    """Return a skip-worthy reason when the Jetson camera runtime is unavailable."""
    if cfg.camera.backend != "jetson_csi":
        return None

    # Deferred, name-scoped import (rather than a module-level constant copy)
    # so a test's ``monkeypatch.setattr(mousedroid.validation.runtime,
    # "_ARGUS_SOCKET_PATH", ...)`` — which patches the package's own
    # attribute — is observed here even though this helper now lives in a
    # submodule.
    from mousedroid.validation.runtime import _ARGUS_SOCKET_PATH

    reasons: list[str] = []
    device_path = str(cfg.camera.device_path).strip()
    if device_path and not Path(device_path).exists():
        reasons.append(f"V4L2 device {device_path} is missing")
    if not Path(_ARGUS_SOCKET_PATH).exists():
        reasons.append(f"libargus socket {_ARGUS_SOCKET_PATH} is missing")

    if not reasons:
        return None

    if exc is not None and str(exc).strip():
        reasons.append(str(exc).strip())
    return "; ".join(reasons)


async def capture_camera_frame(
    cfg: Settings,
    *,
    save_path: Path | None = None,
    frames: int = 1,
) -> tuple[CameraFrameDiagnostics, str]:
    """Capture one or more raw frames through the configured camera factory.

    Returns a :class:`CameraFrameDiagnostics` instance carrying the LAST
    captured frame, per-call timing aggregates, and (optionally) the
    on-disk path of a JPEG snapshot. The 2-tuple ``(diagnostics, backend_name)``
    return shape preserves the legacy destructure pattern in
    ``scripts/verify_sensors.py::check_camera`` so the existing consumer
    keeps working with the new dataclass slot.

    Args:
        cfg: Fully resolved settings.
        save_path: Optional path the LAST captured frame's JPEG bytes are
            written to. When ``None`` (default), no on-disk artifact is
            created — legacy callers see byte-identical behaviour. JPEG
            encoding uses the camera's ``capture_raw_jpeg()`` method when
            available (zero-copy fast path on backends that natively
            produce JPEG); otherwise the uint8 RGB frame is encoded via
            Pillow (already a project dep via the ``[telemetry]`` /
            ``[hardware]`` extras — no new top-level dep introduced).
        frames: Number of consecutive frames to capture (default ``1``).
            The LAST frame is the one returned in the diagnostics record
            and (when ``save_path`` is set) the one written to disk.

    Returns:
        ``(CameraFrameDiagnostics, backend_name)``. The diagnostics record
        always has its ``frame`` field populated; ``saved_to`` is non-None
        only when ``save_path`` was supplied and the snapshot was
        successfully written.

    Raises:
        RuntimeError: If the camera driver cannot expose a raw frame.
    """
    # Deferred import — see ``camera_unavailable_reason`` for why this is
    # resolved through the package rather than imported at module level
    # (keeps ``monkeypatch.setattr(runtime, "build_camera", ...)`` live).
    from mousedroid.validation.runtime import build_camera

    n_frames = max(1, frames)
    camera = build_camera(cfg)
    await camera.start()
    try:
        capture_raw = _resolve_raw_frame_capture(camera)

        per_frame_ms: list[float] = []
        last_frame: NDArray[np.uint8] | None = None
        for _ in range(n_frames):
            frame_start = time.monotonic()
            last_frame = np.asarray(await capture_raw(), dtype=np.uint8)
            per_frame_ms.append((time.monotonic() - frame_start) * 1000.0)

        unwrapped = _unwrap_camera(camera)
        backend_name = str(getattr(unwrapped, "_backend", unwrapped.__class__.__name__))

        saved_to: str | None = None
        if save_path is not None and last_frame is not None:
            jpeg_bytes = await _encode_camera_frame_jpeg(camera, last_frame)
            if jpeg_bytes is not None:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                save_path.write_bytes(jpeg_bytes)
                saved_to = str(save_path.resolve())

        diagnostics = CameraFrameDiagnostics(
            frame=last_frame,
            frames_captured=len(per_frame_ms),
            mean_capture_ms=sum(per_frame_ms) / len(per_frame_ms) if per_frame_ms else 0.0,
            max_capture_ms=max(per_frame_ms) if per_frame_ms else 0.0,
            saved_to=saved_to,
        )
        return diagnostics, backend_name
    finally:
        await camera.stop()


def _unwrap_camera(camera: object) -> object:
    """Reach through ``ResilientCamera`` to the concrete driver.

    ``_backend``/``_cfg`` are driver implementation details outside any
    Protocol — the resilience wrapper only re-exports the
    ``VisionProtocol``/``RawFrameSourceProtocol`` surface (plus the
    ``capture_raw_frame`` convention), so introspection unwraps one level
    via the wrapper's public ``inner`` property to keep working against a
    wrapped camera. Drivers built without wrapping (e.g. ``mock_hardware``)
    have no ``inner`` attribute and pass through unchanged.
    """
    inner = getattr(camera, "inner", None)
    return inner if inner is not None else camera


def _resolve_raw_frame_capture(
    camera: object,
) -> Callable[[], Awaitable[NDArray[np.uint8]]]:
    """Return an async callable that yields one uint8 RGB frame.

    Resolution chain (broadest support — covers both production and mock):
      1. ``capture_raw_frame()`` → direct async method on the camera
         (``IMX500Camera`` exposes this; preferred when available).
      2. ``capture_raw_jpeg()`` + Pillow decode → covers ``MockCamera``
         which only exposes the JPEG-encoded path (used by the
         telemetry server's ``/camera/frame.jpg`` endpoint).
      3. ``_capture_frame`` private legacy blocking helper → covers any
         older driver that predates the async public API.

    Raises:
        RuntimeError: When none of the three paths are available.
    """
    capture_raw = getattr(camera, "capture_raw_frame", None)
    if callable(capture_raw):
        # Some legacy drivers expose ``capture_raw_frame`` as a SYNC method
        # (older Jetson CSI shims, test stubs that predate the async API).
        # ``await`` on a non-coroutine raises a confusing ``TypeError:
        # object NoneType can't be used in 'await' expression`` deep
        # inside ``capture_camera_frame`` — wrap with ``asyncio.to_thread``
        # so sync drivers still work and the smoke produces a clean
        # signal.
        if asyncio.iscoroutinefunction(capture_raw):
            return cast("Callable[[], Awaitable[NDArray[np.uint8]]]", capture_raw)

        async def _via_sync_capture_raw_frame() -> NDArray[np.uint8]:
            return np.asarray(await asyncio.to_thread(capture_raw), dtype=np.uint8)

        return _via_sync_capture_raw_frame

    capture_jpeg = getattr(camera, "capture_raw_jpeg", None)
    if callable(capture_jpeg):

        async def _via_jpeg() -> NDArray[np.uint8]:
            from io import BytesIO

            try:
                from PIL import Image
            except ImportError as exc:
                # Pillow lives in ``[hardware]`` / ``[telemetry]`` extras
                # (pyproject.toml lines 35, 51); bare ``[dev]`` CI installs
                # don't get it. Produce a clear operator-actionable error
                # rather than a bare ImportError traceback from the
                # closure call site.
                msg = (
                    "camera exposes only capture_raw_jpeg() but Pillow is "
                    'unavailable; install with `pip install -e ".[hardware]"` '
                    "or `[telemetry]`"
                )
                raise RuntimeError(msg) from exc

            jpeg_bytes = await capture_jpeg()
            if not jpeg_bytes:
                msg = "camera capture_raw_jpeg returned empty payload"
                raise RuntimeError(msg)
            img = Image.open(BytesIO(jpeg_bytes)).convert("RGB")
            return np.asarray(img, dtype=np.uint8)

        return _via_jpeg

    capture_frame = getattr(camera, "_capture_frame", None)
    if callable(capture_frame):

        async def _via_legacy() -> NDArray[np.uint8]:
            return np.asarray(await asyncio.to_thread(capture_frame), dtype=np.uint8)

        return _via_legacy

    msg = "camera driver does not expose raw frame capture"
    raise RuntimeError(msg)


async def _encode_camera_frame_jpeg(
    camera: object,
    fallback_rgb: NDArray[np.uint8],
) -> bytes | None:
    """Return JPEG bytes for one camera frame.

    Resolution chain:
      1. ``camera.capture_raw_jpeg()`` if the backend exposes it
         (``MockCamera`` does; the production ``IMX500Camera`` does not).
      2. Pillow encoding of the supplied ``fallback_rgb`` uint8 ndarray —
         covers backends that only produce raw frames.
      3. ``None`` when Pillow is unavailable (defensive — Pillow is a
         project dep, but operators running with the bare ``[dev]`` extras
         in CI would land here). That degrade emits
         ``camera_jpeg_encode_skipped_no_pillow`` at WARNING so
         ``--save-frame`` producing no file is diagnosable from the log
         rather than looking like a dead camera.
    """
    capture_jpeg = getattr(camera, "capture_raw_jpeg", None)
    if callable(capture_jpeg):
        result = await capture_jpeg()
        if isinstance(result, bytes) and result:
            return result
    try:
        from io import BytesIO

        from PIL import Image
    except ImportError:
        # Deferred import so ``monkeypatch.setattr(runtime, "_log", ...)``
        # (patching the package's own attribute) is observed here.
        from mousedroid.validation.runtime import _log

        _log.warning(
            "camera_jpeg_encode_skipped_no_pillow",
            hint='install the Pillow extra: pip install -e ".[dev,telemetry,mcp]"',
            frame_shape=tuple(fallback_rgb.shape),
        )
        return None

    quality = _snapshot_jpeg_quality(camera)
    img = Image.fromarray(fallback_rgb)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _snapshot_jpeg_quality(camera: object) -> int:
    """Return the JPEG quality used by the fallback Pillow encoder.

    Prefers ``camera._cfg.snapshot_jpeg_quality`` (the schema-driven
    config the camera was built from) when available, falling back to
    ``90`` for stubs / drivers that don't carry the cfg. Operators can
    bump quality to 100 for lossless visual inspection or drop to 70 for
    smaller snapshot files in disk-pressed deployments.
    """
    cfg_obj = getattr(_unwrap_camera(camera), "_cfg", None)
    if cfg_obj is not None:
        value = getattr(cfg_obj, "snapshot_jpeg_quality", None)
        if isinstance(value, int):
            return max(1, min(100, value))
    return 90
