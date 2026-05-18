"""Reusable runtime validation helpers.

These helpers keep Jetson smoke tests and verification scripts aligned with
the same config overlays and factory-backed drivers used by the application.
"""

from __future__ import annotations

import asyncio
import math
import os
import subprocess
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from mousedroid.config.loader import load_settings
from mousedroid.factory import build_camera, build_microphone, build_speaker, build_voice_engine

if TYPE_CHECKING:
    from mousedroid.config.schema import Settings
    from mousedroid.hardware.lidar.ld19_driver import LD19ReadStats
    from mousedroid.sensing.lidar_scan import LidarScan


_CONFIG_LIST_ENV_VARS = ("MOUSEDROID_CONFIGS", "MOUSEDROID_JETSON_CONFIGS")
_CONFIG_SINGLE_ENV_VARS = ("MOUSEDROID_CONFIG", "MOUSEDROID_JETSON_CONFIG")

# Named constants for paths and phrases used in validation helpers.
_ARGUS_SOCKET_PATH: str = "/tmp/argus_socket"  # noqa: S108
_DEFAULT_SMOKE_PHRASE: str = "Hello hello! Rocky ready!"


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


@dataclass(frozen=True)
class PcieSsdDiagnostics:
    """Diagnostics record for one ``verify_pcie_ssd_layout`` probe (Task 2).

    Mirrors the read-only nature of the probe — every field defaults so a
    SKIP path (no NVMe device on bus, etc.) can return a partially-populated
    instance without raising.
    """

    pcie_devices: tuple[str, ...] = ()
    block_devices: tuple[str, ...] = ()
    mount_target: Path | None = None
    free_gb: float = 0.0
    total_gb: float = 0.0
    required_gb: float = 0.0
    configured_paths: dict[str, str] = field(default_factory=dict)
    smartctl_health: str | None = None


@dataclass(frozen=True)
class HailoDiagnostics:
    """Diagnostics record for one ``verify_hailo_accelerator`` probe (Task 3).

    Every SKIP / FAIL branch in the helper returns a fully-formed instance
    so the CLI consumer can emit precise structured output without ``None``
    checks at every field.
    """

    device_path_exists: bool = False
    sdk_importable: bool = False
    device_info: dict[str, str] = field(default_factory=dict)
    hef_files: dict[str, str] = field(default_factory=dict)
    inference_latency_ms: float | None = None
    fallback_on_failure: bool = True


@dataclass(frozen=True)
class LidarScanDiagnostics:
    """Structured diagnostics for one LiDAR scan acquisition."""

    scan_index: int
    n_points: int
    coverage_deg: float
    validation_coverage_deg: float
    largest_gap_deg: float
    largest_gap_start_deg: float | None
    largest_gap_end_deg: float | None
    min_angle_deg: float | None
    max_angle_deg: float | None
    elapsed_s: float
    bytes_read: int
    chunks_read: int
    empty_reads: int
    prefix_hits: int
    header_search_misses: int
    bytes_discarded: int
    parse_failures: int
    crc_failures: int
    frames_parsed: int
    driver_covered_angle_deg: float
    meets_min_coverage: bool


def resolve_runtime_config_paths(
    config_paths: Sequence[Path | str] | None = None,
) -> tuple[Path, ...]:
    """Resolve runtime config overlays from explicit args or environment.

    Precedence:
        1. Explicit ``config_paths`` passed by the caller.
        2. CSV lists in ``MOUSEDROID_CONFIGS`` or ``MOUSEDROID_JETSON_CONFIGS``.
        3. Single-path ``MOUSEDROID_CONFIG`` or legacy ``MOUSEDROID_JETSON_CONFIG``.

    Args:
        config_paths: Explicit config overlay paths.

    Returns:
        Normalized config paths in precedence order.
    """
    resolved = tuple(Path(str(path)) for path in (config_paths or ()) if str(path).strip())
    if resolved:
        return resolved

    for env_var in _CONFIG_LIST_ENV_VARS:
        raw_value = os.getenv(env_var, "").strip()
        if not raw_value:
            continue
        csv_paths = [Path(part.strip()) for part in raw_value.split(",") if part.strip()]
        if csv_paths:
            return tuple(csv_paths)

    for env_var in _CONFIG_SINGLE_ENV_VARS:
        single_path = os.getenv(env_var, "").strip()
        if single_path:
            return (Path(single_path),)

    return ()


def load_runtime_settings(config_paths: Sequence[Path | str] | None = None) -> Settings:
    """Load runtime settings using the resolved config overlay list."""
    resolved_paths = resolve_runtime_config_paths(config_paths)
    return load_settings(*resolved_paths)


def camera_unavailable_reason(cfg: Settings, exc: Exception | None = None) -> str | None:
    """Return a skip-worthy reason when the Jetson camera runtime is unavailable."""
    if cfg.camera.backend != "jetson_csi":
        return None

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

        backend_name = str(getattr(camera, "_backend", camera.__class__.__name__))

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
        return capture_raw  # type: ignore[no-any-return]

    capture_jpeg = getattr(camera, "capture_raw_jpeg", None)
    if callable(capture_jpeg):

        async def _via_jpeg() -> NDArray[np.uint8]:
            from io import BytesIO

            from PIL import Image

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
         in CI would land here).
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
        return None

    img = Image.fromarray(fallback_rgb)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


async def verify_hailo_accelerator(cfg: Settings) -> HailoDiagnostics:
    """Probe the Hailo-8 accelerator and exercise one synthetic inference.

    Uses the same :func:`mousedroid.factory.build_hailo_runtime` factory the
    orchestrator wires, so the smoke exercises the production code path.
    The runtime is always stopped in a ``finally`` block — the PCIe device
    lock MUST be released before this helper returns, otherwise the next
    orchestrator start would fail to acquire the device.

    SKIP branches (not FAIL — Hailo is opt-in extras and the runtime is
    documented to fall back to GPU when unavailable):

    * ``cfg.hailo.enabled is False`` (caller-side guard; the helper still
      returns a populated record so callers can introspect)
    * ``cfg.hailo.device_path`` does not exist as a /dev node
    * ``hailo_platform`` is not importable

    FAIL branches:

    * HEF path is set but the file is missing or fails to load
    * Inference exceeds ``cfg.hailo.timeout_ms``

    Args:
        cfg: Fully resolved settings.

    Returns:
        :class:`HailoDiagnostics` instance.
    """
    device_path = Path(getattr(cfg.hailo, "device_path", "/dev/hailo0"))
    device_path_exists = device_path.exists()

    # SDK importability — does NOT instantiate the runtime yet.
    try:
        import hailo_platform  # noqa: F401  # presence check only
    except ImportError:
        return HailoDiagnostics(
            device_path_exists=device_path_exists,
            sdk_importable=False,
            fallback_on_failure=bool(getattr(cfg.hailo, "fallback_on_failure", True)),
        )

    if not device_path_exists:
        return HailoDiagnostics(
            device_path_exists=False,
            sdk_importable=True,
            fallback_on_failure=bool(getattr(cfg.hailo, "fallback_on_failure", True)),
        )

    # Construct via the factory so we exercise the same code path as production.
    from mousedroid.factory import build_hailo_runtime

    runtime = build_hailo_runtime(cfg)
    if runtime is None:
        return HailoDiagnostics(
            device_path_exists=True,
            sdk_importable=True,
            fallback_on_failure=bool(getattr(cfg.hailo, "fallback_on_failure", True)),
        )

    hef_files: dict[str, str] = {}
    device_info: dict[str, str] = {}
    inference_latency_ms: float | None = None

    try:
        await runtime.start()

        # Best-effort device info — the protocol does not expose this, so we
        # synthesize from the runtime's ``is_available()`` and any private
        # attributes the concrete HailoRuntime happens to surface.
        device_info["available"] = str(bool(runtime.is_available()))
        for attr in ("_device_id", "_fw_version", "_arch"):
            value = getattr(runtime, attr, None)
            if value is not None:
                device_info[attr.lstrip("_")] = str(value)

        # HEF load-status inventory. We read the runtime's _models dict
        # (private but documented as the source of truth post-start) via
        # getattr-reflection so a future protocol that doesn't expose it
        # degrades gracefully to "loaded-or-not-known".
        models = getattr(runtime, "_models", {})
        for role, field_name in (
            ("yolo", "yolo_hef_path"),
            ("feature_extractor", "feature_extractor_hef_path"),
        ):
            hef_path_value = getattr(cfg.hailo, field_name, None)
            if hef_path_value is None or not str(hef_path_value).strip():
                hef_files[role] = "missing (path not configured)"
                continue
            hef_path = Path(hef_path_value)
            if not hef_path.exists():
                hef_files[role] = f"missing ({hef_path})"
                continue
            if role in models:
                hef_files[role] = f"loaded ({hef_path.name})"
            else:
                hef_files[role] = "loaded (model registered)"

        # Synthetic inference — guard input-shape introspection behind a
        # try/except so a Hailo runtime that doesn't expose vstream info
        # still gets a smoke signal via the canonical (640, 640, 3) shape.
        input_shape: tuple[int, ...] = (640, 640, 3)
        yolo_model = models.get("yolo") if isinstance(models, dict) else None
        if yolo_model is not None:
            try:
                vstreams = yolo_model["input_vstream_infos"]
                if vstreams:
                    input_shape = tuple(vstreams[0].shape)
            except (KeyError, IndexError, AttributeError, TypeError):
                pass

        if "yolo" in models:
            zero_img: NDArray[np.uint8] = np.zeros(input_shape, dtype=np.uint8)
            t0 = time.perf_counter()
            try:
                runtime.infer_sync("yolo", zero_img)
                inference_latency_ms = (time.perf_counter() - t0) * 1000.0
            except Exception as exc:  # pragma: no cover - defensive
                hef_files["yolo"] = f"inference_failed ({type(exc).__name__})"

        return HailoDiagnostics(
            device_path_exists=True,
            sdk_importable=True,
            device_info=device_info,
            hef_files=hef_files,
            inference_latency_ms=inference_latency_ms,
            fallback_on_failure=bool(getattr(cfg.hailo, "fallback_on_failure", True)),
        )
    finally:
        # CRITICAL — release the PCIe device lock even if start/infer
        # raised. Log + swallow shutdown errors so they don't mask the
        # original failure visible to the CLI consumer.
        try:
            await runtime.stop()
        except Exception as stop_exc:  # pragma: no cover - defensive
            import logging  # local import — keep validation cold-import light

            logging.getLogger(__name__).warning(
                "hailo_runtime_stop_failed_in_smoke", exc_info=stop_exc
            )


def verify_pcie_ssd_layout(cfg: Settings) -> PcieSsdDiagnostics:
    """Probe the NVMe SSD on PCIe and assert capacity for the configured paths.

    Read-only probe — never writes, never mutates kernel state. All probes
    use stdlib (``subprocess`` + ``shutil``) so the helper has no extra
    dependencies. SKIP paths (missing tools, no device) populate the
    returned dataclass with safe defaults so the CLI can convert any field
    to PASS / SKIP / FAIL deterministically.

    Resolution chain for the mount target:
      1. ``$MOUSEDROID_SSD_MOUNT`` env override if set + exists.
      2. ``findmnt -no TARGET /dev/nvme0n1p1`` when the block device exists.
      3. The parent dir of ``cfg.experience.path`` if it exists on disk.
      4. ``None`` -> the SKIP branch in the CLI.

    ``required_gb`` is derived from ``cfg.experience.map_size_gb`` (the
    LMDB preallocation) — that's the largest contiguous on-disk
    allocation the runtime makes, so it's the right capacity gate.

    Args:
        cfg: Fully resolved settings — read-only.

    Returns:
        Populated :class:`PcieSsdDiagnostics` instance.
    """
    import shutil

    # 1. PCIe device enumeration (best-effort; missing lspci -> empty list).
    pcie_devices: tuple[str, ...] = ()
    if shutil.which("lspci"):
        try:
            result = subprocess.run(
                ["lspci", "-nn"],  # noqa: S607 - same source
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            if result.returncode == 0:
                pcie_devices = tuple(
                    line.strip()
                    for line in result.stdout.splitlines()
                    if "nvme" in line.lower() or "non-volatile memory" in line.lower()
                )
        except (subprocess.TimeoutExpired, OSError):
            pass

    # 2. Block-device enumeration via lsblk.
    block_devices: tuple[str, ...] = ()
    if shutil.which("lsblk"):
        try:
            result = subprocess.run(
                ["lsblk", "-d", "-o", "NAME,SIZE,TYPE,TRAN", "-n"],  # noqa: S607
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            if result.returncode == 0:
                block_devices = tuple(
                    line.strip() for line in result.stdout.splitlines() if "nvme" in line.lower()
                )
        except (subprocess.TimeoutExpired, OSError):
            pass

    # 3. Mount-target resolution chain.
    mount_target = _resolve_pcie_ssd_mount(cfg)

    # 4. Capacity probe via shutil.disk_usage (stdlib, cross-platform).
    free_gb = 0.0
    total_gb = 0.0
    if mount_target is not None and mount_target.exists():
        try:
            usage = shutil.disk_usage(mount_target)
            free_gb = usage.free / (1024**3)
            total_gb = usage.total / (1024**3)
        except OSError:
            pass

    # 5. SMART health (optional — missing smartctl is a SKIP, not a FAIL).
    smartctl_health: str | None = None
    if shutil.which("smartctl"):
        try:
            result = subprocess.run(
                ["smartctl", "-H", "/dev/nvme0n1"],  # noqa: S607
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if "overall-health" in line.lower():
                        smartctl_health = line.split(":")[-1].strip()
                        break
        except (subprocess.TimeoutExpired, OSError):
            pass

    # 6. Configured-path inventory.
    configured_paths = _collect_configured_runtime_paths(cfg)

    return PcieSsdDiagnostics(
        pcie_devices=pcie_devices,
        block_devices=block_devices,
        mount_target=mount_target,
        free_gb=free_gb,
        total_gb=total_gb,
        required_gb=float(cfg.experience.map_size_gb),
        configured_paths=configured_paths,
        smartctl_health=smartctl_health,
    )


def _resolve_pcie_ssd_mount(cfg: Settings) -> Path | None:
    """Resolution chain for the NVMe mount target (see public helper docstring)."""
    env_mount = os.environ.get("MOUSEDROID_SSD_MOUNT", "").strip()
    if env_mount:
        candidate = Path(env_mount)
        if candidate.exists():
            return candidate

    # findmnt against the canonical first NVMe partition.
    import shutil

    if shutil.which("findmnt"):
        try:
            result = subprocess.run(
                ["findmnt", "-no", "TARGET", "/dev/nvme0n1p1"],  # noqa: S607
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            if result.returncode == 0:
                target = result.stdout.strip()
                if target:
                    candidate = Path(target)
                    if candidate.exists():
                        return candidate
        except (subprocess.TimeoutExpired, OSError):
            pass

    # Last-resort: infer from the LMDB path's parent.
    exp_path = Path(cfg.experience.path)
    if exp_path.parent.exists():
        return exp_path.parent

    return None


def _collect_configured_runtime_paths(cfg: Settings) -> dict[str, str]:
    """Build a mapping of schema-field -> resolved absolute path string.

    Schema fields covered (in the order they appear in ``Settings``):

    * ``experience.path`` — LMDB writer destination
    * ``jetson.tensorrt_cache_dir`` — compiled engine cache
    * ``cloud.weight_update.cache_dir`` — OTA download staging
    * ``harness.journal.path`` — operation journal (when configured)
    """
    paths: dict[str, str] = {}
    paths["experience.path"] = str(Path(cfg.experience.path).resolve())

    jetson_cfg = getattr(cfg, "jetson", None)
    if jetson_cfg is not None:
        cache_dir = getattr(jetson_cfg, "tensorrt_cache_dir", None)
        if cache_dir is not None:
            paths["jetson.tensorrt_cache_dir"] = str(Path(cache_dir).resolve())

    cloud_cfg = getattr(cfg, "cloud", None)
    weight_update_cfg = getattr(cloud_cfg, "weight_update", None) if cloud_cfg else None
    cache_dir = getattr(weight_update_cfg, "cache_dir", None) if weight_update_cfg else None
    if cache_dir is not None:
        paths["cloud.weight_update.cache_dir"] = str(Path(cache_dir).resolve())

    harness_cfg = getattr(cfg, "harness", None)
    journal_cfg = getattr(harness_cfg, "journal", None) if harness_cfg else None
    journal_path = getattr(journal_cfg, "path", None) if journal_cfg else None
    if journal_path is not None:
        paths["harness.journal.path"] = str(Path(journal_path).resolve())

    return paths


async def capture_microphone_chunk(cfg: Settings) -> NDArray[np.float32] | None:
    """Capture one chunk through the configured microphone driver.

    Returns ``None`` when the microphone is disabled in config.

    Args:
        cfg: Fully resolved settings.

    Returns:
        Captured audio chunk, or ``None`` when disabled.

    Raises:
        RuntimeError: If the configured microphone cannot open its runtime stream.
    """
    microphone = build_microphone(cfg)
    if microphone is None:
        return None

    await microphone.start()
    try:
        if getattr(microphone, "_stream", object()) is None:
            msg = "configured microphone device unavailable"
            raise RuntimeError(msg)
        chunk = await microphone.read_chunk()
        return np.asarray(chunk, dtype=np.float32)
    finally:
        await microphone.stop()


async def read_lidar_scan(cfg: Settings) -> LidarScan | None:
    """Read a single LiDAR scan through the configured factory.

    Args:
        cfg: Fully resolved settings.

    Returns:
        Captured scan, or ``None`` when LiDAR is disabled.
    """
    from mousedroid.factory import build_lidar

    lidar = build_lidar(cfg)
    if lidar is None:
        return None

    await lidar.start()
    try:
        return await lidar.read_scan()
    finally:
        await lidar.stop()


def lidar_scan_coverage_deg(scan: LidarScan) -> float:
    """Estimate angular coverage for a LiDAR scan.

    The coverage is computed as the complement of the largest angular gap,
    which handles scans that wrap around 0 degrees.

    Args:
        scan: Captured LiDAR scan.

    Returns:
        Angular coverage in degrees in the inclusive range ``[0, 360]``.
    """
    largest_gap_deg, _, _ = lidar_scan_largest_gap_deg(scan)
    return max(0.0, 360.0 - largest_gap_deg)


def lidar_scan_validation_coverage_deg(
    scan: LidarScan,
    *,
    driver_covered_angle_deg: float | None = None,
) -> float:
    """Return the coverage metric used by smoke and runtime validation.

    Filtered point coverage can under-report healthy scans in sparse environments,
    so validation prefers the driver's frame coverage when available.

    Args:
        scan: Captured LiDAR scan.
        driver_covered_angle_deg: Frame-based coverage reported by the driver.

    Returns:
        Coverage in degrees suitable for validation thresholds.
    """
    point_coverage_deg = lidar_scan_coverage_deg(scan)
    if driver_covered_angle_deg is None:
        return point_coverage_deg

    return max(point_coverage_deg, max(0.0, float(driver_covered_angle_deg)))


def lidar_scan_largest_gap_deg(scan: LidarScan) -> tuple[float, float | None, float | None]:
    """Return the largest angular gap and its bounding angles.

    Args:
        scan: Captured LiDAR scan.

    Returns:
        Tuple of ``(largest_gap_deg, gap_start_deg, gap_end_deg)``.
        When fewer than two points are present, the gap defaults to a full
        360-degree blind spot with unknown bounds.
    """
    if scan.n_points < 2:
        return 360.0, None, None

    angles_deg = np.sort(np.asarray(scan.angles_deg, dtype=np.float32))
    wrapped_angles_deg = np.concatenate((angles_deg, angles_deg[:1] + 360.0))
    gap_sizes_deg = np.diff(wrapped_angles_deg)
    gap_idx = int(np.argmax(gap_sizes_deg))
    return (
        float(gap_sizes_deg[gap_idx]),
        float(wrapped_angles_deg[gap_idx] % 360.0),
        float(wrapped_angles_deg[gap_idx + 1] % 360.0),
    )


async def collect_lidar_diagnostics(
    cfg: Settings,
    *,
    n_scans: int = 1,
) -> list[LidarScanDiagnostics]:
    """Collect repeated LiDAR scan diagnostics through the configured driver."""
    from mousedroid.factory import build_lidar

    if n_scans <= 0:
        return []
    if cfg.lidar is None or not cfg.lidar.enabled:
        return []

    lidar = build_lidar(cfg)
    if lidar is None:
        return []

    read_with_diagnostics = getattr(lidar, "read_scan_with_diagnostics", None)
    diagnostics: list[LidarScanDiagnostics] = []

    await lidar.start()
    try:
        for scan_index in range(n_scans):
            started_at = time.monotonic()
            read_stats: LD19ReadStats | None = None

            if callable(read_with_diagnostics):
                scan, read_stats = await read_with_diagnostics()
            else:
                scan = await lidar.read_scan()

            largest_gap_deg, gap_start_deg, gap_end_deg = lidar_scan_largest_gap_deg(scan)
            coverage_deg = lidar_scan_coverage_deg(scan)
            driver_covered_angle_deg = float(getattr(read_stats, "covered_angle_deg", 0.0))
            validation_coverage_deg = lidar_scan_validation_coverage_deg(
                scan,
                driver_covered_angle_deg=(
                    driver_covered_angle_deg if driver_covered_angle_deg > 0.0 else None
                ),
            )
            diagnostics.append(
                LidarScanDiagnostics(
                    scan_index=scan_index,
                    n_points=scan.n_points,
                    coverage_deg=coverage_deg,
                    validation_coverage_deg=validation_coverage_deg,
                    largest_gap_deg=largest_gap_deg,
                    largest_gap_start_deg=gap_start_deg,
                    largest_gap_end_deg=gap_end_deg,
                    min_angle_deg=(float(np.min(scan.angles_deg)) if scan.n_points else None),
                    max_angle_deg=(float(np.max(scan.angles_deg)) if scan.n_points else None),
                    elapsed_s=float(
                        getattr(read_stats, "elapsed_s", time.monotonic() - started_at),
                    ),
                    bytes_read=int(getattr(read_stats, "bytes_read", 0)),
                    chunks_read=int(getattr(read_stats, "chunks_read", 0)),
                    empty_reads=int(getattr(read_stats, "empty_reads", 0)),
                    prefix_hits=int(getattr(read_stats, "prefix_hits", 0)),
                    header_search_misses=int(getattr(read_stats, "header_search_misses", 0)),
                    bytes_discarded=int(getattr(read_stats, "bytes_discarded", 0)),
                    parse_failures=int(getattr(read_stats, "parse_failures", 0)),
                    crc_failures=int(getattr(read_stats, "crc_failures", 0)),
                    frames_parsed=int(getattr(read_stats, "frames_parsed", 0)),
                    driver_covered_angle_deg=driver_covered_angle_deg,
                    meets_min_coverage=validation_coverage_deg >= cfg.lidar.min_scan_coverage_deg,
                ),
            )
    finally:
        await lidar.stop()

    return diagnostics


async def play_speaker_tone(
    cfg: Settings,
    *,
    duration_s: float = 0.3,
    frequency_hz: float = 440.0,
) -> int | None:
    """Play a short tone through the configured speaker driver.

    Args:
        cfg: Fully resolved settings.
        duration_s: Tone duration.
        frequency_hz: Tone frequency.

    Returns:
        Total number of interleaved samples written (``frames * channels``),
        or ``None`` when the speaker is disabled.

    Raises:
        RuntimeError: If the configured speaker cannot open its runtime stream.
    """
    speaker = build_speaker(cfg)
    if speaker is None:
        return None

    await speaker.start()
    try:
        if getattr(speaker, "_stream", object()) is None:
            msg = "configured speaker device unavailable"
            raise RuntimeError(msg)

        channels = max(1, int(getattr(speaker, "channels", 1)))
        min_frames = max(1, round(float(speaker.sample_rate) * duration_s))
        total_frames = max(
            speaker.chunk_size,
            math.ceil(min_frames / speaker.chunk_size) * speaker.chunk_size,
        )
        time_axis = np.arange(total_frames, dtype=np.float32) / float(speaker.sample_rate)
        mono_tone = (0.2 * np.sin(2.0 * np.pi * frequency_hz * time_axis)).astype(np.float32)
        # Interleave identical tone across channels so each frame is `channels` samples.
        interleaved = np.repeat(mono_tone, channels) if channels > 1 else mono_tone

        samples_per_chunk = speaker.chunk_size * channels
        total_samples = total_frames * channels
        for start in range(0, total_samples, samples_per_chunk):
            chunk = interleaved[start : start + samples_per_chunk]
            if chunk.shape[0] < samples_per_chunk:
                chunk = np.pad(chunk, (0, samples_per_chunk - chunk.shape[0]))
            await speaker.write_chunk(chunk)

        return total_samples
    finally:
        await speaker.stop()


async def play_rocky_voice_phrase(
    cfg: Settings,
    *,
    phrase: str = _DEFAULT_SMOKE_PHRASE,
) -> tuple[int, float] | None:
    """Play a short Rocky voice phrase through the configured voice pipeline.

    Args:
        cfg: Fully resolved settings.
        phrase: Short phrase to synthesize and play.

    Returns:
        Tuple of ``(samples_written, peak_abs_sample)``, or ``None`` when the
        voice engine is disabled.

    Raises:
        RuntimeError: If the voice pipeline cannot load TTS or write to the
            configured speaker.
    """
    if not cfg.voice.enabled:
        return None

    speaker = build_speaker(cfg)
    if speaker is None:
        raise RuntimeError("configured speaker unavailable for Rocky voice")

    engine = build_voice_engine(cfg, speaker=speaker)
    if engine is None:
        raise RuntimeError("Rocky voice engine unavailable")

    await engine.start()
    try:
        samples_written, peak_abs = await engine.play_phrase(phrase)
        if not cfg.mock_hardware and peak_abs <= 1e-6:
            raise RuntimeError("Rocky voice TTS returned silent audio")
        return samples_written, peak_abs
    finally:
        await engine.stop()
