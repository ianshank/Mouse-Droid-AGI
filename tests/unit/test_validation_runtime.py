"""Unit tests for reusable runtime validation helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from mousedroid.config.schema import Settings
from mousedroid.hardware.lidar.ld19_driver import LD19ReadStats
from mousedroid.sensing.lidar_scan import LidarScan
from mousedroid.validation import runtime


def test_resolve_runtime_config_paths_prefers_explicit_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit config arguments override environment-derived paths."""
    monkeypatch.setenv(
        "MOUSEDROID_JETSON_CONFIGS",
        "config/jetson_production.yaml,config/jetson_lidar_only.yaml",
    )

    resolved = runtime.resolve_runtime_config_paths(["config/mock_hardware.yaml"])

    assert resolved == (runtime.Path("config/mock_hardware.yaml"),)


def test_resolve_runtime_config_paths_uses_env_csv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CSV env vars are resolved in order when no explicit args are passed."""
    monkeypatch.delenv("MOUSEDROID_CONFIG", raising=False)
    monkeypatch.setenv(
        "MOUSEDROID_JETSON_CONFIGS",
        "config/jetson_production.yaml, config/jetson_lidar_only.yaml",
    )

    resolved = runtime.resolve_runtime_config_paths()

    assert resolved == (
        runtime.Path("config/jetson_production.yaml"),
        runtime.Path("config/jetson_lidar_only.yaml"),
    )


def test_resolve_runtime_config_paths_legacy_jetson_single_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy ``MOUSEDROID_JETSON_CONFIG`` single-path var is honored."""
    for var in ("MOUSEDROID_CONFIGS", "MOUSEDROID_JETSON_CONFIGS", "MOUSEDROID_CONFIG"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("MOUSEDROID_JETSON_CONFIG", "config/jetson_production.yaml")

    resolved = runtime.resolve_runtime_config_paths()

    assert resolved == (runtime.Path("config/jetson_production.yaml"),)


def test_camera_unavailable_reason_reports_missing_jetson_runtime(tmp_path: Path) -> None:
    cfg = Settings(
        mock_hardware=True,
        camera={
            "backend": "jetson_csi",
            "device_path": str(tmp_path / "video0"),
        },
    )

    reason = runtime.camera_unavailable_reason(
        cfg,
        RuntimeError("Failed to open CSI camera via GStreamer pipeline or V4L2 device"),
    )

    assert reason is not None
    assert "V4L2 device" in reason
    assert "Failed to open CSI camera" in reason


def test_camera_unavailable_reason_non_jetson_returns_none() -> None:
    """Non-jetson_csi backend always returns None."""
    cfg = Settings(mock_hardware=True, camera={"backend": "auto"})
    assert runtime.camera_unavailable_reason(cfg) is None
    assert runtime.camera_unavailable_reason(cfg, RuntimeError("err")) is None


def test_camera_unavailable_reason_returns_none_when_all_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Returns None when both V4L2 device and argus socket exist."""
    device = tmp_path / "video0"
    device.touch()
    argus = tmp_path / "argus_socket"
    argus.touch()
    monkeypatch.setattr(runtime, "_ARGUS_SOCKET_PATH", str(argus))

    cfg = Settings(
        mock_hardware=True,
        camera={"backend": "jetson_csi", "device_path": str(device)},
    )
    assert runtime.camera_unavailable_reason(cfg) is None


def test_camera_unavailable_reason_empty_device_path_skips_v4l2_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When device_path is empty, only the argus socket check runs."""
    # Point argus socket constant to a path that does NOT exist
    monkeypatch.setattr(runtime, "_ARGUS_SOCKET_PATH", str(tmp_path / "argus_socket"))

    cfg = Settings(
        mock_hardware=True,
        camera={"backend": "jetson_csi", "device_path": ""},
    )
    reason = runtime.camera_unavailable_reason(cfg)

    assert reason is not None
    assert "V4L2" not in reason
    assert "argus_socket" in reason


def test_camera_unavailable_reason_only_v4l2_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When argus socket exists but V4L2 device is absent, only device reason returned."""
    argus = tmp_path / "argus_socket"
    argus.touch()
    monkeypatch.setattr(runtime, "_ARGUS_SOCKET_PATH", str(argus))

    cfg = Settings(
        mock_hardware=True,
        camera={"backend": "jetson_csi", "device_path": str(tmp_path / "video0")},
    )
    reason = runtime.camera_unavailable_reason(cfg, exc=None)

    assert reason is not None
    assert "V4L2" in reason
    assert "argus_socket" not in reason


@pytest.mark.asyncio
async def test_capture_camera_frame_uses_runtime_driver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Camera helper should prefer public capture_raw_frame and lifecycle the driver."""

    class StubCamera:
        def __init__(self) -> None:
            self._backend = "gstreamer"
            self.started = False
            self.stopped = False

        async def start(self) -> None:
            self.started = True

        async def stop(self) -> None:
            self.stopped = True

        async def capture_raw_frame(self) -> np.ndarray:
            return np.ones((4, 5, 3), dtype=np.uint8)

    stub = StubCamera()
    monkeypatch.setattr(runtime, "build_camera", lambda cfg: stub)

    diagnostics, backend_name = await runtime.capture_camera_frame(Settings(mock_hardware=True))

    # capture_camera_frame now returns (CameraFrameDiagnostics, backend_name)
    # — the 2-tuple shape is preserved; the frame is in .frame on the dataclass.
    assert diagnostics.frame is not None
    assert diagnostics.frame.shape == (4, 5, 3)
    assert diagnostics.frames_captured == 1
    assert diagnostics.saved_to is None  # no save_path supplied
    assert backend_name == "gstreamer"
    assert stub.started is True
    assert stub.stopped is True


@pytest.mark.asyncio
async def test_capture_camera_frame_falls_back_to_private_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Camera helper falls back to legacy _capture_frame when public method absent."""

    class LegacyStubCamera:
        def __init__(self) -> None:
            self._backend = "v4l2"
            self.started = False
            self.stopped = False

        async def start(self) -> None:
            self.started = True

        async def stop(self) -> None:
            self.stopped = True

        def _capture_frame(self) -> np.ndarray:
            return np.zeros((3, 6, 3), dtype=np.uint8)

    stub = LegacyStubCamera()
    monkeypatch.setattr(runtime, "build_camera", lambda cfg: stub)

    diagnostics, backend_name = await runtime.capture_camera_frame(Settings(mock_hardware=True))

    assert diagnostics.frame is not None
    assert diagnostics.frame.shape == (3, 6, 3)
    assert backend_name == "v4l2"
    assert stub.started is True
    assert stub.stopped is True


@pytest.mark.asyncio
async def test_capture_camera_frame_captures_multiple_frames_and_saves_jpeg(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``--save-frame`` + ``--frames N`` round-trip — Task 1 acceptance regression."""

    class JpegStubCamera:
        """Stub exposing ``capture_raw_frame`` so the JPEG fallback (Pillow encode) fires."""

        def __init__(self) -> None:
            self._backend = "jpegstub"
            self.capture_count = 0

        async def start(self) -> None:
            return None

        async def stop(self) -> None:
            return None

        async def capture_raw_frame(self) -> np.ndarray:
            self.capture_count += 1
            return np.full((8, 12, 3), self.capture_count, dtype=np.uint8)

    stub = JpegStubCamera()
    monkeypatch.setattr(runtime, "build_camera", lambda cfg: stub)

    snap = tmp_path / "snap.jpg"
    diagnostics, backend_name = await runtime.capture_camera_frame(
        Settings(mock_hardware=True),
        save_path=snap,
        frames=3,
    )

    assert backend_name == "jpegstub"
    assert diagnostics.frames_captured == 3
    assert stub.capture_count == 3
    # The LAST frame's data should be in the diagnostics (value=3 from the
    # third call to capture_raw_frame).
    assert diagnostics.frame is not None
    assert int(diagnostics.frame[0, 0, 0]) == 3
    # JPEG snapshot landed on disk.
    assert diagnostics.saved_to is not None
    assert snap.exists()
    assert snap.stat().st_size > 0
    # And it round-trips through Pillow.
    from PIL import Image

    with Image.open(snap) as img:
        assert img.size == (12, 8)  # (width, height)
        assert img.format == "JPEG"


@pytest.mark.asyncio
async def test_capture_microphone_chunk_requires_live_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Microphone helper should fail when the runtime stream never opens."""

    class StubMicrophone:
        def __init__(self) -> None:
            self._stream = None
            self.stopped = False

        async def start(self) -> None:
            return None

        async def stop(self) -> None:
            self.stopped = True

        async def read_chunk(self) -> np.ndarray:
            return np.zeros(4, dtype=np.float32)

    stub = StubMicrophone()
    monkeypatch.setattr(runtime, "build_microphone", lambda cfg: stub)

    with pytest.raises(RuntimeError, match="configured microphone device unavailable"):
        await runtime.capture_microphone_chunk(
            Settings(mock_hardware=True, microphone={"enabled": True, "chunk_size": 4}),
        )

    assert stub.stopped is True


@pytest.mark.asyncio
async def test_play_speaker_tone_writes_chunked_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Speaker helper should chunk and write audio using the runtime driver."""

    class StubSpeaker:
        def __init__(self) -> None:
            self._stream = object()
            self.sample_rate = 1000
            self.chunk_size = 4
            self.channels = 1
            self.writes: list[np.ndarray] = []
            self.stopped = False

        async def start(self) -> None:
            return None

        async def stop(self) -> None:
            self.stopped = True

        async def write_chunk(self, samples: np.ndarray) -> None:
            self.writes.append(samples.copy())

    stub = StubSpeaker()
    monkeypatch.setattr(runtime, "build_speaker", lambda cfg: stub)

    written = await runtime.play_speaker_tone(
        Settings(mock_hardware=True, speaker={"enabled": True, "sample_rate": 1000}),
        duration_s=0.01,
        frequency_hz=110.0,
    )

    assert written == 12
    assert len(stub.writes) == 3
    assert all(chunk.shape == (4,) for chunk in stub.writes)
    assert stub.stopped is True


@pytest.mark.asyncio
async def test_play_speaker_tone_stereo_chunks_match_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multi-channel speakers must receive ``chunk_size * channels`` samples per write."""

    class StubStereoSpeaker:
        def __init__(self) -> None:
            self._stream = object()
            self.sample_rate = 1000
            self.chunk_size = 4
            self.channels = 2
            self.writes: list[np.ndarray] = []
            self.stopped = False

        async def start(self) -> None:
            return None

        async def stop(self) -> None:
            self.stopped = True

        async def write_chunk(self, samples: np.ndarray) -> None:
            self.writes.append(samples.copy())

    stub = StubStereoSpeaker()
    monkeypatch.setattr(runtime, "build_speaker", lambda cfg: stub)

    written = await runtime.play_speaker_tone(
        Settings(
            mock_hardware=True,
            speaker={"enabled": True, "sample_rate": 1000, "channels": 2},
        ),
        duration_s=0.01,
        frequency_hz=110.0,
    )

    # 12 frames * 2 channels = 24 interleaved samples, 3 chunks of 8 samples each.
    assert written == 24
    assert len(stub.writes) == 3
    assert all(chunk.shape == (8,) for chunk in stub.writes)
    # Left and right channels should carry identical interleaved samples.
    for chunk in stub.writes:
        assert np.allclose(chunk[0::2], chunk[1::2])
    assert stub.stopped is True


@pytest.mark.asyncio
async def test_play_rocky_voice_phrase_uses_public_engine_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime voice helper should use the engine's public playback API."""

    class StubSpeaker:
        async def start(self) -> None:
            return None

        async def stop(self) -> None:
            return None

    class StubVoiceEngine:
        def __init__(self) -> None:
            self.started = False
            self.stopped = False
            self.played: list[str] = []

        async def start(self) -> None:
            self.started = True

        async def stop(self) -> None:
            self.stopped = True

        async def play_phrase(self, text: str) -> tuple[int, float]:
            self.played.append(text)
            return 123, 0.42

    stub_speaker = StubSpeaker()
    stub_engine = StubVoiceEngine()
    monkeypatch.setattr(runtime, "build_speaker", lambda cfg: stub_speaker)
    monkeypatch.setattr(runtime, "build_voice_engine", lambda cfg, speaker=None: stub_engine)

    result = await runtime.play_rocky_voice_phrase(
        Settings(mock_hardware=True, voice={"enabled": True, "tts_sample_rate": 22050}),
        phrase="Rocky test phrase",
    )

    assert result == (123, 0.42)
    assert stub_engine.started is True
    assert stub_engine.stopped is True
    assert stub_engine.played == ["Rocky test phrase"]


@pytest.mark.asyncio
async def test_play_rocky_voice_phrase_returns_none_when_voice_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Returns None immediately when voice is disabled in config."""
    cfg = Settings(mock_hardware=True, voice={"enabled": False})
    result = await runtime.play_rocky_voice_phrase(cfg)
    assert result is None


@pytest.mark.asyncio
async def test_play_rocky_voice_phrase_raises_when_speaker_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raises RuntimeError when build_speaker returns None."""
    monkeypatch.setattr(runtime, "build_speaker", lambda cfg: None)
    cfg = Settings(mock_hardware=True, voice={"enabled": True})

    with pytest.raises(RuntimeError, match="speaker unavailable"):
        await runtime.play_rocky_voice_phrase(cfg)


@pytest.mark.asyncio
async def test_play_rocky_voice_phrase_raises_when_engine_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raises RuntimeError when build_voice_engine returns None."""

    class StubSpeaker:
        async def start(self) -> None:
            return None

        async def stop(self) -> None:
            return None

    monkeypatch.setattr(runtime, "build_speaker", lambda cfg: StubSpeaker())
    monkeypatch.setattr(runtime, "build_voice_engine", lambda cfg, speaker=None: None)
    cfg = Settings(mock_hardware=True, voice={"enabled": True})

    with pytest.raises(RuntimeError, match="voice engine unavailable"):
        await runtime.play_rocky_voice_phrase(cfg)


@pytest.mark.asyncio
async def test_play_rocky_voice_phrase_uses_default_phrase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When no phrase is supplied the default smoke phrase is used."""

    class StubSpeaker:
        async def start(self) -> None:
            return None

        async def stop(self) -> None:
            return None

    class StubVoiceEngine:
        def __init__(self) -> None:
            self.played: list[str] = []

        async def start(self) -> None:
            return None

        async def stop(self) -> None:
            return None

        async def play_phrase(self, text: str) -> tuple[int, float]:
            self.played.append(text)
            return 50, 0.1

    stub_engine = StubVoiceEngine()
    monkeypatch.setattr(runtime, "build_speaker", lambda cfg: StubSpeaker())
    monkeypatch.setattr(runtime, "build_voice_engine", lambda cfg, speaker=None: stub_engine)

    await runtime.play_rocky_voice_phrase(Settings(mock_hardware=True, voice={"enabled": True}))

    assert stub_engine.played == [runtime._DEFAULT_SMOKE_PHRASE]


def test_lidar_scan_coverage_deg_measures_partial_span() -> None:
    scan = LidarScan(
        angles_deg=np.array([10.0, 20.0, 30.0], dtype=np.float32),
        distances_mm=np.array([1000.0, 1000.0, 1000.0], dtype=np.float32),
        confidences=np.array([100, 100, 100], dtype=np.uint8),
        timestamp=0.0,
        n_points=3,
    )

    coverage_deg = runtime.lidar_scan_coverage_deg(scan)

    assert coverage_deg == pytest.approx(20.0)


def test_lidar_scan_coverage_deg_handles_wraparound() -> None:
    scan = LidarScan(
        angles_deg=np.array([350.0, 355.0, 0.0, 5.0], dtype=np.float32),
        distances_mm=np.array([1000.0, 1000.0, 1000.0, 1000.0], dtype=np.float32),
        confidences=np.array([100, 100, 100, 100], dtype=np.uint8),
        timestamp=0.0,
        n_points=4,
    )

    coverage_deg = runtime.lidar_scan_coverage_deg(scan)

    assert coverage_deg == pytest.approx(15.0)


def test_lidar_scan_validation_coverage_prefers_driver_coverage() -> None:
    scan = LidarScan(
        angles_deg=np.array([350.0, 355.0, 0.0, 5.0], dtype=np.float32),
        distances_mm=np.array([1000.0, 1000.0, 1000.0, 1000.0], dtype=np.float32),
        confidences=np.array([100, 100, 100, 100], dtype=np.uint8),
        timestamp=0.0,
        n_points=4,
    )

    coverage_deg = runtime.lidar_scan_validation_coverage_deg(
        scan,
        driver_covered_angle_deg=275.0,
    )

    assert coverage_deg == pytest.approx(275.0)


def test_lidar_scan_largest_gap_deg_reports_gap_window() -> None:
    scan = LidarScan(
        angles_deg=np.array([0.0, 10.0, 200.0], dtype=np.float32),
        distances_mm=np.array([1000.0, 1000.0, 1000.0], dtype=np.float32),
        confidences=np.array([100, 100, 100], dtype=np.uint8),
        timestamp=0.0,
        n_points=3,
    )

    largest_gap_deg, gap_start_deg, gap_end_deg = runtime.lidar_scan_largest_gap_deg(scan)

    assert largest_gap_deg == pytest.approx(190.0)
    assert gap_start_deg == pytest.approx(10.0)
    assert gap_end_deg == pytest.approx(200.0)


@pytest.mark.asyncio
async def test_collect_lidar_diagnostics_uses_driver_stats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StubLidar:
        def __init__(self) -> None:
            self.started = False
            self.stopped = False

        async def start(self) -> None:
            self.started = True

        async def stop(self) -> None:
            self.stopped = True

        async def read_scan_with_diagnostics(self) -> tuple[LidarScan, LD19ReadStats]:
            scan = LidarScan(
                angles_deg=np.array([350.0, 355.0, 0.0, 5.0], dtype=np.float32),
                distances_mm=np.array([1000.0, 1000.0, 1000.0, 1000.0], dtype=np.float32),
                confidences=np.array([100, 100, 100, 100], dtype=np.uint8),
                timestamp=0.0,
                n_points=4,
            )
            return scan, LD19ReadStats(
                bytes_read=188,
                chunks_read=1,
                empty_reads=0,
                prefix_hits=2,
                header_search_misses=0,
                bytes_discarded=3,
                parse_failures=1,
                crc_failures=1,
                frames_parsed=2,
                covered_angle_deg=275.0,
                elapsed_s=0.05,
            )

    stub = StubLidar()
    monkeypatch.setattr("mousedroid.factory.build_lidar", lambda cfg: stub)

    diagnostics = await runtime.collect_lidar_diagnostics(
        Settings(mock_hardware=True, lidar={"enabled": True}),
        n_scans=2,
    )

    assert len(diagnostics) == 2
    assert diagnostics[0].coverage_deg == pytest.approx(15.0)
    assert diagnostics[0].validation_coverage_deg == pytest.approx(275.0)
    assert diagnostics[0].largest_gap_start_deg == pytest.approx(5.0)
    assert diagnostics[0].largest_gap_end_deg == pytest.approx(350.0)
    assert diagnostics[0].parse_failures == 1
    assert diagnostics[0].crc_failures == 1
    assert diagnostics[0].bytes_read == 188
    assert diagnostics[0].meets_min_coverage is True
    assert stub.started is True
    assert stub.stopped is True


# ---------------------------------------------------------------------------
# Task 2 — PCIe NVMe SSD smoke (verify_pcie_ssd_layout)
# ---------------------------------------------------------------------------


def test_verify_pcie_ssd_layout_skips_when_no_tools_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing lspci/lsblk/findmnt/smartctl → SKIP-style empty result, not FAIL."""
    monkeypatch.setattr("shutil.which", lambda cmd: None)

    cfg = Settings(mock_hardware=True)
    result = runtime.verify_pcie_ssd_layout(cfg)

    assert result.pcie_devices == ()
    assert result.block_devices == ()
    assert result.smartctl_health is None
    assert result.required_gb == pytest.approx(cfg.experience.map_size_gb)
    # configured_paths populated from cfg regardless of tool availability.
    assert "experience.path" in result.configured_paths


def test_verify_pcie_ssd_layout_uses_env_mount_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``$MOUSEDROID_SSD_MOUNT`` env var wins over findmnt + path-parent inference."""
    custom_mount = tmp_path / "ssd_mount"
    custom_mount.mkdir()
    monkeypatch.setenv("MOUSEDROID_SSD_MOUNT", str(custom_mount))
    monkeypatch.setattr("shutil.which", lambda cmd: None)  # disable subprocess probes

    cfg = Settings(mock_hardware=True)
    result = runtime.verify_pcie_ssd_layout(cfg)

    assert result.mount_target == custom_mount
    assert result.total_gb > 0  # shutil.disk_usage on a real path returns >0


def test_verify_pcie_ssd_layout_skips_mount_when_no_resolution_path_works(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Mount resolution returns ``None`` when env override is unset AND findmnt unavailable.

    Replaces the prior ``..._falls_back_to_experience_path_parent`` test —
    the rootfs-parent fallback was removed in the PR #104 hardening pass
    because it caused a FALSE PASS on rovers with no NVMe at all (the
    parent of ``/home/jetson/mousedroid_experience`` is ``/home/jetson``
    which is the rootfs, NOT the SSD).
    """
    monkeypatch.delenv("MOUSEDROID_SSD_MOUNT", raising=False)
    monkeypatch.setattr("shutil.which", lambda cmd: None)

    exp_path = tmp_path / "lmdb_root" / "experience.lmdb"
    exp_path.parent.mkdir(parents=True)
    cfg = Settings(
        mock_hardware=True,
        experience={"path": str(exp_path)},
    )
    result = runtime.verify_pcie_ssd_layout(cfg)

    # New behaviour: clean SKIP via mount_target=None.
    assert result.mount_target is None
    assert result.free_gb == 0.0
    assert result.total_gb == 0.0


def test_verify_pcie_ssd_layout_collects_configured_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All four documented schema fields land in ``configured_paths`` when present."""
    monkeypatch.setattr("shutil.which", lambda cmd: None)

    cfg = Settings(mock_hardware=True)
    result = runtime.verify_pcie_ssd_layout(cfg)

    paths = result.configured_paths
    assert "experience.path" in paths
    # The other fields land conditionally based on the schema default's
    # presence on the loaded config — at minimum experience.path is always
    # populated, so the test pins that invariant strictly and the others
    # softly (they're populated by default but the schema could change).
    for optional_field in (
        "jetson.tensorrt_cache_dir",
        "cloud.weight_update.cache_dir",
    ):
        if optional_field in paths:
            assert Path(paths[optional_field]).is_absolute()


# ---------------------------------------------------------------------------
# Task 3 — Hailo-8 smoke (verify_hailo_accelerator)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verify_hailo_accelerator_skips_when_device_path_absent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """No ``/dev/hailo0`` → SKIP-style record (device_path_exists=False)."""
    cfg = Settings(
        mock_hardware=True,
        hailo={"enabled": True, "device_path": str(tmp_path / "nonexistent_hailo")},
    )

    # Force hailo_platform to APPEAR importable so we exercise the
    # "SDK present but device missing" branch (the most common operator-
    # facing case post-reseat).
    import sys as _sys
    from types import ModuleType

    fake_hailo = ModuleType("hailo_platform")
    monkeypatch.setitem(_sys.modules, "hailo_platform", fake_hailo)

    result = await runtime.verify_hailo_accelerator(cfg)

    assert result.device_path_exists is False
    assert result.sdk_importable is True
    assert result.inference_latency_ms is None
    assert result.fallback_on_failure is True  # schema default


@pytest.mark.asyncio
async def test_verify_hailo_accelerator_skips_when_sdk_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Missing ``hailo_platform`` import → SKIP record (sdk_importable=False)."""
    cfg = Settings(
        mock_hardware=True,
        hailo={"enabled": True, "device_path": str(tmp_path / "nonexistent_hailo")},
    )

    # Force the import to fail by injecting a sentinel that raises on access.
    import sys as _sys

    monkeypatch.delitem(_sys.modules, "hailo_platform", raising=False)

    class _ImportBlocker:
        def find_module(self, name: str, _path: object = None) -> object | None:
            if name == "hailo_platform":
                return self
            return None

        def find_spec(self, name: str, _path: object = None, _target: object = None) -> None:
            if name == "hailo_platform":
                msg = "blocked for test"
                raise ImportError(msg)
            return None

    blocker = _ImportBlocker()
    _sys.meta_path.insert(0, blocker)
    try:
        result = await runtime.verify_hailo_accelerator(cfg)
    finally:
        _sys.meta_path.remove(blocker)

    assert result.sdk_importable is False
    assert result.inference_latency_ms is None


@pytest.mark.asyncio
async def test_verify_hailo_accelerator_happy_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Full Hailo round-trip: device present, SDK importable, HEFs loaded, infer_sync timed."""
    import sys as _sys
    from types import ModuleType

    # 1. Synthesize a Hailo device path + HEF files.
    fake_dev = tmp_path / "hailo0"
    fake_dev.touch()
    yolo_hef = tmp_path / "yolo.hef"
    yolo_hef.write_bytes(b"\x00" * 64)
    fe_hef = tmp_path / "fe.hef"
    fe_hef.write_bytes(b"\x00" * 64)

    cfg = Settings(
        mock_hardware=True,
        hailo={
            "enabled": True,
            "device_path": str(fake_dev),
            "yolo_hef_path": str(yolo_hef),
            "feature_extractor_hef_path": str(fe_hef),
            "timeout_ms": 100.0,
        },
    )

    # 2. Force hailo_platform to be importable.
    monkeypatch.setitem(_sys.modules, "hailo_platform", ModuleType("hailo_platform"))

    # 3. Build a stub HailoRuntime — protocol-shaped, captures the call order.
    class _StubVStream:
        shape: tuple[int, int, int] = (320, 320, 3)

    class _StubRuntime:
        def __init__(self) -> None:
            self.started = False
            self.stopped = False
            self.inferences: list[tuple[str, tuple[int, ...]]] = []
            # _models is the private attr the smoke reflects against.
            self._models: dict[str, dict[str, object]] = {
                "yolo": {"input_vstream_infos": [_StubVStream()]},
                "feature_extractor": {},
            }

        async def start(self) -> None:
            self.started = True

        async def stop(self) -> None:
            self.stopped = True

        def is_available(self) -> bool:
            return True

        def infer_sync(self, model_name: str, image: np.ndarray) -> np.ndarray:
            self.inferences.append((model_name, image.shape))
            return np.zeros((1, 4), dtype=np.float32)

    stub_runtime = _StubRuntime()
    monkeypatch.setattr(
        "mousedroid.factory.build_hailo_runtime",
        lambda _cfg: stub_runtime,
    )

    # 4. Execute.
    result = await runtime.verify_hailo_accelerator(cfg)

    # 5. Assertions: device + SDK present, both HEFs loaded, inference timed.
    assert result.device_path_exists is True
    assert result.sdk_importable is True
    assert result.hef_files["yolo"].startswith("loaded")
    assert result.hef_files["feature_extractor"].startswith("loaded")
    assert result.inference_latency_ms is not None
    assert result.inference_latency_ms >= 0.0
    # device_info: schema-driven device_path + concrete models-loaded count
    # (NOT the dead _device_id reflection).
    assert result.device_info["device_path"] == str(fake_dev)
    assert result.device_info["models_loaded"] == "2"
    assert result.device_info["models_configured"] == "2"
    # Smoke must have started AND stopped the runtime (PCIe device lock released).
    assert stub_runtime.started is True
    assert stub_runtime.stopped is True
    # Inference used the vstream-reported shape, not the canonical fallback.
    assert stub_runtime.inferences == [("yolo", (320, 320, 3))]


@pytest.mark.asyncio
async def test_verify_hailo_accelerator_uses_synthetic_shape_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """When the runtime exposes no vstream info, the schema-driven shape is used."""
    import sys as _sys
    from types import ModuleType

    fake_dev = tmp_path / "hailo0"
    fake_dev.touch()
    yolo_hef = tmp_path / "yolo.hef"
    yolo_hef.write_bytes(b"\x00")

    cfg = Settings(
        mock_hardware=True,
        hailo={
            "enabled": True,
            "device_path": str(fake_dev),
            "yolo_hef_path": str(yolo_hef),
            "synthetic_input_shape": (128, 128, 3),  # schema override
        },
    )
    monkeypatch.setitem(_sys.modules, "hailo_platform", ModuleType("hailo_platform"))

    class _StubRuntimeNoVStreams:
        def __init__(self) -> None:
            self._models: dict[str, dict[str, object]] = {"yolo": {}}  # no input_vstream_infos
            self.inferences: list[tuple[str, tuple[int, ...]]] = []

        async def start(self) -> None:
            return None

        async def stop(self) -> None:
            return None

        def is_available(self) -> bool:
            return True

        def infer_sync(self, model_name: str, image: np.ndarray) -> np.ndarray:
            self.inferences.append((model_name, image.shape))
            return np.zeros((1, 4), dtype=np.float32)

    stub = _StubRuntimeNoVStreams()
    monkeypatch.setattr("mousedroid.factory.build_hailo_runtime", lambda _cfg: stub)

    result = await runtime.verify_hailo_accelerator(cfg)

    assert result.inference_latency_ms is not None
    # The smoke must have used the schema-driven fallback shape (128, 128, 3).
    assert stub.inferences == [("yolo", (128, 128, 3))]


@pytest.mark.asyncio
async def test_verify_hailo_accelerator_stop_failure_logged_via_get_logger(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Failed runtime.stop() routes through ``mousedroid.logging.setup.get_logger``."""
    import sys as _sys
    from types import ModuleType

    fake_dev = tmp_path / "hailo0"
    fake_dev.touch()
    cfg = Settings(
        mock_hardware=True,
        hailo={"enabled": True, "device_path": str(fake_dev)},
    )
    monkeypatch.setitem(_sys.modules, "hailo_platform", ModuleType("hailo_platform"))

    class _BadStopRuntime:
        # ClassVar empty-dict — the helper reflects via ``getattr`` so the
        # value just needs to be a mapping; no instance state required.
        _models: ClassVar[dict[str, object]] = {}

        async def start(self) -> None:
            return None

        async def stop(self) -> None:
            msg = "simulated PCIe device lock release failure"
            raise RuntimeError(msg)

        def is_available(self) -> bool:
            return True

        def infer_sync(self, model_name: str, image: np.ndarray) -> np.ndarray:
            return np.zeros((1, 4), dtype=np.float32)

    monkeypatch.setattr("mousedroid.factory.build_hailo_runtime", lambda _cfg: _BadStopRuntime())

    # Helper must NOT propagate the stop failure (CLI consumers depend on it).
    result = await runtime.verify_hailo_accelerator(cfg)
    assert result.device_path_exists is True

    # And the structured warning must have landed via structlog (the project
    # processor chain renders to stdout in tests).
    captured = capsys.readouterr()
    assert "hailo_runtime_stop_failed_in_smoke" in (captured.out + captured.err)


# ---------------------------------------------------------------------------
# Task 2 follow-up — PCIe SSD tool-present branches (closes 85% gate)
# ---------------------------------------------------------------------------


def test_verify_pcie_ssd_layout_parses_lspci_and_lsblk_and_smartctl(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Tool-present branches: parsed NVMe lines + SMART health populate the dataclass."""
    mount_dir = tmp_path / "ssd_mount"
    mount_dir.mkdir()
    monkeypatch.setenv("MOUSEDROID_SSD_MOUNT", str(mount_dir))

    # Pretend every probed tool is installed.
    monkeypatch.setattr("shutil.which", lambda cmd: f"/usr/bin/{cmd}")

    fixtures = {
        "lspci": (
            "00:1f.2 SATA controller [0106]\n"
            "01:00.0 Non-Volatile memory controller [0108]: Samsung NVMe SSD\n"
            "02:00.0 USB controller [0c03]\n"
        ),
        "lsblk": ("nvme0n1 1.8T disk nvme\nsda    256G disk sata\n"),
        "smartctl": (
            "SMART overall-health self-assessment test result: PASSED\nCritical Warning: 0x00\n"
        ),
    }

    def _fake_run(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        executable = cmd[0]
        if executable.endswith("/findmnt") or executable == "findmnt":
            return subprocess.CompletedProcess(cmd, 0, str(mount_dir), "")
        for key, output in fixtures.items():
            if executable.endswith(f"/{key}") or executable == key:
                return subprocess.CompletedProcess(cmd, 0, output, "")
        return subprocess.CompletedProcess(cmd, 1, "", "unknown cmd")

    monkeypatch.setattr("subprocess.run", _fake_run)

    cfg = Settings(mock_hardware=True)
    result = runtime.verify_pcie_ssd_layout(cfg)

    assert any("Non-Volatile memory" in dev for dev in result.pcie_devices)
    assert any("nvme0n1" in dev for dev in result.block_devices)
    assert result.smartctl_health == "PASSED"
    assert result.mount_target == mount_dir


def test_verify_pcie_ssd_layout_subprocess_timeout_is_config_driven(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``cfg.experience.diagnostics_subprocess_timeout_s`` is forwarded to subprocess.run."""
    monkeypatch.delenv("MOUSEDROID_SSD_MOUNT", raising=False)
    monkeypatch.setattr("shutil.which", lambda cmd: f"/usr/bin/{cmd}")

    captured_timeouts: list[float] = []

    def _capture_timeout(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured_timeouts.append(float(kwargs.get("timeout", -1.0)))
        return subprocess.CompletedProcess(cmd, 1, "", "")

    monkeypatch.setattr("subprocess.run", _capture_timeout)

    exp_path = tmp_path / "lmdb_root"
    exp_path.mkdir()
    cfg = Settings(
        mock_hardware=True,
        experience={
            "path": str(exp_path / "experience.lmdb"),
            "diagnostics_subprocess_timeout_s": 3.5,
        },
    )
    runtime.verify_pcie_ssd_layout(cfg)

    # Every subprocess call should have received the schema-driven timeout.
    assert captured_timeouts  # at least one tool was probed
    assert all(t == pytest.approx(3.5) for t in captured_timeouts)


def test_verify_pcie_ssd_layout_uses_configured_nvme_device_in_smartctl(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``smartctl`` invocation targets ``cfg.experience.nvme_device`` not a hardcoded path."""
    monkeypatch.delenv("MOUSEDROID_SSD_MOUNT", raising=False)
    monkeypatch.setattr("shutil.which", lambda cmd: f"/usr/bin/{cmd}")

    seen_commands: list[list[str]] = []

    def _capture(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        seen_commands.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr("subprocess.run", _capture)

    cfg = Settings(
        mock_hardware=True,
        experience={
            "path": str(tmp_path / "exp"),
            "nvme_device": "/dev/nvme1n1",  # secondary slot
            "nvme_partition": "/dev/nvme1n1p2",
        },
    )
    runtime.verify_pcie_ssd_layout(cfg)

    smartctl_calls = [c for c in seen_commands if c[0].endswith("smartctl") or c[0] == "smartctl"]
    findmnt_calls = [c for c in seen_commands if c[0].endswith("findmnt") or c[0] == "findmnt"]
    assert smartctl_calls
    assert smartctl_calls[0][-1] == "/dev/nvme1n1"
    assert findmnt_calls
    assert findmnt_calls[0][-1] == "/dev/nvme1n1p2"


def test_resolve_pcie_ssd_mount_no_rootfs_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Footgun guard: no env override + findmnt missing → return None, NOT exp_path.parent."""
    monkeypatch.delenv("MOUSEDROID_SSD_MOUNT", raising=False)
    monkeypatch.setattr("shutil.which", lambda cmd: None)  # disables findmnt

    exp_path = tmp_path / "rootfs_home" / "mousedroid_experience"
    exp_path.parent.mkdir(parents=True)
    cfg = Settings(
        mock_hardware=True,
        experience={"path": str(exp_path)},
    )
    result = runtime._resolve_pcie_ssd_mount(cfg)
    # The previous implementation returned exp_path.parent here, falsely
    # marking the rootfs as the SSD mount. The hardened version returns
    # None so the CLI surfaces a SKIP instead of a FALSE PASS.
    assert result is None


# ---------------------------------------------------------------------------
# Task 1 follow-up — sync capture_raw_frame + JPEG-quality config + HEF roles
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_capture_camera_frame_supports_sync_capture_raw_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A driver exposing a SYNC ``capture_raw_frame`` is wrapped via asyncio.to_thread."""

    class _SyncDriverCamera:
        _backend = "sync_stub"

        async def start(self) -> None:
            return None

        async def stop(self) -> None:
            return None

        def capture_raw_frame(self) -> np.ndarray:  # SYNC — would crash old code on `await`
            return np.full((6, 9, 3), 7, dtype=np.uint8)

    monkeypatch.setattr(runtime, "build_camera", lambda cfg: _SyncDriverCamera())

    diagnostics, backend_name = await runtime.capture_camera_frame(
        Settings(mock_hardware=True),
    )

    assert backend_name == "sync_stub"
    assert diagnostics.frame is not None
    assert diagnostics.frame.shape == (6, 9, 3)
    assert int(diagnostics.frame[0, 0, 0]) == 7


@pytest.mark.asyncio
async def test_capture_camera_frame_honours_snapshot_jpeg_quality(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``cfg.camera.snapshot_jpeg_quality`` flows through to Pillow's encoder."""

    class _StubCamera:
        _backend = "qualstub"

        def __init__(self, cfg_camera: object) -> None:
            self._cfg = cfg_camera

        async def start(self) -> None:
            return None

        async def stop(self) -> None:
            return None

        async def capture_raw_frame(self) -> np.ndarray:
            # Deterministic content so JPEG-quality differences produce
            # detectable file-size deltas.
            return (
                (np.indices((40, 40)).sum(axis=0).astype(np.uint8) * 4)
                .reshape(40, 40, 1)
                .repeat(3, axis=2)
            )

    cfg_high = Settings(mock_hardware=True, camera={"snapshot_jpeg_quality": 95})
    monkeypatch.setattr(runtime, "build_camera", lambda _cfg: _StubCamera(cfg_high.camera))

    high_path = tmp_path / "high.jpg"
    diag_high, _ = await runtime.capture_camera_frame(cfg_high, save_path=high_path)
    assert diag_high.saved_to is not None
    high_size = high_path.stat().st_size

    cfg_low = Settings(mock_hardware=True, camera={"snapshot_jpeg_quality": 10})
    monkeypatch.setattr(runtime, "build_camera", lambda _cfg: _StubCamera(cfg_low.camera))

    low_path = tmp_path / "low.jpg"
    await runtime.capture_camera_frame(cfg_low, save_path=low_path)
    low_size = low_path.stat().st_size

    # Higher quality produces a larger JPEG for the same pixel data.
    assert high_size > low_size


def test_discover_hef_role_fields_from_schema() -> None:
    """``_discover_hef_role_fields`` enumerates every ``*_hef_path`` schema field."""
    from mousedroid.config.schema import HailoConfig

    cfg = HailoConfig()
    roles = dict(runtime._discover_hef_role_fields(cfg))
    assert roles["yolo"] == "yolo_hef_path"
    assert roles["feature_extractor"] == "feature_extractor_hef_path"


def test_discover_hef_role_fields_falls_back_when_model_fields_absent() -> None:
    """Stubs without ``model_fields`` still get the canonical YOLO + feature-extractor pair."""

    class _StubHailoCfg:
        """Bare stub — no Pydantic ``model_fields`` introspection support."""

        yolo_hef_path = "ignored"

    roles = dict(runtime._discover_hef_role_fields(_StubHailoCfg()))
    assert roles == {"yolo": "yolo_hef_path", "feature_extractor": "feature_extractor_hef_path"}
