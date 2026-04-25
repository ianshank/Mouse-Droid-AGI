"""Unit tests for reusable runtime validation helpers."""

from __future__ import annotations

from pathlib import Path

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

    frame, backend_name = await runtime.capture_camera_frame(Settings(mock_hardware=True))

    assert frame.shape == (4, 5, 3)
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

    frame, backend_name = await runtime.capture_camera_frame(Settings(mock_hardware=True))

    assert frame.shape == (3, 6, 3)
    assert backend_name == "v4l2"
    assert stub.started is True
    assert stub.stopped is True


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
