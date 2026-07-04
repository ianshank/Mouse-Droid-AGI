"""Unit tests for the non-mock branches of the preflight checks.

These branches lazily import :mod:`mousedroid.validation.runtime` /
:mod:`mousedroid.factory` at call time, so monkeypatching the module
attributes exercises the real-hardware code paths without any device.
Added alongside F-017 so the whole preflight module sits under the
changed-file coverage gate, not just the new ``host_env_keys`` check.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import mousedroid.factory as factory
import mousedroid.validation.preflight as preflight
import mousedroid.validation.runtime as runtime
from mousedroid.config.schema import Settings
from mousedroid.validation.preflight import (
    PreflightStatus,
    _check_camera,
    _check_config,
    _check_esp32,
    _check_lidar,
    _check_microphone,
    _check_speaker,
    _list_video_device_nodes,
    _load_proc_modules,
)


@pytest.fixture
def real_cfg() -> Settings:
    # Construct in mock mode (satisfies the distance-sensor validator), then
    # flip to real mode — the cli/preflight.py --mock-hardware pattern.
    cfg = Settings(mock_hardware=True)
    cfg.mock_hardware = False
    return cfg


class TestCameraRealBranch:
    async def test_unavailable_reason_fails(
        self, real_cfg: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(preflight, "_detect_csi_ribbon_disconnect", lambda: None)
        monkeypatch.setattr(runtime, "camera_unavailable_reason", lambda cfg: "no /dev/video0")
        result = await _check_camera(real_cfg)
        assert result.status is PreflightStatus.FAIL
        assert "no /dev/video0" in result.detail

    async def test_ribbon_disconnect_warns(
        self, real_cfg: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            preflight, "_detect_csi_ribbon_disconnect", lambda: "CSI ribbon disconnected"
        )
        result = await _check_camera(real_cfg)
        assert result.status is PreflightStatus.WARN
        assert "ribbon" in result.detail

    async def test_capture_ok(self, real_cfg: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_capture(cfg: Settings) -> tuple[SimpleNamespace, str]:
            return SimpleNamespace(shape=(480, 640, 3)), "v4l2"

        monkeypatch.setattr(preflight, "_detect_csi_ribbon_disconnect", lambda: None)
        monkeypatch.setattr(runtime, "camera_unavailable_reason", lambda cfg: None)
        monkeypatch.setattr(runtime, "capture_camera_frame", fake_capture)
        result = await _check_camera(real_cfg)
        assert result.status is PreflightStatus.OK
        assert "v4l2" in result.detail


class TestMicrophoneRealBranch:
    async def test_no_chunk_warns(
        self, real_cfg: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_chunk(cfg: Settings) -> None:
            return None

        monkeypatch.setattr(runtime, "capture_microphone_chunk", fake_chunk)
        result = await _check_microphone(real_cfg)
        assert result.status is PreflightStatus.WARN

    async def test_chunk_ok(self, real_cfg: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_chunk(cfg: Settings) -> SimpleNamespace:
            return SimpleNamespace(size=1024, dtype="int16")

        monkeypatch.setattr(runtime, "capture_microphone_chunk", fake_chunk)
        result = await _check_microphone(real_cfg)
        assert result.status is PreflightStatus.OK
        assert "samples=1024" in result.detail


class TestSpeakerRealBranch:
    async def test_tone_ok(self, real_cfg: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_tone(cfg: Settings) -> int:
            return 22050

        monkeypatch.setattr(runtime, "play_speaker_tone", fake_tone)
        result = await _check_speaker(real_cfg)
        assert result.status is PreflightStatus.OK
        assert "samples_written=22050" in result.detail


def _diag(n_points: int, coverage: float) -> SimpleNamespace:
    return SimpleNamespace(n_points=n_points, validation_coverage_deg=coverage)


class TestLidarRealBranch:
    async def _run(
        self,
        real_cfg: Settings,
        monkeypatch: pytest.MonkeyPatch,
        diags: list[SimpleNamespace],
    ) -> preflight.PreflightCheckResult:
        async def fake_diags(cfg: Settings) -> list[SimpleNamespace]:
            return diags

        monkeypatch.setattr(runtime, "collect_lidar_diagnostics", fake_diags)
        return await _check_lidar(real_cfg)

    async def test_empty_diagnostics_fail(
        self, real_cfg: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = await self._run(real_cfg, monkeypatch, [])
        assert result.status is PreflightStatus.FAIL

    async def test_zero_points_fail(
        self, real_cfg: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = await self._run(real_cfg, monkeypatch, [_diag(0, 360.0)])
        assert result.status is PreflightStatus.FAIL

    async def test_low_coverage_warns(
        self, real_cfg: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        floor = getattr(real_cfg.lidar, "min_scan_coverage_deg", 0.0)
        if floor <= 0.0:
            pytest.skip("config has no positive coverage floor to undershoot")
        result = await self._run(real_cfg, monkeypatch, [_diag(100, floor / 2)])
        assert result.status is PreflightStatus.WARN

    async def test_healthy_scan_ok(
        self, real_cfg: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = await self._run(real_cfg, monkeypatch, [_diag(450, 360.0), _diag(440, 359.0)])
        assert result.status is PreflightStatus.OK
        assert "scans=2" in result.detail


class TestEsp32RealBranch:
    async def test_none_driver_fails(
        self, real_cfg: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(factory, "build_esp32_driver", lambda cfg: None)
        result = await _check_esp32(real_cfg)
        assert result.status is PreflightStatus.FAIL

    async def test_driver_ok(self, real_cfg: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(factory, "build_esp32_driver", lambda cfg: SimpleNamespace())
        result = await _check_esp32(real_cfg)
        assert result.status is PreflightStatus.OK


class TestConfigCheckIssueBranch:
    async def test_bad_invariants_fail(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = Settings(mock_hardware=True)
        monkeypatch.setattr(cfg.model, "action_dim", 0, raising=True)
        monkeypatch.setattr(cfg.loop, "control_hz", 0.0, raising=True)
        result = await _check_config(cfg)
        assert result.status is PreflightStatus.FAIL
        assert "action_dim" in result.detail
        assert "control_hz" in result.detail


class TestHostHelpers:
    def test_load_proc_modules_never_raises(self) -> None:
        # Real read on Linux, empty string in restrictive sandboxes — both fine.
        assert isinstance(_load_proc_modules(), str)

    def test_list_video_device_nodes_sorted(self) -> None:
        nodes = _list_video_device_nodes()
        assert nodes == sorted(nodes)
