from __future__ import annotations

from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from mousedroid.config.schema import HailoConfig
from mousedroid.hardware.accelerator.hailo_runtime import (
    HailoRuntimeProtocol,
    MockHailoRuntime,
)


def _make_hailo_cfg(**overrides: Any) -> HailoConfig:
    defaults: dict[str, Any] = {
        "enabled": True,
        "device_path": "/dev/hailo0",
        "yolo_hef_path": "models/hailo/yolo.hef",
        "feature_extractor_hef_path": "models/hailo/feat.hef",
    }
    defaults.update(overrides)
    return HailoConfig(**defaults)


# ---------------------------------------------------------------------------
# MockHailoRuntime
# ---------------------------------------------------------------------------


class TestMockHailoRuntime:
    @pytest.mark.asyncio
    async def test_start_sets_available(self) -> None:
        cfg = _make_hailo_cfg()
        rt = MockHailoRuntime(cfg)
        assert not rt.is_available()
        await rt.start()
        assert rt.is_available()

    @pytest.mark.asyncio
    async def test_stop_clears_available(self) -> None:
        cfg = _make_hailo_cfg()
        rt = MockHailoRuntime(cfg)
        await rt.start()
        await rt.stop()
        assert not rt.is_available()

    @pytest.mark.asyncio
    async def test_yolo_output_shape(self) -> None:
        cfg = _make_hailo_cfg()
        rt = MockHailoRuntime(cfg)
        await rt.start()
        dummy_input = np.zeros((480, 640, 3), dtype=np.uint8)
        result = await rt.run_inference("yolo", dummy_input)
        assert result.shape == (25200, 85)
        assert result.dtype == np.float32

    @pytest.mark.asyncio
    async def test_feature_extractor_output_shape(self) -> None:
        cfg = _make_hailo_cfg()
        rt = MockHailoRuntime(cfg)
        await rt.start()
        dummy_input = np.zeros((480, 640, 3), dtype=np.uint8)
        result = await rt.run_inference("feature_extractor", dummy_input)
        assert result.shape == (256,)
        assert result.dtype == np.float32

    @pytest.mark.asyncio
    async def test_unknown_model_raises(self) -> None:
        cfg = _make_hailo_cfg()
        rt = MockHailoRuntime(cfg)
        await rt.start()
        with pytest.raises(RuntimeError, match="Unknown model"):
            await rt.run_inference("nonexistent", np.zeros(1, dtype=np.uint8))

    @pytest.mark.asyncio
    async def test_not_started_raises(self) -> None:
        cfg = _make_hailo_cfg()
        rt = MockHailoRuntime(cfg)
        with pytest.raises(RuntimeError, match="not available"):
            await rt.run_inference("yolo", np.zeros(1, dtype=np.uint8))

    def test_implements_protocol(self) -> None:
        cfg = _make_hailo_cfg()
        rt = MockHailoRuntime(cfg)
        assert isinstance(rt, HailoRuntimeProtocol)


# ---------------------------------------------------------------------------
# HailoRuntime (with mocked hailo_platform)
# ---------------------------------------------------------------------------


class TestHailoRuntimeImportGuard:
    def test_runtime_available_without_hailort(self) -> None:
        """HailoRuntime module loads even if hailo_platform is absent."""
        with patch.dict("sys.modules", {"hailo_platform": None}):
            import importlib

            import mousedroid.hardware.accelerator.hailo_runtime as mod

            importlib.reload(mod)
            # Module should load; _hailort should be None
            assert mod._hailort is None
