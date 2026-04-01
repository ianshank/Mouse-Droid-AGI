"""Tests for ArmPerception facade."""

from __future__ import annotations

import numpy as np
import pytest

from mousedroid.arm.perception.facade import ArmPerception
from mousedroid.config.schema import ArmPerceptionConfig, ArmTaskConfig


def _make_perception() -> ArmPerception:
    perception_cfg = ArmPerceptionConfig()
    task_cfg = ArmTaskConfig(num_disks=3, num_pegs=3)
    intrinsics = np.array(
        [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return ArmPerception(perception_cfg, task_cfg, intrinsics)


class TestArmPerception:
    """Tests for ArmPerception facade."""

    @pytest.mark.asyncio
    async def test_start_and_stop(self) -> None:
        perception = _make_perception()
        await perception.start()
        await perception.stop()

    @pytest.mark.asyncio
    async def test_detect_without_frames_returns_empty(self) -> None:
        perception = _make_perception()
        result = await perception.detect_objects()
        assert result == []

    @pytest.mark.asyncio
    async def test_detect_with_frames_returns_list(self) -> None:
        perception = _make_perception()
        await perception.start()
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        depth = np.full((480, 640), 1.0, dtype=np.float32)
        perception.update_frames(rgb, depth)
        result = await perception.detect_objects()
        # No YOLO model loaded, so returns empty
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_get_symbolic_state_returns_state(self) -> None:
        perception = _make_perception()
        await perception.start()
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        depth = np.full((480, 640), 1.0, dtype=np.float32)
        perception.update_frames(rgb, depth)
        state = await perception.get_symbolic_state()
        assert hasattr(state, "predicates")
        assert hasattr(state, "objects")

    def test_update_frames_stores_data(self) -> None:
        perception = _make_perception()
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        depth = np.full((480, 640), 1.0, dtype=np.float32)
        perception.update_frames(rgb, depth)
        assert perception._last_rgb is not None
        assert perception._last_depth is not None

    @pytest.mark.asyncio
    async def test_stop_clears_frames(self) -> None:
        perception = _make_perception()
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        depth = np.full((480, 640), 1.0, dtype=np.float32)
        perception.update_frames(rgb, depth)
        await perception.stop()
        assert perception._last_rgb is None
        assert perception._last_depth is None
