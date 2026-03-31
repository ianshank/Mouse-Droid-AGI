"""Tests for symbolic state extraction from detections."""

from __future__ import annotations

import numpy as np

from mousedroid.arm.perception.state_extractor import StateExtractor
from mousedroid.arm.protocols import DetectedObject
from mousedroid.config.schema import ArmTaskConfig


def _make_extractor(num_pegs: int = 3) -> StateExtractor:
    """Create state extractor with default task config."""
    positions = [[0.2 + i * 0.1, 0.0, 0.0] for i in range(num_pegs)]
    cfg = ArmTaskConfig(num_pegs=num_pegs, peg_positions=positions)
    return StateExtractor(cfg)


def _make_disk(disk_id: str, x: float, y: float = 0.0, z: float = 0.1) -> DetectedObject:
    """Create a mock disk detection."""
    return DetectedObject(
        object_id=disk_id,
        class_name=f"disk_{disk_id[-1]}",
        confidence=0.95,
        position_m=np.array([x, y, z], dtype=np.float64),
        orientation_rad=np.zeros(3, dtype=np.float64),
        bbox=np.array([0, 0, 50, 50], dtype=np.float64),
    )


class TestStateExtractor:
    """Test StateExtractor symbolic state generation."""

    def test_all_disks_on_peg_a(self) -> None:
        extractor = _make_extractor()
        detections = [
            _make_disk("disk_1", x=0.20, z=0.3),  # top
            _make_disk("disk_2", x=0.20, z=0.2),  # middle
            _make_disk("disk_3", x=0.20, z=0.1),  # bottom
        ]
        state = extractor.extract(detections)

        assert "on(disk_3, peg_A)" in state.predicates
        assert "on(disk_2, disk_3)" in state.predicates
        assert "on(disk_1, disk_2)" in state.predicates
        assert "clear(disk_1)" in state.predicates

    def test_disks_split_across_pegs(self) -> None:
        extractor = _make_extractor()
        detections = [
            _make_disk("disk_1", x=0.40, z=0.1),  # peg C
            _make_disk("disk_2", x=0.20, z=0.1),  # peg A
        ]
        state = extractor.extract(detections)

        assert "on(disk_1, peg_C)" in state.predicates
        assert "on(disk_2, peg_A)" in state.predicates
        assert "clear(disk_1)" in state.predicates
        assert "clear(disk_2)" in state.predicates

    def test_empty_peg_is_clear(self) -> None:
        extractor = _make_extractor()
        detections = [
            _make_disk("disk_1", x=0.20, z=0.1),  # peg A only
        ]
        state = extractor.extract(detections)

        assert "clear(peg_B)" in state.predicates
        assert "clear(peg_C)" in state.predicates

    def test_no_disks_all_pegs_clear(self) -> None:
        extractor = _make_extractor()
        state = extractor.extract([])

        assert "clear(peg_A)" in state.predicates
        assert "clear(peg_B)" in state.predicates
        assert "clear(peg_C)" in state.predicates

    def test_objects_contain_disk_and_peg_types(self) -> None:
        extractor = _make_extractor()
        detections = [_make_disk("disk_1", x=0.20, z=0.1)]
        state = extractor.extract(detections)

        assert state.objects["peg_A"] == "peg"
        assert state.objects["disk_1"] == "disk"

    def test_non_disk_detections_ignored(self) -> None:
        extractor = _make_extractor()
        garment = DetectedObject(
            object_id="shirt_1",
            class_name="shirt",
            confidence=0.9,
            position_m=np.array([0.2, 0.0, 0.1], dtype=np.float64),
            orientation_rad=np.zeros(3, dtype=np.float64),
            bbox=np.array([0, 0, 50, 50], dtype=np.float64),
        )
        state = extractor.extract([garment])
        # Only peg predicates, no disk predicates
        assert all("disk" not in p or "clear" in p for p in state.predicates if "disk" in p)
