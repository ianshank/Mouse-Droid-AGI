"""Perception facade composing detection, pose, and state extraction.

Implements ``ArmPerceptionProtocol`` by orchestrating the four
perception sub-components into a single coherent pipeline:
depth processing -> object detection -> pose estimation -> state extraction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mousedroid.arm.perception.depth_processor import DepthProcessor
from mousedroid.arm.perception.object_detector import ObjectDetector, ObjectDetectorProtocol
from mousedroid.arm.perception.pose_estimator import PoseEstimator
from mousedroid.arm.perception.state_extractor import StateExtractor
from mousedroid.arm.protocols import DetectedObject, SymbolicState
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mousedroid.config.schema import ArmPerceptionConfig, ArmTaskConfig

_log = get_logger(__name__)


class ArmPerception:
    """Unified perception pipeline for robot arm platform.

    Composes depth processing, object detection, pose estimation,
    and symbolic state extraction into the ``ArmPerceptionProtocol``.

    Args:
        perception_cfg: Perception configuration.
        task_cfg: Task configuration for state extraction.
        intrinsics: Camera intrinsic matrix (3x3).
        object_detector: Optional pre-built detector (e.g. Hailo-accelerated).
            When ``None``, builds the default ``ObjectDetector``.
    """

    def __init__(
        self,
        perception_cfg: ArmPerceptionConfig,
        task_cfg: ArmTaskConfig,
        intrinsics: NDArray[np.float64],
        object_detector: ObjectDetectorProtocol | None = None,
    ) -> None:
        """Initialise perception pipeline.

        Args:
            perception_cfg: Config for depth, YOLO, and pose settings.
            task_cfg: Config for peg/basket positions.
            intrinsics: Camera intrinsic matrix, shape ``(3, 3)``.
            object_detector: Optional pre-built detector override.
        """
        self._depth_processor = DepthProcessor(perception_cfg, intrinsics)
        self._detector: ObjectDetectorProtocol = (
            object_detector if object_detector is not None else ObjectDetector(perception_cfg)
        )
        self._pose_estimator = PoseEstimator(perception_cfg, intrinsics)
        self._state_extractor = StateExtractor(task_cfg)
        self._last_depth: NDArray[np.float32] | None = None
        self._last_rgb: NDArray[np.uint8] | None = None
        _log.info("arm_perception_init")

    async def start(self) -> None:
        """Start perception pipeline (load models)."""
        self._detector.load_model()
        _log.info("perception_started")

    async def stop(self) -> None:
        """Stop perception pipeline."""
        self._last_depth = None
        self._last_rgb = None
        _log.info("perception_stopped")

    def update_frames(
        self,
        rgb: NDArray[np.uint8],
        depth: NDArray[np.float32],
    ) -> None:
        """Update cached camera frames for subsequent queries.

        Args:
            rgb: RGB image, shape ``(H, W, 3)``.
            depth: Depth image, shape ``(H, W)``, metres.
        """
        self._last_rgb = rgb
        self._last_depth = self._depth_processor.filter_depth(depth)

    async def detect_objects(self) -> list[DetectedObject]:
        """Detect objects in current camera frame.

        Returns:
            List of detected objects with populated 6-DoF poses.
        """
        if self._last_rgb is None or self._last_depth is None:
            _log.warning("no_frames_available")
            return []

        detections = self._detector.detect(self._last_rgb)
        if detections:
            detections = self._pose_estimator.estimate_poses(detections, self._last_depth)
        _log.debug("perception_detect", count=len(detections))
        return detections

    async def get_symbolic_state(self) -> SymbolicState:
        """Convert current detections to symbolic PDDL state.

        Returns:
            Symbolic state with active predicates.
        """
        detections = await self.detect_objects()
        state = self._state_extractor.extract(detections)
        _log.debug("perception_symbolic_state", predicates=len(state.predicates))
        return state
