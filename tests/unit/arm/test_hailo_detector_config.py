"""Test hailo_detector configuration schema."""

from __future__ import annotations

from mousedroid.config.schema import ArmPerceptionConfig


def test_arm_perception_config_has_nms_iou_threshold() -> None:
    """Config schema must include yolo_nms_iou_threshold for hailo detector."""
    cfg = ArmPerceptionConfig()
    assert hasattr(cfg, "yolo_nms_iou_threshold")
    assert 0 < cfg.yolo_nms_iou_threshold <= 1.0


def test_arm_perception_config_nms_iou_default_matches_postprocess() -> None:
    """Default NMS IoU should be 0.45 to match _hailo_yolo_postprocess default."""
    cfg = ArmPerceptionConfig()
    assert cfg.yolo_nms_iou_threshold == 0.45
