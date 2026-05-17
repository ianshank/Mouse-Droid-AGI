"""Tier C2.3: build_vlm_progress factory gate + happy-path tests."""

from __future__ import annotations

from mousedroid.config.schema import Settings
from mousedroid.factory import build_vlm_progress
from mousedroid.reward.vlm_progress import MockVLMProgress, VLMProgressHead


def test_returns_none_when_disabled() -> None:
    cfg = Settings(mock_hardware=True)
    assert cfg.mission.vlm_progress_enabled is False
    assert build_vlm_progress(cfg) is None


def test_returns_vlm_progress_head_when_enabled() -> None:
    cfg = Settings(mock_hardware=True)
    cfg.mission.vlm_progress_enabled = True
    head = build_vlm_progress(cfg)
    assert head is not None
    assert isinstance(head, VLMProgressHead)


def test_default_backend_is_mock_vlm_progress() -> None:
    cfg = Settings(mock_hardware=True)
    cfg.mission.vlm_progress_enabled = True
    cfg.mission.vlm_mock_progress_value = 0.75
    head = build_vlm_progress(cfg)
    assert head is not None
    assert isinstance(head._backend, MockVLMProgress)  # noqa: SLF001
