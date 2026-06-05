"""Unit tests for TemplateCommentaryComposer (deterministic, config-driven)."""

from __future__ import annotations

import pytest

from mousedroid.commentary.composers import TemplateCommentaryComposer
from mousedroid.commentary.protocol import CommentaryFacts
from mousedroid.config.schema import CommentaryConfig


def _facts(
    *,
    min_clearance_m: float = 5.0,
    audio_rms: float = 0.0,
    speed_mps: float = 0.0,
    battery_v: float = 12.0,
    lidar_valid: bool = True,
    audio_valid: bool = True,
) -> CommentaryFacts:
    return CommentaryFacts(
        min_clearance_m=min_clearance_m,
        forward_distance_m=min_clearance_m,
        audio_rms=audio_rms,
        speed_mps=speed_mps,
        turn_rate=0.0,
        battery_v=battery_v,
        novelty=1.0,
        is_emergency=False,
        lidar_valid=lidar_valid,
        audio_valid=audio_valid,
        timestamp=0.0,
    )


@pytest.fixture
def cfg() -> CommentaryConfig:
    return CommentaryConfig(enabled=True, composer="template")


@pytest.mark.asyncio
async def test_low_battery(cfg: CommentaryConfig) -> None:
    out = await TemplateCommentaryComposer(cfg).compose(_facts(battery_v=10.0))
    assert out == cfg.templates["low_battery"]


@pytest.mark.asyncio
async def test_tight_space(cfg: CommentaryConfig) -> None:
    out = await TemplateCommentaryComposer(cfg).compose(_facts(min_clearance_m=0.3))
    assert out == cfg.templates["tight_space"]


@pytest.mark.asyncio
async def test_loud(cfg: CommentaryConfig) -> None:
    out = await TemplateCommentaryComposer(cfg).compose(_facts(audio_rms=0.5))
    assert out == cfg.templates["loud"]


@pytest.mark.asyncio
async def test_moving_fast(cfg: CommentaryConfig) -> None:
    out = await TemplateCommentaryComposer(cfg).compose(_facts(speed_mps=1.0))
    assert out == cfg.templates["moving_fast"]


@pytest.mark.asyncio
async def test_open_space(cfg: CommentaryConfig) -> None:
    out = await TemplateCommentaryComposer(cfg).compose(_facts(min_clearance_m=5.0))
    assert out == cfg.templates["open_space"]


@pytest.mark.asyncio
async def test_default(cfg: CommentaryConfig) -> None:
    out = await TemplateCommentaryComposer(cfg).compose(_facts(min_clearance_m=1.0))
    assert out == cfg.templates["default"]


@pytest.mark.asyncio
async def test_priority_low_battery_beats_tight_space(cfg: CommentaryConfig) -> None:
    out = await TemplateCommentaryComposer(cfg).compose(_facts(min_clearance_m=0.3, battery_v=10.0))
    assert out == cfg.templates["low_battery"]


@pytest.mark.asyncio
async def test_invalid_audio_never_claims_loud(cfg: CommentaryConfig) -> None:
    out = await TemplateCommentaryComposer(cfg).compose(_facts(audio_rms=0.9, audio_valid=False))
    assert out != cfg.templates["loud"]


@pytest.mark.asyncio
async def test_invalid_lidar_never_claims_open_space(cfg: CommentaryConfig) -> None:
    out = await TemplateCommentaryComposer(cfg).compose(
        _facts(min_clearance_m=5.0, lidar_valid=False)
    )
    assert out != cfg.templates["open_space"]


@pytest.mark.asyncio
async def test_output_is_config_driven() -> None:
    """Swapping cfg.templates changes the output (no hardcoded strings)."""
    cfg = CommentaryConfig(
        enabled=True,
        composer="template",
        templates={"default": "custom line", "low_battery": "battery weak"},
    )
    out = await TemplateCommentaryComposer(cfg).compose(_facts(battery_v=10.0))
    assert out == "battery weak"
