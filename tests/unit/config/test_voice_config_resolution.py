"""Unit tests for VoiceConfig.resolved_tts_model_path() and new schema fields."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mousedroid.config.schema import VoiceConfig


def _cfg(**overrides: object) -> VoiceConfig:
    return VoiceConfig(**overrides)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# resolved_tts_model_path()
# ---------------------------------------------------------------------------


class TestResolvedTtsModelPath:
    """Tests for the personality-to-model resolution helper."""

    def test_empty_map_returns_tts_model_path(self) -> None:
        """With no map, resolved path equals tts_model_path."""
        cfg = _cfg(tts_model_path="/models/rocky.onnx")
        assert cfg.resolved_tts_model_path() == "/models/rocky.onnx"

    def test_map_hit_returns_map_value(self) -> None:
        """Map hit overrides tts_model_path."""
        cfg = _cfg(
            personality="rocky",
            tts_model_path="/models/default.onnx",
            personality_to_model_map={"rocky": "/models/rocky_hd.onnx"},
        )
        assert cfg.resolved_tts_model_path() == "/models/rocky_hd.onnx"

    def test_map_miss_falls_back_to_tts_model_path(self) -> None:
        """Map exists but personality has no entry → falls back."""
        cfg = _cfg(
            personality="rocky",
            tts_model_path="/models/default.onnx",
            personality_to_model_map={"other_voice": "/models/other.onnx"},
        )
        assert cfg.resolved_tts_model_path() == "/models/default.onnx"

    def test_both_none_returns_none(self) -> None:
        """No map and no tts_model_path returns None."""
        cfg = _cfg(tts_model_path=None)
        assert cfg.resolved_tts_model_path() is None

    def test_map_hit_with_none_tts_model_path(self) -> None:
        """Map hit is returned even when tts_model_path is None."""
        cfg = _cfg(
            personality="rocky",
            tts_model_path=None,
            personality_to_model_map={"rocky": "/models/rocky.onnx"},
        )
        assert cfg.resolved_tts_model_path() == "/models/rocky.onnx"

    def test_map_miss_with_none_tts_model_path(self) -> None:
        """Map exists but no matching personality and tts_model_path=None → None."""
        cfg = _cfg(
            personality="rocky",
            tts_model_path=None,
            personality_to_model_map={"other": "/models/other.onnx"},
        )
        assert cfg.resolved_tts_model_path() is None

    def test_personality_string_not_restricted_to_rocky(self) -> None:
        """personality is now a plain str, not Literal['rocky']."""
        cfg = _cfg(
            personality="custom_voice",
            personality_to_model_map={"custom_voice": "/models/custom.onnx"},
        )
        assert cfg.resolved_tts_model_path() == "/models/custom.onnx"


# ---------------------------------------------------------------------------
# event_intensity_thresholds defaults & validation
# ---------------------------------------------------------------------------


class TestEventIntensityThresholds:
    """Tests for the event_intensity_thresholds field."""

    def test_defaults_to_empty_dict(self) -> None:
        """event_intensity_thresholds defaults to an empty dict."""
        cfg = _cfg()
        assert cfg.event_intensity_thresholds == {}

    def test_valid_thresholds_accepted(self) -> None:
        """Valid 0-1 threshold values are accepted."""
        cfg = _cfg(event_intensity_thresholds={"obstacle_detected": 0.5, "error": 1.0})
        assert cfg.event_intensity_thresholds["obstacle_detected"] == pytest.approx(0.5)

    def test_zero_threshold_accepted(self) -> None:
        """Threshold value of 0.0 is valid."""
        cfg = _cfg(event_intensity_thresholds={"idle": 0.0})
        assert cfg.event_intensity_thresholds["idle"] == pytest.approx(0.0)

    def test_invalid_threshold_above_one_raises(self) -> None:
        """Threshold above 1.0 raises ValidationError."""
        with pytest.raises(ValidationError, match="event_intensity_thresholds"):
            _cfg(event_intensity_thresholds={"error": 1.5})

    def test_invalid_threshold_below_zero_raises(self) -> None:
        """Threshold below 0.0 raises ValidationError."""
        with pytest.raises(ValidationError, match="event_intensity_thresholds"):
            _cfg(event_intensity_thresholds={"error": -0.1})


# ---------------------------------------------------------------------------
# personality_to_model_map defaults
# ---------------------------------------------------------------------------


class TestPersonalityToModelMap:
    """Tests for the personality_to_model_map field."""

    def test_defaults_to_empty_dict(self) -> None:
        """personality_to_model_map defaults to empty dict."""
        cfg = _cfg()
        assert cfg.personality_to_model_map == {}

    def test_multiple_personalities_accepted(self) -> None:
        """Multiple personality→path entries are accepted."""
        cfg = _cfg(
            personality_to_model_map={
                "rocky": "/models/rocky.onnx",
                "friendly": "/models/friendly.onnx",
            }
        )
        assert len(cfg.personality_to_model_map) == 2

    def test_relative_path_raises_validation_error(self) -> None:
        """A relative model path must be rejected."""
        with pytest.raises(ValidationError, match="absolute path"):
            _cfg(personality_to_model_map={"rocky": "models/rocky.onnx"})

    def test_empty_string_path_raises_validation_error(self) -> None:
        """An empty model path must be rejected."""
        with pytest.raises(ValidationError, match="non-empty path"):
            _cfg(personality_to_model_map={"rocky": ""})

    def test_whitespace_only_path_raises_validation_error(self) -> None:
        """A whitespace-only model path must be rejected."""
        with pytest.raises(ValidationError, match="non-empty path"):
            _cfg(personality_to_model_map={"rocky": "   "})

    def test_absolute_path_accepted(self) -> None:
        """Absolute model paths pass validation."""
        cfg = _cfg(personality_to_model_map={"rocky": "/models/rocky.onnx"})
        assert cfg.personality_to_model_map["rocky"] == "/models/rocky.onnx"


class TestOutputVolume:
    """Tests for the output_volume field."""

    def test_defaults_to_unity_gain(self) -> None:
        cfg = _cfg()
        assert cfg.output_volume == pytest.approx(1.0)

    def test_negative_gain_rejected(self) -> None:
        with pytest.raises(ValidationError, match="output_volume"):
            _cfg(output_volume=-0.1)
