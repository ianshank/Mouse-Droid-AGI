"""Unit tests for the VoiceConfig event-name model validator (PR #6, V4).

Typos in ``event_intensity_thresholds`` or ``cooldown_per_event`` previously
fell back silently to global defaults. The model validator now rejects keys
not present in the default phrase bank (or in ``phrase_overrides``) so
operators see the typo at config-load time.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mousedroid.config.schema import VoiceConfig


def test_known_event_passes_validation() -> None:
    """A known phrase-bank event in event_intensity_thresholds is accepted."""
    cfg = VoiceConfig(
        enabled=True,
        event_intensity_thresholds={"obstacle_detected": 0.8},
    )
    assert cfg.event_intensity_thresholds == {"obstacle_detected": 0.8}


def test_unknown_event_in_intensity_thresholds_raises() -> None:
    """A typo'd event name in event_intensity_thresholds raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        VoiceConfig(
            enabled=True,
            event_intensity_thresholds={"OBSTACLE": 0.8},  # wrong casing → unknown
        )
    msg = str(exc_info.value)
    assert "OBSTACLE" in msg
    assert "event_intensity_thresholds" in msg


def test_unknown_event_in_cooldown_per_event_raises() -> None:
    """A typo'd event name in cooldown_per_event raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        VoiceConfig(
            enabled=True,
            cooldown_per_event={"obstcle_detected": 1.0},  # typo
        )
    msg = str(exc_info.value)
    assert "obstcle_detected" in msg
    assert "cooldown_per_event" in msg


def test_phrase_override_events_are_accepted() -> None:
    """Events registered through phrase_overrides count as known."""
    cfg = VoiceConfig(
        enabled=True,
        phrase_overrides={"custom_event": ["Hello!"]},
        event_intensity_thresholds={"custom_event": 0.5},
        cooldown_per_event={"custom_event": 2.0},
    )
    assert cfg.event_intensity_thresholds == {"custom_event": 0.5}
    assert cfg.cooldown_per_event == {"custom_event": 2.0}


def test_multiple_bad_keys_all_reported() -> None:
    """Validation surfaces every typo across both fields in one error."""
    with pytest.raises(ValidationError) as exc_info:
        VoiceConfig(
            enabled=True,
            event_intensity_thresholds={"NOPE": 0.5},
            cooldown_per_event={"AlsoBad": 1.0},
        )
    msg = str(exc_info.value)
    assert "NOPE" in msg
    assert "AlsoBad" in msg


def test_default_voiceconfig_loads_unchanged() -> None:
    """Backwards compatibility — empty config still loads without raising."""
    cfg = VoiceConfig()
    assert cfg.event_intensity_thresholds == {}
    assert cfg.cooldown_per_event == {}
    assert cfg.token_bucket_capacity == 3
    assert cfg.token_bucket_refill_rate == 1.0


def test_per_event_cooldown_must_be_positive() -> None:
    """cooldown_per_event values must be > 0 to mirror the global cooldown_s constraint."""
    with pytest.raises(ValidationError):
        VoiceConfig(
            enabled=True,
            cooldown_per_event={"obstacle_detected": 0.0},
        )
