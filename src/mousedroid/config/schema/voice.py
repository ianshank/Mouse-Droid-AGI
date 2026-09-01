"""Voice and face-display configuration models.

USB microphone/speaker I/O, the Rocky voice personality engine, and the
SSD1306 OLED face-display affect mapping.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from mousedroid.config.schema._primitives import Self, StrictBaseModel


class MicrophoneConfig(StrictBaseModel):
    """USB microphone configuration."""

    enabled: bool = Field(True, description="Enable audio capture from this microphone")
    device_index: int | None = Field(None, description="ALSA device index (None=auto-detect)")
    device_name: str = Field("USB", description="USB device name substring for auto-detect")
    sample_rate: int = Field(16000, gt=0, description="Audio sample rate (Hz)")
    channels: int = Field(1, gt=0, le=2, description="Audio channels (1=mono, 2=stereo)")
    chunk_size: int = Field(1024, gt=0, description="Samples per read chunk")
    format: Literal["float32", "int16"] = Field("float32", description="Audio sample format")
    n_mels: int = Field(64, gt=0, description="Number of mel filter bank bins")
    n_fft: int = Field(512, gt=0, description="FFT window size for mel spectrogram")
    hop_length: int = Field(256, gt=0, description="Hop length for mel spectrogram")


class SpeakerConfig(StrictBaseModel):
    """USB speaker output configuration."""

    enabled: bool = Field(True, description="Enable audio playback through this speaker")
    device_index: int | None = Field(None, description="ALSA device index (None=auto-detect)")
    device_name: str = Field("USB", description="USB device name substring for auto-detect")
    sample_rate: int = Field(22050, gt=0, description="Audio output sample rate (Hz)")
    channels: int = Field(1, gt=0, le=2, description="Audio output channels (1=mono, 2=stereo)")
    chunk_size: int = Field(1024, gt=0, description="Samples per write chunk")
    format: Literal["float32", "int16"] = Field("float32", description="Audio sample format")
    write_timeout_s: float = Field(
        0.5,
        gt=0,
        description="Max seconds to wait for speaker buffer space before failing a write",
    )
    write_poll_interval_s: float = Field(
        0.01,
        gt=0,
        description="Seconds between speaker buffer readiness polls",
    )
    reconnect_backoff_initial_s: float = Field(
        0.5,
        gt=0,
        description="Initial backoff delay (seconds) between USB speaker open retries",
    )
    reconnect_backoff_max_s: float = Field(
        10.0,
        gt=0,
        description="Maximum backoff delay (seconds) between USB speaker open retries",
    )
    reconnect_max_attempts: int = Field(
        3,
        ge=1,
        description="Maximum USB speaker open attempts before raising SpeakerUnavailable",
    )


class VoiceConfig(StrictBaseModel):
    """Rocky voice engine configuration."""

    enabled: bool = Field(False, description="Enable Rocky voice output")
    cooldown_s: float = Field(5.0, gt=0, description="Min seconds between utterances")
    personality: str = Field(
        "rocky",
        description=(
            "Voice personality name. If personality_to_model_map contains this key, "
            "its model path overrides tts_model_path; otherwise tts_model_path is used."
        ),
    )
    tts_model_path: str | None = Field(
        None, description="Path to piper voice model (None=disable TTS model loading)"
    )
    tts_sample_rate: int = Field(22050, gt=0, description="TTS output sample rate (Hz)")
    queue_size: int = Field(16, gt=0, description="Max queued speech requests")
    queue_poll_timeout_s: float = Field(1.0, gt=0, description="Worker queue poll timeout (s)")
    phrase_overrides: dict[str, list[str]] = Field(
        default_factory=dict, description="Custom phrase overrides by event name"
    )
    intensity_threshold: float = Field(
        0.7,
        ge=0,
        le=1,
        description="Minimum intensity for Rocky voice transform effects",
    )
    personality_to_model_map: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Optional map of personality name → Piper model path. "
            "When the active personality has an entry here it overrides tts_model_path. "
            "Paths must be absolute (resolved inside the container)."
        ),
    )
    event_intensity_thresholds: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Per-event intensity threshold overrides (0.0-1.0). "
            "Keyed by event name; falls back to intensity_threshold when absent."
        ),
    )
    tts_failure_threshold: int = Field(
        3,
        ge=1,
        description=(
            "Consecutive TTS synthesis failures before promoting warning log to ERROR. "
            "Counter resets on any successful synthesis."
        ),
    )
    cooldown_per_event: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Per-event cooldown overrides (seconds). Keyed by event name; "
            "events not listed fall back to the global cooldown_s. "
            "Must be > 0 for each entry."
        ),
    )
    token_bucket_capacity: int = Field(
        3,
        gt=0,
        description=(
            "Max tokens per priority-class bucket. Each HIGH/NORMAL priority "
            "class has its own token bucket; EMERGENCY is never rate-limited."
        ),
    )
    token_bucket_refill_rate: float = Field(
        1.0,
        gt=0,
        description="Token-bucket refill rate (tokens/second) per priority class.",
    )
    output_volume: float = Field(
        1.0,
        ge=0.0,
        description=(
            "Linear gain applied to synthesized samples before they reach the "
            "speaker. 1.0 = unity gain; values >1 amplify but are clipped to "
            "[-1, 1] in float32 to keep DAC output in the safe range."
        ),
    )

    @field_validator("personality_to_model_map", mode="after")
    @classmethod
    def _validate_personality_model_map(cls, v: dict[str, str]) -> dict[str, str]:
        """Validate and normalize personality→model path map.

        Strips whitespace from each value so runtime consumers (Piper loader)
        receive exactly the validated path. Empty/whitespace-only and relative
        paths are rejected at schema-load time.
        """
        from pathlib import PurePosixPath

        normalized: dict[str, str] = {}
        for key, value in v.items():
            stripped = value.strip()
            if not stripped:
                raise ValueError(
                    f"personality_to_model_map[{key!r}] must be a non-empty path, got {value!r}"
                )
            if not PurePosixPath(stripped).is_absolute():
                raise ValueError(
                    f"personality_to_model_map[{key!r}] must be an absolute path, got {value!r}"
                )
            normalized[key] = stripped
        return normalized

    @field_validator("event_intensity_thresholds", mode="after")
    @classmethod
    def _validate_event_thresholds(cls, v: dict[str, float]) -> dict[str, float]:
        for key, value in v.items():
            if not (0.0 <= value <= 1.0):
                raise ValueError(
                    f"event_intensity_thresholds[{key!r}] must be in [0.0, 1.0], got {value!r}"
                )
        return v

    @field_validator("cooldown_per_event", mode="after")
    @classmethod
    def _validate_cooldown_per_event(cls, v: dict[str, float]) -> dict[str, float]:
        for key, value in v.items():
            if value <= 0.0:
                raise ValueError(f"cooldown_per_event[{key!r}] must be > 0.0, got {value!r}")
        return v

    @model_validator(mode="after")
    def _validate_event_names_in_phrase_bank(self) -> Self:
        """Ensure event keys reference known phrase-bank events.

        Validates that every key in ``event_intensity_thresholds`` and
        ``cooldown_per_event`` exists in the default phrase bank or has been
        registered via ``phrase_overrides``. Typos that previously fell back
        silently to the global defaults now fail at config-load time.
        """
        # Local import keeps the schema module decoupled from the voice
        # package import graph at module load time.
        from mousedroid.voice.phrase_bank import DEFAULT_PHRASES

        known: set[str] = set(DEFAULT_PHRASES.keys()) | set(self.phrase_overrides.keys())

        bad_thresholds = sorted(set(self.event_intensity_thresholds.keys()) - known)
        bad_cooldowns = sorted(set(self.cooldown_per_event.keys()) - known)

        if bad_thresholds or bad_cooldowns:
            parts: list[str] = []
            if bad_thresholds:
                parts.append(
                    f"event_intensity_thresholds contains unknown event(s): {bad_thresholds!r}"
                )
            if bad_cooldowns:
                parts.append(f"cooldown_per_event contains unknown event(s): {bad_cooldowns!r}")
            parts.append(
                "Known events come from mousedroid.voice.phrase_bank.DEFAULT_PHRASES "
                "and any keys registered via phrase_overrides."
            )
            raise ValueError(" ".join(parts))
        return self

    def resolved_tts_model_path(self) -> str | None:
        """Return the effective TTS model path for the configured personality.

        Resolution order:

        1. ``personality_to_model_map[personality]`` — per-personality override
           (values are validated as absolute paths at schema load time).
        2. ``tts_model_path`` — global fallback (used as-is).

        Returns:
            Path string, or ``None`` when no model is configured.
        """
        mapped = self.personality_to_model_map.get(self.personality)
        return mapped if mapped is not None else self.tts_model_path


class FaceDisplayConfig(StrictBaseModel):
    """SSD1306 OLED face-display configuration.

    All thresholds consumed by the affect→expression mapping and the blink
    animation live here so there are no magic numbers in driver or
    controller code. New deployments must opt in by setting ``enabled=True``;
    existing YAML files (which omit the section entirely) remain unaffected
    because :class:`Settings` defaults the field to ``None``.
    """

    enabled: bool = Field(False, description="Enable the face-display subsystem")
    i2c_bus: int = Field(7, ge=0, description="I²C bus index (Jetson Orin Nano header = 7)")
    i2c_address: int = Field(0x3C, ge=0, le=0x7F, description="SSD1306 I²C address")
    width: int = Field(128, gt=0, description="Panel width in pixels")
    height: int = Field(64, gt=0, description="Panel height in pixels")
    rotate: int = Field(0, ge=0, le=3, description="Rotation in 90° steps (0..3)")
    refresh_hz: float = Field(10.0, gt=0, description="Maximum face-controller update rate (Hz)")
    boot_message: str = Field("MSE-6 online", description="Boot banner text")
    idle_blink_interval_s: float = Field(
        4.0,
        ge=0,
        description="Idle blink period (s); 0 disables the blink animation",
    )
    blink_close_duration_s: float = Field(
        0.15, gt=0, description="How long the eyes stay closed during a blink"
    )
    min_dwell_s: float = Field(
        0.6,
        ge=0,
        description="Hysteresis dwell — minimum time on an expression before switching",
    )
    fallback_to_mock_on_error: bool = Field(
        True,
        description="Fall back to the mock driver when the I²C probe fails",
    )
    valence_happy_min: float = Field(
        0.35, ge=-1.0, le=1.0, description="Valence threshold for HAPPY"
    )
    valence_sad_max: float = Field(-0.35, ge=-1.0, le=1.0, description="Valence threshold for SAD")
    arousal_alert_min: float = Field(
        0.55, ge=-1.0, le=1.0, description="Arousal threshold for ALERT"
    )
    arousal_sleepy_max: float = Field(
        -0.45, ge=-1.0, le=1.0, description="Arousal threshold for SLEEPY"
    )
    angry_valence_max: float = Field(
        -0.25, ge=-1.0, le=1.0, description="Valence ceiling for ANGRY"
    )
    angry_arousal_min: float = Field(0.45, ge=-1.0, le=1.0, description="Arousal floor for ANGRY")
    idle_sleepy_after_s: float = Field(
        20.0,
        gt=0,
        description="Idle duration after which the face goes SLEEPY",
    )
    idle_action_epsilon: float = Field(
        1e-3,
        gt=0,
        description=(
            "Action magnitude below which the agent is considered idle. "
            "Tolerates small NN-output noise so the SLEEPY path can trigger."
        ),
    )
