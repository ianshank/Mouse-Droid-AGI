"""Unit tests for the voice-degradation counters.

Two pure-add families mirroring the on-device-learning / LLM-gateway counter
pattern: family absent until first write, gated behind
``track_voice_degradation``, low-cardinality labels guarded against
module-level frozensets (out-of-set values dropped with a DEBUG log).

* ``voice_speaker_degraded_total{subsystem}`` — the USB speaker exhausted its
  reconnect retries (``usb_speaker``) or the engine fell back to a MockSpeaker
  (``rocky_fallback``).
* ``voice_tts_synthesize_failures_total{api}`` — a Piper synthesis call raised.
"""

from __future__ import annotations

from mousedroid.config.schema import MetricsConfig
from mousedroid.telemetry import metrics as metrics_mod
from mousedroid.telemetry.metrics import MetricsRegistry, generate_metrics_sample


def _registry(**overrides: object) -> MetricsRegistry:
    return MetricsRegistry(MetricsConfig(**overrides))  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Pure-add: each family absent until its first write
# --------------------------------------------------------------------------- #
def test_families_absent_until_written() -> None:
    out = _registry().render_prometheus()
    assert "voice_speaker_degraded_total" not in out
    assert "voice_tts_synthesize_failures_total" not in out


# --------------------------------------------------------------------------- #
# Speaker-degradation counter
# --------------------------------------------------------------------------- #
def test_speaker_counter_renders_per_subsystem() -> None:
    reg = _registry()
    reg.inc_voice_speaker_degraded("usb_speaker")
    reg.inc_voice_speaker_degraded("usb_speaker")
    reg.inc_voice_speaker_degraded("rocky_fallback")
    out = reg.render_prometheus()
    assert 'mousedroid_voice_speaker_degraded_total{subsystem="usb_speaker"} 2' in out
    assert 'mousedroid_voice_speaker_degraded_total{subsystem="rocky_fallback"} 1' in out


def test_speaker_counter_namespaced() -> None:
    reg = _registry(namespace="rover")
    reg.inc_voice_speaker_degraded("usb_speaker")
    assert "rover_voice_speaker_degraded_total" in reg.render_prometheus()


def test_speaker_counter_noop_on_nonpositive() -> None:
    reg = _registry()
    reg.inc_voice_speaker_degraded("usb_speaker", amount=0)
    assert "voice_speaker_degraded_total" not in reg.render_prometheus()


def test_speaker_counter_drops_invalid_subsystem() -> None:
    reg = _registry()
    reg.inc_voice_speaker_degraded("device name leaked /dev/ttyUSB0")  # free text
    assert "voice_speaker_degraded_total" not in reg.render_prometheus()


def test_speaker_counter_accepts_full_valid_subsystem_set() -> None:
    reg = _registry()
    for subsystem in ("usb_speaker", "rocky_fallback"):
        reg.inc_voice_speaker_degraded(subsystem)
    assert reg.render_prometheus().count("voice_speaker_degraded_total{") == 2


# --------------------------------------------------------------------------- #
# TTS synthesis-failure counter
# --------------------------------------------------------------------------- #
def test_tts_counter_renders_per_api() -> None:
    reg = _registry()
    reg.inc_voice_tts_synthesize_failures("synthesize")
    reg.inc_voice_tts_synthesize_failures("synthesize")
    reg.inc_voice_tts_synthesize_failures("synthesize_wav_file")
    out = reg.render_prometheus()
    assert 'mousedroid_voice_tts_synthesize_failures_total{api="synthesize"} 2' in out
    assert 'mousedroid_voice_tts_synthesize_failures_total{api="synthesize_wav_file"} 1' in out


def test_tts_counter_noop_on_nonpositive() -> None:
    reg = _registry()
    reg.inc_voice_tts_synthesize_failures("synthesize", amount=0)
    assert "voice_tts_synthesize_failures_total" not in reg.render_prometheus()


def test_tts_counter_drops_invalid_api() -> None:
    reg = _registry()
    reg.inc_voice_tts_synthesize_failures("synthesize_wav(text,wav_file)")  # log label form
    assert "voice_tts_synthesize_failures_total" not in reg.render_prometheus()


def test_tts_counter_accepts_full_valid_api_set() -> None:
    reg = _registry()
    for api in ("synthesize", "synthesize_wav", "synthesize_wav_file"):
        reg.inc_voice_tts_synthesize_failures(api)
    assert reg.render_prometheus().count("voice_tts_synthesize_failures_total{") == 3


# --------------------------------------------------------------------------- #
# Label-constant sets are pinned (single source of truth)
# --------------------------------------------------------------------------- #
def test_label_constant_sets_are_pinned() -> None:
    assert (
        frozenset({"usb_speaker", "rocky_fallback"})
        == metrics_mod._VOICE_SPEAKER_DEGRADED_SUBSYSTEMS
    )
    assert (
        frozenset({"synthesize", "synthesize_wav", "synthesize_wav_file"})
        == metrics_mod._VOICE_TTS_APIS
    )


# --------------------------------------------------------------------------- #
# track_voice_degradation flag gates BOTH families
# --------------------------------------------------------------------------- #
def test_track_flag_off_suppresses_both_families() -> None:
    reg = _registry(track_voice_degradation=False)
    reg.inc_voice_speaker_degraded("usb_speaker")
    reg.inc_voice_tts_synthesize_failures("synthesize")
    out = reg.render_prometheus()
    assert "voice_speaker_degraded_total" not in out
    assert "voice_tts_synthesize_failures_total" not in out


# --------------------------------------------------------------------------- #
# promtool seeding contract
# --------------------------------------------------------------------------- #
def test_generate_metrics_sample_seeds_both_families() -> None:
    sample = generate_metrics_sample()
    assert "voice_speaker_degraded_total" in sample
    assert "voice_tts_synthesize_failures_total" in sample
