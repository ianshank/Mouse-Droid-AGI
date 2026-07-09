"""AQA: schema-field hygiene + counter cardinality for voice-degradation metrics.

Architectural-quality assertions locking the contracts a future refactor could
silently break: the ``track_voice_degradation`` gate flag (documented,
default-on), the registry helper names (rename guard), the fixed
low-cardinality label sets, the keyword-only ``metrics`` params threaded through
the voice factory builders, and the byte-identical ``/metrics`` render when the
families are unused.
"""

from __future__ import annotations

import inspect

from mousedroid.config.schema import MetricsConfig
from mousedroid.factory import build_speaker, build_voice_engine
from mousedroid.telemetry import metrics as metrics_mod
from mousedroid.telemetry.metrics import MetricsRegistry


def test_metrics_gate_flag_documented_and_default_on() -> None:
    field = MetricsConfig.model_fields["track_voice_degradation"]
    assert field.default is True
    assert field.description, "track_voice_degradation must carry an operator description"


def test_registry_helpers_exist_rename_guard() -> None:
    for name in ("inc_voice_speaker_degraded", "inc_voice_tts_synthesize_failures"):
        assert callable(getattr(MetricsRegistry, name, None)), f"missing registry helper: {name}"


def test_label_value_sets_are_pinned() -> None:
    assert (
        frozenset({"usb_speaker", "rocky_fallback"})
        == metrics_mod._VOICE_SPEAKER_DEGRADED_SUBSYSTEMS
    )
    assert (
        frozenset({"synthesize", "synthesize_wav", "synthesize_wav_file"})
        == metrics_mod._VOICE_TTS_APIS
    )


def test_speaker_subsystem_label_set_fixed() -> None:
    """Render reflects only the fixed low-cardinality subsystem enum."""
    reg = MetricsRegistry(MetricsConfig())
    for subsystem in ("usb_speaker", "rocky_fallback"):
        reg.inc_voice_speaker_degraded(subsystem)
    assert reg.render_prometheus().count("voice_speaker_degraded_total{") == 2


def test_tts_api_label_set_fixed() -> None:
    """Render reflects only the fixed low-cardinality api enum."""
    reg = MetricsRegistry(MetricsConfig())
    for api in ("synthesize", "synthesize_wav", "synthesize_wav_file"):
        reg.inc_voice_tts_synthesize_failures(api)
    assert reg.render_prometheus().count("voice_tts_synthesize_failures_total{") == 3


def test_build_speaker_metrics_keyword_only_none_default() -> None:
    param = inspect.signature(build_speaker).parameters["metrics"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.default is None


def test_build_voice_engine_metrics_keyword_only_none_default() -> None:
    param = inspect.signature(build_voice_engine).parameters["metrics"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.default is None


def test_metrics_byte_identical_when_families_unused() -> None:
    """A registry with no voice-degradation write renders nothing for the families."""
    out = MetricsRegistry(MetricsConfig()).render_prometheus()
    assert "voice_speaker_degraded_total" not in out
    assert "voice_tts_synthesize_failures_total" not in out
