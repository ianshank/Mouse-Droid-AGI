"""Voice speaker-degradation + TTS synthesis-failure metrics.

Distinct from the Phase-7 ``inc_voice_event`` counter in
``_registry_phase7.py`` (generic voice lifecycle events like ``"startup"``) —
this family is specifically about *degradation*: reconnect exhaustion,
MockSpeaker fallback, and Piper synthesis failures.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mousedroid.telemetry.metrics.primitives import (
    _VOICE_SPEAKER_DEGRADED_SUBSYSTEMS,
    _VOICE_TTS_APIS,
    _LabeledCounter,
    _log,
    _render_labeled_counter,
)

if TYPE_CHECKING:
    from mousedroid.config.schema import MetricsConfig


class _VoiceMetricsMixin:
    """Voice speaker-degradation + TTS synthesis-failure metric family."""

    # Populated by ``_CoreMetricsMixin._init_core_metrics``, which always runs
    # first from ``MetricsRegistry.__init__``.
    _cfg: MetricsConfig

    def _init_voice_metrics(self, cfg: MetricsConfig) -> None:
        """Initialise voice-degradation counters.

        Args:
            cfg: Metrics configuration with namespace and toggle flags.
        """
        ns = cfg.namespace

        # Voice-degradation counters. Pure-add: each omitted from /metrics until
        # its first increment; both gated by cfg.track_voice_degradation.
        self._voice_speaker_degraded = _LabeledCounter()
        self._voice_tts_synthesize_failures = _LabeledCounter()

        # Voice-degradation counter names (render helper suffixes ``_total``).
        self._name_voice_speaker_degraded = f"{ns}_voice_speaker_degraded"
        self._name_voice_tts_synthesize_failures = f"{ns}_voice_tts_synthesize_failures"

    def inc_voice_speaker_degraded(self, subsystem: str, amount: int = 1) -> None:
        """Increment the voice speaker-degradation counter (label: subsystem).

        Fired when a speaker path degrades: the USB speaker exhausts its
        reconnect retries, or the voice engine downgrades to a MockSpeaker so
        the orchestrator keeps running silently. Pure-add and gated by
        ``cfg.track_voice_degradation``.

        Args:
            subsystem: One of ``"usb_speaker"`` (the ``UsbSpeaker`` driver gave
                up after ``reconnect_max_attempts``) or ``"rocky_fallback"``
                (``RockyVoiceEngine`` caught ``SpeakerUnavailableError`` and
                swapped in a MockSpeaker). Out-of-set values are dropped with a
                DEBUG log so a driver string never leaks cardinality.
            amount: Increment magnitude (default 1); ``<= 0`` is a no-op.
        """
        if not self._cfg.track_voice_degradation or amount <= 0:
            return
        if subsystem not in _VOICE_SPEAKER_DEGRADED_SUBSYSTEMS:
            _log.debug("voice_speaker_degraded_dropped_invalid_subsystem", subsystem=subsystem)
            return
        self._voice_speaker_degraded.inc(subsystem, amount)

    def inc_voice_tts_synthesize_failures(self, api: str, amount: int = 1) -> None:
        """Increment the TTS synthesis-failure counter (label: api).

        Fired when a Piper synthesis call raises and the engine returns silence.
        Pure-add and gated by ``cfg.track_voice_degradation``.

        Args:
            api: The resolved synthesis API — one of ``"synthesize"`` (legacy
                raw path), ``"synthesize_wav"`` (piper ``synthesize_wav(text)``),
                or ``"synthesize_wav_file"`` (piper
                ``synthesize_wav(text, wav_file)``). Out-of-set values are
                dropped with a DEBUG log so a runtime string never leaks
                cardinality.
            amount: Increment magnitude (default 1); ``<= 0`` is a no-op.
        """
        if not self._cfg.track_voice_degradation or amount <= 0:
            return
        if api not in _VOICE_TTS_APIS:
            _log.debug("voice_tts_synthesize_failures_dropped_invalid_api", api=api)
            return
        self._voice_tts_synthesize_failures.inc(api, amount)

    # ------------------------------------------------------------------
    # Prometheus text exposition — family renderer
    # ------------------------------------------------------------------

    def _families_voice_degradation(self) -> list[list[str]]:
        """Voice speaker-degradation + TTS synthesis-failure counters."""
        cfg = self._cfg
        out: list[list[str]] = []
        # Both families are pure-add: emitted only after the first increment
        # (snapshot non-empty), so default deployments render byte-identically.
        if cfg.track_voice_degradation:
            speaker_snapshot = self._voice_speaker_degraded.snapshot()
            if speaker_snapshot:
                out.append(
                    _render_labeled_counter(
                        self._name_voice_speaker_degraded,
                        "Voice speaker degradations (label: subsystem)",
                        "subsystem",
                        speaker_snapshot,
                    )
                )
            tts_snapshot = self._voice_tts_synthesize_failures.snapshot()
            if tts_snapshot:
                out.append(
                    _render_labeled_counter(
                        self._name_voice_tts_synthesize_failures,
                        "Voice TTS synthesis failures (label: api)",
                        "api",
                        tts_snapshot,
                    )
                )
        return out
