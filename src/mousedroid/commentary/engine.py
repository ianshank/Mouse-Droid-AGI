"""CommentaryEngine — out-of-loop novelty/safety-gated narration.

Owns a scale-invariant statistical model of the novelty stream (exponentially-
weighted mean + variance, West incremental form — numerically stable, forgets the
non-stationary drift of the forward-model MSE) and a background cadence loop that,
when a window is a statistical outlier AND the droid is idle/safe AND outside its
cooldown, composes one plain line, styles it with ``rocky_transform``, and speaks
it via ``play_phrase``. Never raises into its task; all timing via an injected
:class:`ClockProtocol` for deterministic tests.
"""

from __future__ import annotations

import asyncio
import math
from typing import TYPE_CHECKING

from mousedroid.commentary.protocol import SpeakerBusyProtocol
from mousedroid.common.time.protocol import RealClock
from mousedroid.logging.setup import get_logger
from mousedroid.voice.rocky import rocky_transform

if TYPE_CHECKING:
    from mousedroid.commentary.protocol import (
        CommentaryComposerProtocol,
        CommentaryFacts,
    )
    from mousedroid.common.time.protocol import ClockProtocol
    from mousedroid.config.schema import CommentaryConfig
    from mousedroid.telemetry.metrics import MetricsRegistry
    from mousedroid.voice.protocol import VoiceEngineProtocol

_log = get_logger(__name__)

# Suppression reasons (fixed low-cardinality label set — keep in sync with the
# AQA label-hygiene test). Each gate records exactly one of these.
_REASON_NO_VOICE = "no_voice"
_REASON_EMERGENCY = "emergency"
_REASON_BUSY = "busy"
_REASON_NO_FACTS = "no_facts"
_REASON_NOT_IDLE = "not_idle"
_REASON_BELOW_THRESHOLD = "below_threshold"
_REASON_NO_NOVELTY_SIGNAL = "no_novelty_signal"
_REASON_COOLDOWN = "cooldown"
_REASON_EMPTY = "empty"
_REASON_EMPTY_AFTER_TRANSFORM = "empty_after_transform"

SUPPRESSION_REASONS: frozenset[str] = frozenset(
    {
        _REASON_NO_VOICE,
        _REASON_EMERGENCY,
        _REASON_BUSY,
        _REASON_NO_FACTS,
        _REASON_NOT_IDLE,
        _REASON_BELOW_THRESHOLD,
        _REASON_NO_NOVELTY_SIGNAL,
        _REASON_COOLDOWN,
        _REASON_EMPTY,
        _REASON_EMPTY_AFTER_TRANSFORM,
    }
)


class CommentaryEngine:
    """Out-of-loop, novelty/safety-gated spoken commentary engine."""

    def __init__(
        self,
        cfg: CommentaryConfig,
        *,
        voice_engine: VoiceEngineProtocol,
        composer: CommentaryComposerProtocol,
        metrics: MetricsRegistry | None = None,
        clock: ClockProtocol | None = None,
        intensity_threshold: float | None = None,
    ) -> None:
        """Initialise the engine.

        Args:
            cfg: Resolved :class:`CommentaryConfig`.
            voice_engine: Started voice engine (caller owns lifecycle). If it
                also satisfies :class:`SpeakerBusyProtocol`, commentary defers
                while it is speaking.
            composer: Plain-text composer (template or LLM).
            metrics: Optional shared registry (no-op when ``None``).
            clock: Injected clock (defaults to :class:`RealClock`); drives the
                cadence sleep and all cooldown/quiet-window timing.
            intensity_threshold: Forwarded ``VoiceConfig.intensity_threshold`` so
                an operator override isn't shadowed (the greeting lesson).
        """
        self._cfg = cfg
        self._voice = voice_engine
        self._composer = composer
        self._metrics = metrics
        self._clock: ClockProtocol = clock if clock is not None else RealClock()
        self._intensity_threshold = intensity_threshold

        # EW novelty statistics (West incremental mean + variance).
        self._mean: float = 0.0
        self._var: float = 0.0
        self._count: int = 0
        self._seen_novelty: bool = False

        # Per-window accumulators (reset every run() evaluation — P2).
        self._window_facts: CommentaryFacts | None = None
        self._window_outlier: bool = False
        self._peak_novelty: float = -math.inf

        # Emergency / cooldown state (monotonic seconds).
        self._emergency_active: bool = False
        self._last_emergency_t: float = -math.inf
        self._last_fire_t: float = -math.inf
        self._stopped: bool = False

    # -- Hot-path hooks (cheap) --------------------------------------------

    def observe_emergency(self, is_emergency: bool) -> None:
        """Record this tick's emergency state (O(1), called every control tick)."""
        if is_emergency:
            self._emergency_active = True
            self._last_emergency_t = self._clock.monotonic()
        else:
            self._emergency_active = False

    def observe(self, novelty: float | None, facts: CommentaryFacts | None) -> None:
        """Feed one strided novelty sample + facts snapshot (cheap, ~2 Hz).

        Compares each sample against the distribution BEFORE folding it in
        (no self-masking — P2), updates the EW stats, and tracks the window's
        peak-novelty facts.
        """
        if novelty is not None:
            self._seen_novelty = True
            is_outlier = (
                self._count >= self._cfg.novelty_warmup_n
                and self._std() >= self._cfg.novelty_std_floor
                and novelty > self._mean + self._cfg.novelty_sigma * self._std()
            )
            self._update_stats(novelty)
            if facts is not None and novelty >= self._peak_novelty:
                self._peak_novelty = novelty
                self._window_facts = facts
            self._window_outlier = self._window_outlier or is_outlier
        elif facts is not None:
            # No novelty signal — retain latest facts for the
            # allow_without_novelty cadence path.
            self._window_facts = facts

    # -- Background loop ----------------------------------------------------

    async def run(self) -> None:
        """Background cadence loop: evaluate the gate and maybe speak."""
        _log.info(
            "commentary_loop_started",
            cadence_s=self._cfg.cadence_s,
            composer=type(self._composer).__name__,
        )
        while not self._stopped:
            await self._clock.sleep(self._cfg.cadence_s)
            try:
                await self._evaluate_and_speak()
            except asyncio.CancelledError:
                raise
            except Exception:
                _log.warning("commentary_cycle_failed", exc_info=True)

    async def stop(self) -> None:
        """Signal shutdown; the spawning task is cancelled+drained externally."""
        self._stopped = True

    # -- Internals ----------------------------------------------------------

    def _std(self) -> float:
        """Standard deviation from the EW variance (floored at 0)."""
        return math.sqrt(max(self._var, 0.0))

    def _update_stats(self, value: float) -> None:
        """Fold ``value`` into the EW mean/variance (West incremental form)."""
        if self._count == 0:
            self._mean = value
            self._var = 0.0
        else:
            alpha = self._cfg.novelty_gate_alpha
            diff = value - self._mean
            incr = alpha * diff
            self._mean += incr
            self._var = (1.0 - alpha) * (self._var + diff * incr)
        self._count += 1

    def _drain_window(self) -> tuple[CommentaryFacts | None, bool, float]:
        """Snapshot + reset the per-window accumulators (peak reset — P2)."""
        facts = self._window_facts
        outlier = self._window_outlier
        peak = self._peak_novelty
        self._window_facts = None
        self._window_outlier = False
        self._peak_novelty = -math.inf
        return facts, outlier, peak

    def _suppress(self, reason: str) -> None:
        # DEBUG so an operator can answer "why is it silent?" by enabling debug
        # logs — without adding production-log noise (suppression is the common
        # case). The Prometheus counter is the always-on aggregate signal.
        _log.debug("commentary_suppressed", reason=reason)
        if self._metrics is not None:
            self._metrics.inc_commentary_suppressed(reason)

    async def _evaluate_and_speak(self) -> None:
        """One gate evaluation; speak iff every gate passes."""
        facts, outlier, peak = self._drain_window()
        now = self._clock.monotonic()
        if self._metrics is not None:
            self._metrics.inc_commentary_considered()
        # DEBUG snapshot of the gate inputs — the single most useful signal for
        # tuning ``novelty_sigma`` / ``novelty_gate_alpha`` on the rover (mean,
        # std, peak, sample count) and for diagnosing an unexpectedly quiet
        # droid. Off by default; no hot-loop cost (this runs at ``cadence_s``).
        _log.debug(
            "commentary_evaluating",
            peak_novelty=peak if math.isfinite(peak) else None,
            novelty_mean=self._mean,
            novelty_std=self._std(),
            samples=self._count,
            outlier=outlier,
            has_facts=facts is not None,
            emergency_active=self._emergency_active,
        )

        if self._voice is None:  # defensive; the factory always wires a voice
            self._suppress(_REASON_NO_VOICE)
            return
        # Emergency hard-gate, re-checked here at fire time (P1): the live
        # per-tick flag, the post-emergency quiet window, and this window's
        # captured emergency state.
        in_quiet = (now - self._last_emergency_t) < self._cfg.post_emergency_quiet_s
        if self._emergency_active or in_quiet or (facts is not None and facts.is_emergency):
            self._suppress(_REASON_EMERGENCY)
            return
        if isinstance(self._voice, SpeakerBusyProtocol) and self._voice.is_speaking:
            self._suppress(_REASON_BUSY)
            return
        if facts is None:
            self._suppress(_REASON_NO_FACTS)
            return
        # Idle/safe gate (P1): only muse when slow and clear.
        if facts.speed_mps > self._cfg.idle_speed_mps or (
            facts.min_clearance_m < self._cfg.idle_min_clearance_m
        ):
            self._suppress(_REASON_NOT_IDLE)
            return
        # Novelty gate.
        if self._seen_novelty:
            if not outlier:
                self._suppress(_REASON_BELOW_THRESHOLD)
                return
        elif not self._cfg.allow_without_novelty:
            self._suppress(_REASON_NO_NOVELTY_SIGNAL)
            return
        if (now - self._last_fire_t) < self._cfg.min_interval_s:
            self._suppress(_REASON_COOLDOWN)
            return

        compose_start = self._clock.monotonic()
        text = await self._composer.compose(facts)
        if self._metrics is not None:
            self._metrics.observe_commentary_compose_seconds(
                self._clock.monotonic() - compose_start
            )
        if not text:
            self._suppress(_REASON_EMPTY)
            return
        styled = self._style(text)
        if not styled:
            self._suppress(_REASON_EMPTY_AFTER_TRANSFORM)
            return

        await self._voice.play_phrase(styled)
        self._last_fire_t = now
        if self._metrics is not None:
            self._metrics.inc_commentary_emitted()
            if self._seen_novelty and math.isfinite(peak):
                self._metrics.set_commentary_novelty(peak)
        _log.info("commentary_spoken", text=styled, novelty=peak if self._seen_novelty else None)

    def _style(self, text: str) -> str:
        """Apply Rocky styling, forwarding the operator intensity threshold."""
        kwargs: dict[str, float] = {"intensity": self._cfg.excitement_intensity}
        if self._intensity_threshold is not None:
            kwargs["intensity_threshold"] = self._intensity_threshold
        return rocky_transform(text, **kwargs)


__all__ = ["SUPPRESSION_REASONS", "CommentaryEngine"]
