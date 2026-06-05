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
    import numpy as np
    from numpy.typing import NDArray

    from mousedroid.commentary.protocol import (
        CommentaryComposerProtocol,
        CommentaryFacts,
        GroundedReferentStoreProtocol,
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
# Phase-1: a recognised place was hit again within its recognition cooldown — we
# stay quiet (and skip the novelty path, since the place is already known).
_REASON_RECOGNITION_COOLDOWN = "recognition_cooldown"

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
        _REASON_RECOGNITION_COOLDOWN,
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
        referent_store: GroundedReferentStoreProtocol | None = None,
        embedding_dim: int | None = None,
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
            referent_store: Optional Phase-1 referent store. When supplied AND
                ``cfg.recognition_enabled``, the engine stores the embedding of
                each fired novelty (keyed by the spoken phrase) and narrates
                recognition on a close match. ``None`` (default) keeps Phase-0
                behaviour byte-identical.
            embedding_dim: Expected referent-embedding width (``semantic_dim``);
                a fact embedding of any other width is skipped (with a one-time
                warning) so a CfC-enabled rover never crashes the FAISS store.
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

        # Phase-1 recognition state.
        self._referent_store = referent_store
        self._embedding_dim = embedding_dim
        self._recognition_active = referent_store is not None and cfg.recognition_enabled
        self._last_recognition_t: float = -math.inf
        self._referent_count: int = 0
        self._dim_warned: bool = False
        self._full_warned: bool = False

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
        # Phase-1 recognition: a known place short-circuits the novelty path —
        # either narrating recognition or staying quiet within its cooldown.
        if await self._handle_recognition(facts, now):
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
        # Learn this (unrecognised) place: store its embedding keyed by the plain
        # spoken phrase so a later visit recalls "last time I said: <text>".
        self._store_referent(facts, text)

    # -- Phase-1 recognition -----------------------------------------------

    def _usable_embedding(self, facts: CommentaryFacts) -> NDArray[np.float32] | None:
        """Return the fact embedding iff recognition is active + dim matches."""
        if not self._recognition_active or facts.embedding is None:
            return None
        emb = facts.embedding
        if self._embedding_dim is not None and int(emb.shape[0]) != self._embedding_dim:
            if not self._dim_warned:
                _log.warning(
                    "commentary_recognition_dim_mismatch",
                    got=int(emb.shape[0]),
                    expected=self._embedding_dim,
                )
                self._dim_warned = True
            return None
        return emb

    def _nearest_referent(self, facts: CommentaryFacts) -> str | None:
        """Return the recalled phrase of a recognised place, or ``None``."""
        emb = self._usable_embedding(facts)
        if emb is None or self._referent_store is None:
            return None
        results = self._referent_store.retrieve(emb, k=1)
        if not results:
            return None
        key, distance = results[0]
        # DEBUG so an operator can calibrate ``recognition_distance_threshold``
        # against the live nearest distance.
        _log.debug("commentary_recognition_probe", distance=distance, nearest=key)
        return key if distance <= self._cfg.recognition_distance_threshold else None

    async def _handle_recognition(self, facts: CommentaryFacts, now: float) -> bool:
        """Handle a recognised place; return ``True`` iff it owns this evaluation.

        Returns ``False`` for an unknown place so the caller falls through to the
        novelty path. A recognised place either narrates recognition (when its
        cooldown elapsed) or stays quiet — in both cases it owns the evaluation
        (the novelty path is skipped, since the place is already known).
        """
        recalled = self._nearest_referent(facts)
        if recalled is None:
            return False
        if (now - self._last_recognition_t) < self._cfg.recognition_min_interval_s:
            self._suppress(_REASON_RECOGNITION_COOLDOWN)
            return True
        styled = self._style(self._cfg.recognition_template.format(phrase=recalled))
        if not styled:
            self._suppress(_REASON_EMPTY_AFTER_TRANSFORM)
            return True
        await self._voice.play_phrase(styled)
        self._last_recognition_t = now
        if self._metrics is not None:
            self._metrics.inc_commentary_recognitions()
        _log.info("commentary_recognition_spoken", text=styled, recalled=recalled)
        return True

    def _store_referent(self, facts: CommentaryFacts, phrase: str) -> None:
        """Persist this (new) place's embedding keyed by the spoken phrase."""
        emb = self._usable_embedding(facts)
        if emb is None or self._referent_store is None:
            return
        if self._referent_count >= self._cfg.recognition_max_referents:
            if not self._full_warned:
                _log.warning(
                    "commentary_referent_store_full",
                    cap=self._cfg.recognition_max_referents,
                )
                self._full_warned = True
            return
        self._referent_store.store(phrase, emb)
        self._referent_count += 1
        if self._metrics is not None:
            self._metrics.inc_commentary_referents_stored()
        _log.debug("commentary_referent_stored", phrase=phrase, count=self._referent_count)

    def _style(self, text: str) -> str:
        """Apply Rocky styling, forwarding the operator intensity threshold."""
        kwargs: dict[str, float] = {"intensity": self._cfg.excitement_intensity}
        if self._intensity_threshold is not None:
            kwargs["intensity_threshold"] = self._intensity_threshold
        return rocky_transform(text, **kwargs)


__all__ = ["SUPPRESSION_REASONS", "CommentaryEngine"]
