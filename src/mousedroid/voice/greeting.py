"""Operator-tools: MSE-6 spoken greeting subsystem.

Drives a one-shot named greeting through the existing
:class:`~mousedroid.voice.protocol.VoiceEngineProtocol`:

1. Plays a pre-flourish phrase-bank event (``greeting_excited`` by
   default) as an MSE-6-style audible chirp.
2. Waits ``inter_chirp_delay_s`` to keep the chirp tail audible.
3. Synthesises and plays the operator-configured ``message_template``
   with names interpolated through an Oxford-comma helper, after
   running it through ``rocky_transform`` for excited prosody.

The OLED face controller is deliberately **NOT** wired here — the
operator's current dev rover has no SSD1306 attached. ``Greeter``
exposes a ``face`` kwarg slot so the face can be plugged in later
without touching this module.

Reuses the existing voice stack (no reimplementation):

* :class:`~mousedroid.voice.rocky.RockyVoiceEngine` —
  ``play_phrase(text)`` for synchronous one-shot delivery.
* :mod:`mousedroid.voice.phrase_bank` — sources pre-flourish text by
  event name (deterministic first-entry pick for reproducibility).
* :func:`~mousedroid.voice.rocky.rocky_transform` — applies the
  personality engine's excited stylisation.

Architecture invariants (CLAUDE.md):

* Protocol-based DI — typed against ``VoiceEngineProtocol``, never
  ``RockyVoiceEngine``.
* No hardcoded values — every tunable lives in :class:`GreetingConfig`.
* Structured logging via :mod:`mousedroid.logging.setup`.
* Asyncio-only — no threading.
* Strict typing — passes ``mypy --strict``.
"""

from __future__ import annotations

import asyncio
import random
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from mousedroid.common.text_utils import format_names_oxford
from mousedroid.logging.setup import get_logger
from mousedroid.voice.phrase_bank import DEFAULT_PHRASES
from mousedroid.voice.rocky import rocky_transform

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mousedroid.config.schema import GreetingConfig
    from mousedroid.voice.protocol import VoiceEngineProtocol


@runtime_checkable
class GreeterProtocol(Protocol):
    """Structural type for the one-shot greeting subsystem.

    Lets the orchestrator depend on the greeting capability without
    importing the concrete :class:`Greeter` (Architecture Invariant 1:
    protocol-based DI). The factory wires a concrete ``Greeter`` in.
    """

    async def greet(self, names: Sequence[str] | None = None) -> None:
        """Play the greeting once. See :meth:`Greeter.greet`."""
        ...

_log = get_logger(__name__)

# Cap on how many phrase-bank events surface in the
# ``greeting_pre_chirp_event_unknown`` warning log. Tunable here only —
# the warning is operator-diagnostic, not a hot path, so this is fine as
# a module-level named constant rather than a config field.
_AVAILABLE_EVENTS_LOG_LIMIT: int = 8


def _select_chirp_text(event_name: str, rng: random.Random | None = None) -> str | None:
    """Resolve a phrase-bank ``event_name`` to a single text string.

    Returns ``None`` when the event is empty / unknown / has no entries
    so the caller can skip the pre-flourish without raising. ``rng`` is
    injectable for deterministic tests; ``None`` picks the first entry
    (also deterministic — no clock-coupled randomness for the dev path).

    Args:
        event_name: Phrase-bank event key (e.g. ``"greeting_excited"``).
        rng: Optional :class:`random.Random` for entry selection.

    Returns:
        A single phrase string, or ``None`` if the event has no usable
        entries.
    """
    if not event_name:
        return None
    entries = DEFAULT_PHRASES.get(event_name)
    if not entries:
        # ``event`` is the structlog ``BoundLogger.warning`` positional —
        # cannot be reused as a kwarg here (mypy + runtime TypeError).
        # Use a distinct key for the operator-visible phrase-event name.
        _log.warning(
            "greeting_pre_chirp_event_unknown",
            phrase_event=event_name,
            available=sorted(DEFAULT_PHRASES.keys())[:_AVAILABLE_EVENTS_LOG_LIMIT],
        )
        return None
    if rng is None:
        # Deterministic first-entry pick keeps dev/test reproducible without
        # a clock-coupled seed. Operators wanting variety provide an `rng`.
        return entries[0]
    return rng.choice(entries)


class Greeter:
    """Plays the operator-configured greeting through a voice engine.

    Constructed by :func:`mousedroid.factory.build_greeter`. Conforms
    to no formal protocol — single-purpose helper exposed only through
    the factory so callers don't import the concrete type.

    The ``face`` kwarg slot is a documented extension point for the
    SSD1306 face animation deferred per the operator's "no OLED
    attached" note. When a face driver is plugged in later, the
    coordination would be a small ``asyncio.gather`` block before the
    ``play_phrase`` await.
    """

    def __init__(
        self,
        voice_engine: VoiceEngineProtocol,
        cfg: GreetingConfig,
        *,
        intensity_threshold: float | None = None,
        rng: random.Random | None = None,
    ) -> None:
        """Initialise the greeter.

        Args:
            voice_engine: Started :class:`VoiceEngineProtocol`
                instance. Caller owns its lifecycle (call ``start()``
                BEFORE and ``stop()`` AFTER ``greet``); the greeter
                deliberately does not manage engine lifecycle so the
                same engine instance can be reused by the orchestrator
                without competing for start/stop.
            cfg: Resolved :class:`GreetingConfig` from
                ``Settings.greeting``. Caller MUST pass a non-``None``
                config; the factory enforces that.
            intensity_threshold: Threshold passed to
                :func:`rocky_transform` — effects activate when the
                excitement intensity exceeds this value. ``None``
                (default) lets ``rocky_transform`` fall back to its
                own pinned default. The factory passes the resolved
                ``VoiceConfig.intensity_threshold`` so an operator
                tuning the voice config sees the same threshold here.
            rng: Optional :class:`random.Random` for chirp-entry
                selection. Defaults to ``None`` which picks the first
                phrase-bank entry deterministically.
        """
        self._voice = voice_engine
        self._cfg = cfg
        self._intensity_threshold = intensity_threshold
        self._rng = rng

    @property
    def voice_engine(self) -> VoiceEngineProtocol:
        """Read-only handle on the wrapped voice engine.

        Exposed so callers (e.g. ``scripts/greet_intro.py``) can manage
        the engine's ``start()`` / ``stop()`` lifecycle without piercing
        ``_voice`` private attribute access (code-reviewer round-1
        finding #3: a ``type: ignore`` suppressing the private-access
        warning hid the contract that the caller owns the lifecycle).
        Typed against the protocol so an internal rename never silently
        breaks callers.
        """
        return self._voice

    async def greet(self, names: Sequence[str] | None = None) -> None:
        """Play the greeting once.

        Resolves the names (parameter override OR ``cfg.names``),
        fires the pre-flourish phrase (if configured), waits the
        inter-chirp delay, then plays the rocky-transformed custom
        message. ALL operations sequential through the same speaker
        so the chirp and message don't interleave.

        Args:
            names: Optional override for ``cfg.names``. ``None``
                (default) uses the configured list. Provided for
                tests + future operator-tools needing one-off names
                without re-loading config; the YAML overlay remains
                the canonical source.

        Raises:
            ValueError: If the resolved name list is empty (caller
                error — the config validator catches the YAML
                case at parse time, but a runtime override with
                ``[]`` still reaches us here).
        """
        resolved = list(names) if names is not None else list(self._cfg.names)
        if not resolved:
            msg = "Greeter.greet() requires at least one name"
            raise ValueError(msg)

        _log.info(
            "greeting_started",
            name_count=len(resolved),
            pre_chirp_event=self._cfg.pre_chirp_event or "(none)",
            excitement=self._cfg.excitement_intensity,
        )

        # Pre-flourish (skip silently if event empty / unknown).
        chirp_text = _select_chirp_text(self._cfg.pre_chirp_event, rng=self._rng)
        if chirp_text is not None:
            _log.debug("greeting_chirp_playing", text=chirp_text)
            samples, peak = await self._voice.play_phrase(chirp_text)
            _log.debug("greeting_chirp_done", samples=samples, peak_abs=peak)
            if self._cfg.inter_chirp_delay_s > 0:
                await asyncio.sleep(self._cfg.inter_chirp_delay_s)

        # Custom message with rocky-style excitement. Forward the
        # operator-configured ``VoiceConfig.intensity_threshold`` so the
        # personality engine fires at the same threshold the rest of the
        # voice subsystem uses — silent shadowing of an operator override
        # would violate CLAUDE.md "no hardcoded values".
        oxford_names = format_names_oxford(resolved)
        raw_message = self._cfg.message_template.format(names=oxford_names)
        rocky_kwargs: dict[str, float] = {"intensity": self._cfg.excitement_intensity}
        if self._intensity_threshold is not None:
            rocky_kwargs["intensity_threshold"] = self._intensity_threshold
        styled = rocky_transform(raw_message, **rocky_kwargs)
        _log.info(
            "greeting_message_playing",
            text=styled,
            names=oxford_names,
            len=len(styled),
        )
        samples, peak = await self._voice.play_phrase(styled)
        _log.info(
            "greeting_done",
            samples=samples,
            peak_abs=peak,
        )


__all__ = ["Greeter", "GreeterProtocol", "format_names_oxford"]
