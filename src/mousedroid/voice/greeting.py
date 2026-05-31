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
from typing import TYPE_CHECKING

from mousedroid.logging.setup import get_logger
from mousedroid.voice.phrase_bank import DEFAULT_PHRASES
from mousedroid.voice.rocky import rocky_transform

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mousedroid.config.schema import GreetingConfig
    from mousedroid.voice.protocol import VoiceEngineProtocol

_log = get_logger(__name__)


def format_names_oxford(names: Sequence[str]) -> str:
    """Format ``names`` as an Oxford-comma list (``"A, B, C and D"``).

    Empty list returns ``""``; single name returns the name verbatim;
    two names join with ``" and "``; three or more use commas + Oxford
    ``" and "`` before the final item. Pure function, no I/O — pinned
    by ``tests/unit/voice/test_greeting.py``.

    Args:
        names: Ordered list of names to format.

    Returns:
        Oxford-comma-joined string.
    """
    cleaned = [n.strip() for n in names if n.strip()]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"
    return f"{', '.join(cleaned[:-1])} and {cleaned[-1]}"


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
            available=sorted(DEFAULT_PHRASES.keys())[:8],
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
            rng: Optional :class:`random.Random` for chirp-entry
                selection. Defaults to ``None`` which picks the first
                phrase-bank entry deterministically.
        """
        self._voice = voice_engine
        self._cfg = cfg
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

        # Custom message with rocky-style excitement.
        oxford_names = format_names_oxford(resolved)
        raw_message = self._cfg.message_template.format(names=oxford_names)
        styled = rocky_transform(raw_message, intensity=self._cfg.excitement_intensity)
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


__all__ = ["Greeter", "format_names_oxford"]
