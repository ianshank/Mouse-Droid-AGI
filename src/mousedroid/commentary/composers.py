"""Commentary composers: deterministic template (offline) + LLM (rich).

Both satisfy :class:`CommentaryComposerProtocol` and return PLAIN English — the
engine applies :func:`rocky_transform` uniformly, so neither composer pre-styles.
Both phrase strictly over the grounded numeric facts and NEVER name objects (the
mouse-droid loop has no object labels).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mousedroid.logging.setup import get_logger
from mousedroid.security.injection_filter import InjectionRejected

if TYPE_CHECKING:
    from mousedroid.commentary.protocol import CommentaryFacts
    from mousedroid.config.schema import CommentaryConfig
    from mousedroid.llm_gateway.protocol import QueryCapableLLMProtocol

_log = get_logger(__name__)

# Classification key constants — must exist in ``cfg.templates`` (validator
# enforces only the mandatory ``"default"``; missing optional keys fall back).
_KEY_LOW_BATTERY = "low_battery"
_KEY_TIGHT_SPACE = "tight_space"
_KEY_LOUD = "loud"
_KEY_MOVING_FAST = "moving_fast"
_KEY_OPEN_SPACE = "open_space"
_KEY_DEFAULT = "default"


class TemplateCommentaryComposer:
    """Deterministic, offline composer: classify facts -> a config template.

    Classification is config-driven (every threshold is a ``CommentaryConfig``
    field) and honours the ``*_valid`` flags so it never claims a clearance /
    loudness it doesn't actually have.
    """

    def __init__(self, cfg: CommentaryConfig) -> None:
        self._cfg = cfg

    def _classify(self, facts: CommentaryFacts) -> str:
        """Return the highest-priority classification key for ``facts``."""
        cfg = self._cfg
        if facts.battery_v < cfg.low_battery_v:
            return _KEY_LOW_BATTERY
        # ``min_clearance_m`` is populated even from the forward ultrasonic, so a
        # "too close" claim is honest without the 360 view.
        if facts.min_clearance_m < cfg.tight_space_m:
            return _KEY_TIGHT_SPACE
        if facts.audio_valid and facts.audio_rms > cfg.loud_rms:
            return _KEY_LOUD
        if facts.speed_mps > cfg.fast_mps:
            return _KEY_MOVING_FAST
        # "lots of open room" is a 360 claim — only when LiDAR was actually valid.
        if facts.lidar_valid and facts.min_clearance_m > cfg.open_space_m:
            return _KEY_OPEN_SPACE
        return _KEY_DEFAULT

    async def compose(self, facts: CommentaryFacts) -> str:
        """Return the plain template string for the classified situation."""
        key = self._classify(facts)
        # ``default`` is guaranteed present by the config validator; an absent
        # optional key falls back to ``default`` rather than raising.
        return self._cfg.templates.get(key, self._cfg.templates[_KEY_DEFAULT])


class LLMCommentaryComposer:
    """LLM composer: grounded facts -> ``answer_query`` -> a short narration.

    Builds a compact, label-free fact string, fills ``cfg.llm_prompt_template``,
    and asks the gateway. Returns ``""`` on any degrade / rejection so the engine
    simply doesn't speak (never raises into the loop). The gateway's
    prompt-injection filter is applied inside ``answer_query``.
    """

    def __init__(self, gateway: QueryCapableLLMProtocol, cfg: CommentaryConfig) -> None:
        self._gateway = gateway
        self._cfg = cfg

    def _facts_string(self, facts: CommentaryFacts) -> str:
        """Compact, label-free grounded fact string for the prompt."""
        parts: list[str] = []
        if facts.lidar_valid:
            parts.append(f"nearest obstacle {facts.min_clearance_m:.2f} m")
        else:
            parts.append(f"forward distance {facts.forward_distance_m:.2f} m")
        if facts.audio_valid:
            parts.append(f"sound level {facts.audio_rms:.2f}")
        parts.append(f"speed {facts.speed_mps:.2f} m/s")
        parts.append(f"battery {facts.battery_v:.1f} V")
        if facts.novelty is not None:
            parts.append(f"novelty {facts.novelty:.3f}")
        return ", ".join(parts)

    def _truncate(self, text: str) -> str:
        """Clamp ``text`` to ``cfg.max_words`` words."""
        words = text.split()
        if len(words) <= self._cfg.max_words:
            return text.strip()
        return " ".join(words[: self._cfg.max_words])

    async def compose(self, facts: CommentaryFacts) -> str:
        """Ask the gateway for a short narration; ``""`` on any failure."""
        prompt = self._cfg.llm_prompt_template.format(facts=self._facts_string(facts))
        try:
            answer = await self._gateway.answer_query(prompt)
        except InjectionRejected:
            # The grounded prompt should never trip the filter; if it does,
            # treat as "nothing to say" rather than surfacing a caller error.
            _log.warning("commentary_llm_injection_rejected")
            return ""
        except ValueError:
            _log.warning("commentary_llm_value_error")
            return ""
        return self._truncate(answer) if answer else ""


__all__ = ["LLMCommentaryComposer", "TemplateCommentaryComposer"]
