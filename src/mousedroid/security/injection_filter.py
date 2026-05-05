"""Prompt-injection filter shared by the LLM gateway and OpenClaw channel.

Lifts the regex-based sanitisation that previously lived inside
:class:`mousedroid.llm_gateway.gateway.LLMGateway` so the same envelope
applies to every NL ingress (LLM gateway, REST mission endpoint, future
channel adapters). Patterns are config-driven; an empty pattern list
disables the regex check while leaving length truncation in place.

The filter never logs the candidate text — only structured fields about
*why* a rejection happened. This keeps user-supplied mission text out of
the log buffer (defence-in-depth against accidental token leakage).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)


class InjectionRejected(ValueError):  # noqa: N818 - read as "rejection event", not a class of errors
    """Raised by :meth:`PromptInjectionFilterProtocol.sanitize` on a hit.

    Inherits from :class:`ValueError` so existing call sites that catch
    ``ValueError`` (notably :meth:`LLMGateway.translate_mission`) keep
    working unchanged.
    """


@runtime_checkable
class PromptInjectionFilterProtocol(Protocol):
    """Synchronous filter that sanitises an NL command or rejects it."""

    def sanitize(self, text: str) -> str:
        """Return the sanitised command, or raise :class:`InjectionRejected`.

        The returned string is the trimmed and length-truncated form of
        ``text``; callers should treat it as the canonical command body.
        """
        ...


class RegexInjectionFilter:
    """Compile-once regex filter with a configurable pattern list.

    An invalid user-supplied regex must not crash the host process — the
    constructor logs a structured warning and falls back to the
    "patterns-disabled" state. Length truncation still applies so the
    filter is never byte-identical to a no-op.
    """

    def __init__(self, patterns: Iterable[str], *, max_len: int) -> None:
        """Initialise the filter.

        Args:
            patterns: Iterable of regex source strings combined into a
                single alternation. Empty iterable disables the regex
                check (length truncation still applies).
            max_len: Maximum command length in characters. Inputs longer
                than this are truncated *before* the regex runs so an
                attacker cannot bypass detection by appending a long
                payload after a sentinel.
        """
        if max_len <= 0:
            msg = "max_len must be positive"
            raise ValueError(msg)
        self._max_len = max_len
        pattern_list = list(patterns)
        if not pattern_list:
            self._regex: re.Pattern[str] | None = None
            return
        try:
            self._regex = re.compile("(" + "|".join(pattern_list) + ")", re.IGNORECASE)
        except re.error as exc:
            _log.warning(
                "injection_filter_invalid_pattern",
                error=str(exc),
                pattern_count=len(pattern_list),
            )
            self._regex = None

    @property
    def max_len(self) -> int:
        """The configured maximum-length cap."""
        return self._max_len

    @property
    def has_regex(self) -> bool:
        """Whether a compiled regex is active (False = length-only mode)."""
        return self._regex is not None

    def sanitize(self, text: str) -> str:
        """Trim, truncate, and regex-check ``text``.

        Args:
            text: Candidate command. ``None`` is treated as an empty string
                by the caller; this method itself only operates on ``str``.

        Returns:
            The trimmed and length-truncated command body.

        Raises:
            InjectionRejected: When the regex matches the truncated body.
        """
        trimmed = text.strip()[: self._max_len]
        if self._regex is not None and self._regex.search(trimmed):
            _log.info(
                "injection_filter_rejected",
                length=len(trimmed),
                has_regex=True,
            )
            msg = "command contains disallowed content"
            raise InjectionRejected(msg)
        return trimmed


__all__ = [
    "InjectionRejected",
    "PromptInjectionFilterProtocol",
    "RegexInjectionFilter",
]
