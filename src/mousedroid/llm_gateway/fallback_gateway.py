"""Tier C-rover: cloud-primary / local-secondary failover gateway.

A composite :class:`LLMGatewayProtocol` that routes mission translation to a
``primary`` gateway (typically the cloud :class:`AnthropicLLMGateway`) and
transparently falls back to a ``secondary`` LOCAL gateway (llama-cpp GGUF, or
an OpenAI-compatible endpoint pointed at a local Ollama) when the primary is
unavailable or degraded — e.g. the Jetson is off-network and cannot reach the
Claude API. This keeps the rover autonomous without connectivity: cloud Claude
for quality when connected, a local model when not.

Failover semantics (the key design point — see PR peer review):

* The shipped gateways **never raise** on a backend failure; they return a
  neutral :class:`GoalVector` and flip ``is_degraded``. So failover is keyed
  off the ``is_degraded`` flag, NOT off the returned vector (a legitimate
  ``"stop"`` command also returns a neutral vector — that must not trigger
  spurious failover).
* The primary is used only while it is *usable* (``is_ready`` and not
  ``is_degraded``). If a call degrades the primary mid-flight, this command
  and all subsequent ones route to the secondary (we stop hammering a dead
  endpoint until the next :meth:`start`).
* A *command* rejection (:class:`ValueError` / ``InjectionRejected``) is a
  caller error, not a backend failure — both backends would reject the same
  input identically, so it propagates rather than triggering failover.

``is_degraded`` is read via :func:`getattr` so the composite works with any
:class:`LLMGatewayProtocol` implementation, even one that does not expose the
(non-protocol) flag — such a gateway is simply treated as never-degraded.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from mousedroid.llm_gateway.protocol import GoalVector
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from mousedroid.llm_gateway.protocol import LLMGatewayProtocol

_log = get_logger(__name__)

# Default seconds between primary re-probe attempts once it is degraded.
# Used only when the factory does not supply ``cfg.llm.fallback_retry_cooldown_s``
# (e.g. a direct construction in a test). Production always passes the
# config value, so this is a sane fallback, not an operator-tunable knob
# living in code.
_DEFAULT_RETRY_COOLDOWN_S = 30.0


def _is_degraded(gateway: LLMGatewayProtocol) -> bool:
    """Return the gateway's ``is_degraded`` flag, defaulting to ``False``.

    ``is_degraded`` is not part of :class:`LLMGatewayProtocol`; all shipped
    concrete gateways expose it. A gateway without the attribute is treated
    as never-degraded.
    """
    return bool(getattr(gateway, "is_degraded", False))


class FallbackLLMGateway:
    """Primary + secondary composite with degrade-triggered failover.

    Conforms structurally to :class:`LLMGatewayProtocol`. Constructed by
    :func:`mousedroid.factory.build_llm_gateway` when
    ``cfg.llm.fallback_backend != "none"``.
    """

    def __init__(
        self,
        primary: LLMGatewayProtocol,
        secondary: LLMGatewayProtocol,
        *,
        retry_cooldown_s: float = _DEFAULT_RETRY_COOLDOWN_S,
        clock: Callable[[], float] | None = None,
    ) -> None:
        """Initialise the composite.

        Args:
            primary: Preferred gateway (e.g. cloud Claude).
            secondary: Local fallback gateway used when the primary is
                unavailable / degraded.
            retry_cooldown_s: Seconds to wait before re-probing a degraded
                primary. A mobile rover sees transient WAN dropouts, so the
                composite periodically re-attempts the cloud primary rather
                than bypassing it forever once degraded. The factory wires
                ``cfg.llm.fallback_retry_cooldown_s`` here.
            clock: Monotonic time source (seconds). Defaults to
                :func:`time.monotonic`; injectable so tests can advance the
                cooldown deterministically.
        """
        self._primary = primary
        self._secondary = secondary
        self._retry_cooldown_s = retry_cooldown_s
        self._clock: Callable[[], float] = clock if clock is not None else time.monotonic
        # ``-inf`` guarantees the first call always attempts the primary,
        # regardless of the cooldown window.
        self._last_primary_attempt = float("-inf")

    @property
    def is_ready(self) -> bool:
        """True iff either child gateway is ready to serve translations."""
        return self._primary.is_ready or self._secondary.is_ready

    @property
    def is_degraded(self) -> bool:
        """True iff BOTH children are degraded (no usable path remains)."""
        return _is_degraded(self._primary) and _is_degraded(self._secondary)

    async def start(self) -> None:
        """Start both children so the secondary is warm before it's needed."""
        await self._primary.start()
        await self._secondary.start()
        _log.info(
            "fallback_gateway_started",
            primary_ready=self._primary.is_ready,
            primary_degraded=_is_degraded(self._primary),
            secondary_ready=self._secondary.is_ready,
            secondary_degraded=_is_degraded(self._secondary),
        )

    async def translate_mission(self, nl_command: str) -> GoalVector:
        """Route to the primary, falling back to the secondary on failure.

        Args:
            nl_command: Natural language mission.

        Returns:
            ``GoalVector`` from whichever tier served the command (neutral if
            both are unavailable).

        Raises:
            ValueError: Propagated from a child when the command is empty or
                rejected by the injection filter (caller error — no failover).
        """
        # Use the primary when it's ready AND either healthy or its cooldown
        # has elapsed since the last attempt. The cooldown re-probe lets the
        # rover recover the cloud connection after a transient WAN dropout
        # instead of being pinned to the local secondary until the next
        # start(). A successful primary call clears its own ``is_degraded``
        # (see AnthropicLLMGateway), so a recovered primary resumes serving.
        now = self._clock()
        primary_degraded = _is_degraded(self._primary)
        cooldown_elapsed = (now - self._last_primary_attempt) >= self._retry_cooldown_s
        use_primary = self._primary.is_ready and (not primary_degraded or cooldown_elapsed)
        if use_primary:
            self._last_primary_attempt = now
            goal: GoalVector | None
            try:
                goal = await self._primary.translate_mission(nl_command)
            except ValueError:
                # Empty / injection-rejected command — the secondary would
                # reject it identically. Propagate; do not failover.
                raise
            except Exception as exc:
                _log.warning(
                    "fallback_primary_exception",
                    error=f"{type(exc).__name__}:{exc}",
                )
                goal = None
            # A primary that stayed healthy served this command (even a
            # legitimately-neutral GoalVector). Only failover when the call
            # degraded the primary or raised unexpectedly.
            if goal is not None and not _is_degraded(self._primary):
                _log.debug("fallback_served", served_by="primary", retry_probe=primary_degraded)
                return goal
            _log.warning("fallback_primary_to_secondary", was_retry_probe=primary_degraded)

        goal = await self._secondary.translate_mission(nl_command)
        _log.debug(
            "fallback_served",
            served_by="secondary",
            secondary_ready=self._secondary.is_ready,
        )
        return goal

    async def stop(self) -> None:
        """Stop both children."""
        await self._primary.stop()
        await self._secondary.stop()
        _log.info("fallback_gateway_stopped")


__all__ = ["FallbackLLMGateway"]
