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

import asyncio
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
        """Start both children CONCURRENTLY so the secondary is warm sooner.

        Both child ``start()`` calls are I/O-bound (cloud TLS handshake +
        local GGUF model mmap) and fully independent. Sequential startup
        blocks the rover for ``T_primary + T_secondary``;
        ``asyncio.gather`` runs them in parallel so the rover is ready in
        ``max(T_primary, T_secondary)`` instead (code-reviewer PR #107
        finding 5). ``return_exceptions=True`` ensures a primary-start
        failure does not abort the secondary's startup — the composite's
        whole reason for existing is to gracefully degrade.
        """
        results = await asyncio.gather(
            self._primary.start(),
            self._secondary.start(),
            return_exceptions=True,
        )
        # Surface any start-time exception in the structured log without
        # raising — both children already mark themselves degraded on
        # init failure, and the composite's ``is_ready`` / ``is_degraded``
        # properties already reflect that. Re-raising would defeat the
        # gateway's own degrade-don't-crash contract.
        for tier_name, outcome in zip(("primary", "secondary"), results, strict=True):
            if isinstance(outcome, BaseException):
                _log.warning(
                    "fallback_gateway_child_start_failed",
                    tier=tier_name,
                    error=f"{type(outcome).__name__}:{outcome}",
                )
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
            goal: GoalVector | None
            try:
                goal = await self._primary.translate_mission(nl_command)
            except asyncio.CancelledError:
                # Cooperative cancellation (orchestrator loop teardown, e-stop
                # tearing down the pending mission task). Propagate without
                # touching ``_last_primary_attempt`` or ``_degraded`` so the
                # cancelled probe does not poison the cooldown timer or
                # falsely mark the primary as degraded. The composite
                # contract reserves ``never raises on backend failure`` for
                # *backend* failures, not for caller-initiated cancellation
                # (code-reviewer PR #107 round-3 finding 1).
                raise
            except ValueError:
                # Empty / injection-rejected command — the secondary would
                # reject it identically. Propagate; do not failover.
                # Stamp the attempt timestamp because the primary did "see"
                # the request — operator can still see the timestamp move
                # in observability dashboards.
                self._last_primary_attempt = now
                raise
            except Exception as exc:
                _log.warning(
                    "fallback_primary_exception",
                    error=f"{type(exc).__name__}:{exc}",
                )
                goal = None
            # Stamp the attempt timestamp AFTER the await returns so a
            # cancelled probe does not advance the cooldown window
            # (would falsely keep the secondary serving even when the
            # primary recovered — code-reviewer PR #107 round-3 finding 1).
            self._last_primary_attempt = now
            # A primary that stayed healthy served this command (even a
            # legitimately-neutral GoalVector). Only failover when the call
            # degraded the primary or raised unexpectedly.
            if goal is not None and not _is_degraded(self._primary):
                _log.debug("fallback_served", served_by="primary", retry_probe=primary_degraded)
                return goal
            _log.warning("fallback_primary_to_secondary", was_retry_probe=primary_degraded)

        try:
            goal = await self._secondary.translate_mission(nl_command)
        except asyncio.CancelledError:
            # Same cooperative-cancellation handling as the primary branch.
            raise
        except ValueError:
            # Caller error — propagate. Symmetric with the primary branch:
            # neither backend gets a second chance at a malformed command.
            raise
        except Exception as exc:
            # Preserve the composite-level "never raises on backend failure"
            # invariant: a malloc failure / corrupted-weight crash in the
            # local secondary becomes a neutral GoalVector + structured
            # warning, not a crash propagating into the orchestrator's
            # mission handler (code-reviewer PR #107 finding 2).
            _log.warning(
                "fallback_secondary_exception",
                error=f"{type(exc).__name__}:{exc}",
            )
            return GoalVector()
        _log.debug(
            "fallback_served",
            served_by="secondary",
            secondary_ready=self._secondary.is_ready,
        )
        return goal

    async def stop(self) -> None:
        """Stop both children — never let a primary failure skip the secondary.

        ``asyncio.gather(return_exceptions=True)`` so a raise in one
        child's ``stop`` does not skip the other's cleanup. A leaked
        GGUF mmap or HTTP session in the secondary would be a real
        resource leak on the long-running Jetson process (code-explorer
        PR #107 finding).
        """
        results = await asyncio.gather(
            self._primary.stop(),
            self._secondary.stop(),
            return_exceptions=True,
        )
        for tier_name, outcome in zip(("primary", "secondary"), results, strict=True):
            if isinstance(outcome, BaseException):
                _log.warning(
                    "fallback_gateway_child_stop_failed",
                    tier=tier_name,
                    error=f"{type(outcome).__name__}:{outcome}",
                )
        _log.info("fallback_gateway_stopped")


__all__ = ["FallbackLLMGateway"]
