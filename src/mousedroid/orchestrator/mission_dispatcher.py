"""Single ingress for every NL mission channel (REST, MCP, future).

The dispatcher wraps :meth:`MouseDroidOrchestrator.process_mission` with
the cross-cutting concerns the OpenClaw integration depends on:

* Channel allow-list enforced from :class:`OpenClawConfig.allowed_channels`.
* Length cap and prompt-injection rejection via the shared
  :class:`PromptInjectionFilterProtocol` so REST + MCP behave identically.
* A stable ``trace_id`` bound into ``structlog.contextvars`` for the
  duration of the dispatch and returned to the caller for correlation
  across telemetry, logs, and the MEMORY.md exporter.
* Structured logs that NEVER carry the raw ``nl_command`` — only
  ``command_hash`` (sha256 prefix) and observable structured fields.

The dispatcher is also the single hook the orchestrator's POST_TICK
exporter consults via :attr:`mission_just_completed` to decide whether a
new MEMORY.md snapshot should be written.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from structlog.contextvars import bind_contextvars, unbind_contextvars

from mousedroid.llm_gateway.protocol import GoalVector
from mousedroid.logging.setup import get_logger
from mousedroid.security.injection_filter import (
    InjectionRejected,
    PromptInjectionFilterProtocol,
)

if TYPE_CHECKING:
    from mousedroid.config.schema import OpenClawConfig

_log = get_logger(__name__)

# Length of the sha256 hex prefix logged in lieu of the raw command.
# Long enough to disambiguate replays during a session, short enough that
# log lines stay readable.
_COMMAND_HASH_PREFIX = 12

# Length of the uuid4 hex prefix used as the per-dispatch trace id. 16
# hex chars = 64 bits of entropy, plenty to disambiguate dispatches
# within a session while staying short enough to grep across telemetry.
_TRACE_ID_PREFIX = 16


@runtime_checkable
class MissionDispatcherProtocol(Protocol):
    """Single ingress contract for every NL mission channel."""

    @property
    def mission_just_completed(self) -> bool:
        """One-shot flag set after a successful dispatch.

        The Phase D MEMORY.md exporter consults this in the orchestrator's
        POST_TICK hook so a snapshot is only written when there is fresh
        episodic data to capture. The hook is responsible for clearing
        the flag via :meth:`clear_mission_completed`.
        """
        ...

    def clear_mission_completed(self) -> None:
        """Clear the one-shot completion flag (called by the export hook)."""
        ...

    async def dispatch(
        self,
        nl_command: str,
        *,
        channel: str,
        peer: str,
    ) -> DispatchResult:
        """Dispatch a sanitised command to the orchestrator.

        Args:
            nl_command: Raw NL command from the channel adapter.
            channel: One of ``OpenClawConfig.allowed_channels`` (e.g.
                ``"rest"``, ``"mcp"``).
            peer: Best-effort caller identifier (IP, MCP session id, ...).

        Returns:
            A :class:`DispatchResult` carrying the goal vector, the
            generated trace id, and a timing summary.

        Raises:
            ValueError: When the channel is not allowed or the command
                fails the injection filter.
        """
        ...


class DispatchResult:
    """Plain result object returned by :meth:`MissionDispatcherProtocol.dispatch`.

    A class (not a dataclass) so we can keep ``__slots__`` and avoid
    surprise mutability without dragging in attrs.
    """

    __slots__ = ("channel", "command_hash", "goal_vector", "latency_ms", "peer", "trace_id")

    def __init__(
        self,
        *,
        goal_vector: GoalVector,
        trace_id: str,
        command_hash: str,
        channel: str,
        peer: str,
        latency_ms: float,
    ) -> None:
        self.goal_vector = goal_vector
        self.trace_id = trace_id
        self.command_hash = command_hash
        self.channel = channel
        self.peer = peer
        self.latency_ms = latency_ms


@runtime_checkable
class _OrchestratorProcessMissionLike(Protocol):
    """Structural protocol for the slice of orchestrator we depend on.

    Avoids a hard import of :class:`MouseDroidOrchestrator` so the
    dispatcher remains testable with a tiny stand-in (used in unit tests
    and the e2e smoke).
    """

    async def process_mission(self, nl_command: str) -> GoalVector: ...


class DeferredOrchestratorRef:
    """Forward-reference shim for the build-order cycle.

    The dispatcher must be constructed before the telemetry server
    (which now takes it as a constructor arg), but the orchestrator —
    which the dispatcher ultimately calls — is constructed last because
    it owns every other component. This shim implements the structural
    protocol the dispatcher expects and is :meth:`bind`-ed to the real
    orchestrator after construction.

    Calling :meth:`process_mission` before :meth:`bind` raises
    :class:`RuntimeError`; this is a programming error and never reached
    in production because :func:`mousedroid.factory.build_orchestrator`
    binds before returning.
    """

    __slots__ = ("_target",)

    def __init__(self) -> None:
        self._target: _OrchestratorProcessMissionLike | None = None

    def bind(self, target: _OrchestratorProcessMissionLike) -> None:
        """Attach the real orchestrator (call exactly once at the end of wiring)."""
        self._target = target

    async def process_mission(self, nl_command: str) -> GoalVector:
        """Forward to the bound orchestrator; raise if not yet bound."""
        if self._target is None:
            msg = (
                "DeferredOrchestratorRef.process_mission called before bind(); "
                "factory wiring did not complete."
            )
            raise RuntimeError(msg)
        return await self._target.process_mission(nl_command)


class OrchestratorMissionDispatcher:
    """Default :class:`MissionDispatcherProtocol` implementation."""

    def __init__(
        self,
        orchestrator: _OrchestratorProcessMissionLike,
        *,
        injection_filter: PromptInjectionFilterProtocol,
        cfg: OpenClawConfig,
    ) -> None:
        """Wire the dispatcher.

        Args:
            orchestrator: Object exposing
                ``async process_mission(nl_command) -> GoalVector``.
            injection_filter: Shared prompt-injection filter.
            cfg: OpenClaw config (allowed_channels, max_command_len).
        """
        self._orchestrator = orchestrator
        self._filter = injection_filter
        self._cfg = cfg
        self._allowed = frozenset(cfg.allowed_channels)
        self._mission_completed = False

    @property
    def mission_just_completed(self) -> bool:
        """One-shot flag — see protocol docstring."""
        return self._mission_completed

    def clear_mission_completed(self) -> None:
        """Clear the one-shot flag after the export hook runs."""
        self._mission_completed = False

    async def dispatch(
        self,
        nl_command: str,
        *,
        channel: str,
        peer: str,
    ) -> DispatchResult:
        """Dispatch path for every NL channel — see protocol docstring."""
        trace_id = uuid.uuid4().hex[:_TRACE_ID_PREFIX]
        bind_contextvars(trace_id=trace_id, channel=channel)
        try:
            if channel not in self._allowed:
                _log.warning(
                    "mission_dispatched_rejected",
                    reason="channel_not_allowed",
                    peer=peer,
                    allowed=sorted(self._allowed),
                )
                msg = f"channel {channel!r} not in allowed_channels"
                raise ValueError(msg)
            if not nl_command or not nl_command.strip():
                _log.warning(
                    "mission_dispatched_rejected",
                    reason="empty_command",
                    peer=peer,
                )
                msg = "nl_command must be non-empty"
                raise ValueError(msg)
            if len(nl_command) > self._cfg.max_command_len:
                _log.warning(
                    "mission_dispatched_rejected",
                    reason="command_too_long",
                    peer=peer,
                    length=len(nl_command),
                    limit=self._cfg.max_command_len,
                )
                msg = f"nl_command exceeds max_command_len ({self._cfg.max_command_len})"
                raise ValueError(msg)
            try:
                sanitised = self._filter.sanitize(nl_command)
            except InjectionRejected:
                _log.warning(
                    "mission_dispatched_rejected",
                    reason="injection_pattern",
                    peer=peer,
                )
                raise

            command_hash = hashlib.sha256(sanitised.encode("utf-8")).hexdigest()[
                :_COMMAND_HASH_PREFIX
            ]
            start = time.monotonic()
            goal = await self._orchestrator.process_mission(sanitised)
            latency_ms = (time.monotonic() - start) * 1000.0

            self._mission_completed = True
            _log.info(
                "mission_dispatched",
                peer=peer,
                command_hash=command_hash,
                latency_ms=latency_ms,
                vx=goal.vx_target,
                vy=goal.vy_target,
                omega=goal.omega_target,
            )
            return DispatchResult(
                goal_vector=goal,
                trace_id=trace_id,
                command_hash=command_hash,
                channel=channel,
                peer=peer,
                latency_ms=latency_ms,
            )
        finally:
            unbind_contextvars("trace_id", "channel")


def build_mission_dispatcher(
    cfg: OpenClawConfig | None,
    *,
    injection_filter: PromptInjectionFilterProtocol,
) -> tuple[MissionDispatcherProtocol | None, DeferredOrchestratorRef | None]:
    """Build the dispatcher and its forward-reference shim.

    Returns ``(None, None)`` when OpenClaw is disabled so existing
    deployments incur zero overhead. Otherwise returns the dispatcher
    paired with the :class:`DeferredOrchestratorRef` the factory must
    later :meth:`DeferredOrchestratorRef.bind` to the real orchestrator.
    """
    if cfg is None or not cfg.enabled:
        _log.debug(
            "mission_dispatcher_disabled",
            reason="openclaw_none" if cfg is None else "openclaw_disabled",
        )
        return None, None
    deferred = DeferredOrchestratorRef()
    dispatcher = OrchestratorMissionDispatcher(
        deferred,
        injection_filter=injection_filter,
        cfg=cfg,
    )
    _log.info(
        "mission_dispatcher_built",
        allowed_channels=sorted(cfg.allowed_channels),
        max_command_len=cfg.max_command_len,
    )
    return dispatcher, deferred


__all__ = [
    "DeferredOrchestratorRef",
    "DispatchResult",
    "MissionDispatcherProtocol",
    "OrchestratorMissionDispatcher",
    "build_mission_dispatcher",
]
