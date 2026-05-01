"""Phase-keyed registry of orchestrator tick hooks.

The registry stores hooks per :class:`HookPhase` in registration order
and runs them sequentially when the orchestrator invokes ``run_phase``.
The hot-loop fast path (no hooks registered for a phase) is a single
dict lookup followed by an early return.

All error handling is config-driven via the per-hook ``error_policy``.
A no-op default registry is provided so the orchestrator's existing
behaviour is unchanged when ``Settings.harness is None``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mousedroid.harness.protocol import (
    HookPhase,
    HookRegistryProtocol,
    HookSpec,
    TickContext,
)
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    pass

_log = get_logger(__name__)

_VALID_ERROR_POLICIES: frozenset[str] = frozenset({"raise", "warn", "swallow"})


class HookRegistryError(RuntimeError):
    """Raised when a hook spec cannot be accepted by the registry."""


class HookRegistry:
    """Concrete :class:`HookRegistryProtocol` implementation.

    Hooks are stored in per-phase lists, preserving registration order so
    behaviour is deterministic. Re-registering a name within the same
    phase replaces the prior spec (logged at WARNING). Re-registering the
    same name across different phases is allowed; the registry simply
    tracks both bindings.
    """

    def __init__(self) -> None:
        self._by_phase: dict[HookPhase, list[HookSpec]] = {phase: [] for phase in HookPhase}

    # --------------------------------------------------------------- API
    def register(self, spec: HookSpec) -> None:
        if spec.error_policy not in _VALID_ERROR_POLICIES:
            msg = (
                f"Invalid error_policy {spec.error_policy!r}; "
                f"expected one of {sorted(_VALID_ERROR_POLICIES)}"
            )
            raise HookRegistryError(msg)
        bucket = self._by_phase[spec.phase]
        for idx, existing in enumerate(bucket):
            if existing.name == spec.name:
                bucket[idx] = spec
                _log.warning(
                    "hook_replaced",
                    name=spec.name,
                    phase=spec.phase.value,
                )
                return
        bucket.append(spec)
        _log.debug("hook_registered", name=spec.name, phase=spec.phase.value)

    def unregister(self, name: str) -> bool:
        removed = False
        for phase, bucket in self._by_phase.items():
            for idx, existing in enumerate(bucket):
                if existing.name == name:
                    bucket.pop(idx)
                    removed = True
                    _log.debug("hook_unregistered", name=name, phase=phase.value)
                    break
        return removed

    def for_phase(self, phase: HookPhase) -> tuple[HookSpec, ...]:
        return tuple(self._by_phase[phase])

    async def run_phase(self, phase: HookPhase, ctx: TickContext) -> None:
        bucket = self._by_phase[phase]
        if not bucket:
            return  # hot-loop fast path
        for spec in bucket:
            try:
                await spec.handler(ctx)
            except Exception as exc:  # pylint: disable=broad-except
                if spec.error_policy == "raise":
                    _log.error(
                        "hook_failed_raise",
                        name=spec.name,
                        phase=phase.value,
                        error=str(exc),
                        exc_info=True,
                    )
                    raise
                if spec.error_policy == "warn":
                    _log.warning(
                        "hook_failed",
                        name=spec.name,
                        phase=phase.value,
                        error=str(exc),
                        exc_info=True,
                    )
                else:  # swallow
                    _log.debug(
                        "hook_failed_swallow",
                        name=spec.name,
                        phase=phase.value,
                        error=str(exc),
                    )


class NullHookRegistry:
    """No-op :class:`HookRegistryProtocol` for the disabled harness path.

    Every call is a constant-time no-op. The orchestrator wires this
    instance when ``Settings.harness is None`` so the 30 Hz tick loop
    never pays for hook dispatch.
    """

    def register(self, spec: HookSpec) -> None:  # pragma: no cover - trivial
        return None

    def unregister(self, name: str) -> bool:  # pragma: no cover - trivial
        return False

    def for_phase(self, phase: HookPhase) -> tuple[HookSpec, ...]:  # pragma: no cover
        return ()

    async def run_phase(self, phase: HookPhase, ctx: TickContext) -> None:
        return None


# Static check that both classes implement the protocol surface.
_PROTOCOL_CHECK: HookRegistryProtocol = HookRegistry()
_PROTOCOL_CHECK_NULL: HookRegistryProtocol = NullHookRegistry()
del _PROTOCOL_CHECK
del _PROTOCOL_CHECK_NULL


__all__ = ["HookRegistry", "HookRegistryError", "NullHookRegistry"]
