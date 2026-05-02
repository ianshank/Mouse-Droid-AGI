"""Tests for ``mousedroid.harness.hooks``."""

from __future__ import annotations

import pytest

from mousedroid.harness.hooks import (
    HookRegistry,
    HookRegistryError,
    NullHookRegistry,
)
from mousedroid.harness.protocol import (
    HookPhase,
    HookRegistryProtocol,
    HookSpec,
    TickContext,
)


def _ctx(tick: int = 0) -> TickContext:
    return TickContext(tick_index=tick, timestamp_s=float(tick))


@pytest.fixture
def registry() -> HookRegistry:
    return HookRegistry()


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_hook_registry_implements_protocol(registry: HookRegistry) -> None:
    assert isinstance(registry, HookRegistryProtocol)


def test_null_registry_implements_protocol() -> None:
    assert isinstance(NullHookRegistry(), HookRegistryProtocol)


# ---------------------------------------------------------------------------
# register / unregister / for_phase
# ---------------------------------------------------------------------------


def test_register_and_for_phase(registry: HookRegistry) -> None:
    async def h(ctx: TickContext) -> None:
        pass

    spec = HookSpec(name="x", phase=HookPhase.PRE_TICK, handler=h)
    registry.register(spec)
    assert registry.for_phase(HookPhase.PRE_TICK) == (spec,)
    assert registry.for_phase(HookPhase.POST_TICK) == ()


def test_register_invalid_error_policy_raises(registry: HookRegistry) -> None:
    async def h(ctx: TickContext) -> None:
        pass

    with pytest.raises(HookRegistryError):
        registry.register(
            HookSpec(name="x", phase=HookPhase.PRE_TICK, handler=h, error_policy="boom")
        )


def test_register_replaces_same_name_within_phase(registry: HookRegistry) -> None:
    async def h1(ctx: TickContext) -> None:
        pass

    async def h2(ctx: TickContext) -> None:
        pass

    registry.register(HookSpec(name="x", phase=HookPhase.PRE_TICK, handler=h1))
    registry.register(HookSpec(name="x", phase=HookPhase.PRE_TICK, handler=h2))
    bucket = registry.for_phase(HookPhase.PRE_TICK)
    assert len(bucket) == 1
    assert bucket[0].handler is h2


def test_unregister_removes_from_all_phases(registry: HookRegistry) -> None:
    async def h(ctx: TickContext) -> None:
        pass

    registry.register(HookSpec(name="x", phase=HookPhase.PRE_TICK, handler=h))
    registry.register(HookSpec(name="x", phase=HookPhase.POST_TICK, handler=h))
    # ``unregister`` strips the name from every phase in a single call.
    assert registry.unregister("x") is True
    assert registry.for_phase(HookPhase.PRE_TICK) == ()
    assert registry.for_phase(HookPhase.POST_TICK) == ()
    # Subsequent call is a no-op.
    assert registry.unregister("x") is False


def test_unregister_unknown_returns_false(registry: HookRegistry) -> None:
    assert registry.unregister("missing") is False


# ---------------------------------------------------------------------------
# run_phase semantics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_phase_is_noop_fast_path(registry: HookRegistry) -> None:
    # No hooks registered — must complete immediately and not raise.
    await registry.run_phase(HookPhase.PRE_TICK, _ctx())


@pytest.mark.asyncio
async def test_hooks_run_in_registration_order(registry: HookRegistry) -> None:
    calls: list[str] = []

    async def first(ctx: TickContext) -> None:
        calls.append("first")

    async def second(ctx: TickContext) -> None:
        calls.append("second")

    registry.register(HookSpec(name="a", phase=HookPhase.PRE_TICK, handler=first))
    registry.register(HookSpec(name="b", phase=HookPhase.PRE_TICK, handler=second))
    await registry.run_phase(HookPhase.PRE_TICK, _ctx())
    assert calls == ["first", "second"]


@pytest.mark.asyncio
async def test_warn_policy_isolates_failures(registry: HookRegistry) -> None:
    calls: list[str] = []

    async def boom(ctx: TickContext) -> None:
        raise RuntimeError("kapow")

    async def after(ctx: TickContext) -> None:
        calls.append("after")

    registry.register(
        HookSpec(name="boom", phase=HookPhase.PRE_TICK, handler=boom, error_policy="warn")
    )
    registry.register(HookSpec(name="after", phase=HookPhase.PRE_TICK, handler=after))
    await registry.run_phase(HookPhase.PRE_TICK, _ctx())
    assert calls == ["after"]


@pytest.mark.asyncio
async def test_swallow_policy_silences_errors(registry: HookRegistry) -> None:
    async def boom(ctx: TickContext) -> None:
        raise RuntimeError("kapow")

    registry.register(
        HookSpec(name="boom", phase=HookPhase.PRE_TICK, handler=boom, error_policy="swallow")
    )
    await registry.run_phase(HookPhase.PRE_TICK, _ctx())  # must not raise


@pytest.mark.asyncio
async def test_raise_policy_propagates(registry: HookRegistry) -> None:
    async def boom(ctx: TickContext) -> None:
        raise RuntimeError("kapow")

    registry.register(
        HookSpec(name="boom", phase=HookPhase.PRE_TICK, handler=boom, error_policy="raise")
    )
    with pytest.raises(RuntimeError, match="kapow"):
        await registry.run_phase(HookPhase.PRE_TICK, _ctx())


@pytest.mark.asyncio
async def test_null_registry_run_phase_is_noop() -> None:
    null = NullHookRegistry()
    await null.run_phase(HookPhase.PRE_TICK, _ctx())  # must not raise
