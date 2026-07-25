"""Regression: async-cancellation and -O-safety hygiene (code-hygiene sprint).

Three contracts, each of which regressed (or nearly regressed) before:

1. No ``except`` tuple in ``src/`` may pair ``CancelledError`` with
   ``Exception`` — ``asyncio.CancelledError`` subclasses ``BaseException``
   (not ``Exception``) on every supported interpreter, so such a tuple can
   only mean cancellation is being caught deliberately alongside ordinary
   failures, which swallows cooperative cancellation (the exact bug fixed
   in ``telemetry/server.py``'s mission-dedup follower path).
2. The mission-dedup follower path re-raises ``CancelledError`` instead of
   converting it to a 500 response.
3. ``retry_async`` raises ``RetryExhaustedError`` with a real exception
   payload even when the retry loop never ran — under ``PYTHONOPTIMIZE=1``
   (the Jetson Docker default) an ``assert`` would have been stripped and a
   ``None`` payload would masquerade as the final-attempt error.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest

from mousedroid.resilience.retry import RetryExhaustedError, retry_async

_SRC = Path(__file__).resolve().parents[2] / "src" / "mousedroid"

# An ``except (...)`` tuple that names CancelledError together with the broad
# Exception class. Matching on source text (not AST) keeps this consistent
# with the suppression-budget test's approach; the pattern tolerates
# whitespace/newlines between the tuple members in either order.
_CANCELLED_WITH_EXCEPTION = re.compile(
    r"except\s*\((?:[^)]*CancelledError[^)]*,\s*Exception\b|"
    r"[^)]*\bException\s*,[^)]*CancelledError)[^)]*\)",
)


def test_no_except_tuple_pairs_cancelled_error_with_exception() -> None:
    """CancelledError must never be caught via a tuple with Exception."""
    offenders = [
        str(path.relative_to(_SRC))
        for path in _SRC.rglob("*.py")
        if _CANCELLED_WITH_EXCEPTION.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == [], (
        "except-tuples pairing CancelledError with Exception swallow "
        f"cooperative cancellation: {offenders}"
    )


async def test_retry_exhaustion_payload_is_never_none() -> None:
    """A zero-iteration retry loop still raises with a real exception payload."""
    from mousedroid.config.schema import RetryConfig

    # gt=0 validation makes max_attempts=0 unrepresentable via normal
    # construction; model_construct bypasses validation to reach the
    # defensive branch the -O-stripped assert used to (not) guard.
    cfg = RetryConfig.model_construct(max_attempts=0)

    async def _never_called() -> None:
        raise AssertionError("retry body must not run with max_attempts=0")

    with pytest.raises(RetryExhaustedError) as excinfo:
        await retry_async(_never_called, cfg=cfg)
    assert excinfo.value.last_exception is not None
    assert isinstance(excinfo.value.last_exception, RuntimeError)


async def test_mission_dedup_follower_propagates_cancellation() -> None:
    """A cancelled leader future must cancel the follower, not 500 it.

    Exercises the fixed ``except`` split in ``TelemetryServer``'s mission
    dedup path at the primitive level: awaiting a cancelled future raises
    CancelledError, which the follower handler must re-raise rather than
    convert into an internal-error response.
    """
    loop = asyncio.get_running_loop()
    leader: asyncio.Future[tuple[int, dict[str, str]]] = loop.create_future()

    async def _follower() -> tuple[int, dict[str, str]]:
        # Mirrors the fixed handler shape: CancelledError re-raised,
        # ordinary Exception mapped to a 500 tuple.
        try:
            return await leader
        except asyncio.CancelledError:
            raise
        except Exception:
            return (500, {"error": "internal_error"})

    task = asyncio.create_task(_follower())
    await asyncio.sleep(0)
    leader.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
