"""A thin adapter between the reviewer and ``tools/credit_ledger.py`` (F-059 WS-B / WS-D).

* :func:`precheck` - ``credit_ledger.charge`` for the reviewer's workstream, called
  BEFORE the SDK is touched. It reads the ledger and never writes it.
* :func:`account_run` - turns what a run produced (or failed to produce) into the
  numbers and the status that go in the row.
* :func:`record_run` - ``credit_ledger.record`` with those numbers.

How a run is costed
-------------------
The cost is ``ResultMessage.total_cost_usd`` - the CLI's own whole-tree figure
(subagents included), present on success AND on error results. When there is no
trustworthy number the row is charged the worst case, not zero:

============================================  ===========  =====================
situation                                     cost_usd     cost_source
============================================  ===========  =====================
result with a usable ``total_cost_usd``       that number  ``ResultMessage.total_cost_usd``
result whose cost is missing / NaN / negative the run cap  ``cap_assumed``
no result, the run had started (crash, ^C,    the run cap  ``cap_assumed``
timeout)
no result, nothing had started                0.0          ``not_started``
============================================  ===========  =====================

"The run cap" is the ``--cap-usd`` the operator authorised, which is also the SDK's
``max_budget_usd``: a run cannot have knowingly spent more than it, so charging it
only over-counts, and an over-count is corrected by the operator reading the
console while an under-count lets the next run overspend.

Token columns come from ``ResultMessage.model_usage`` (all models, subagents
included) and fall back to ``usage`` (the main loop only). Cache-creation tokens are
newly processed prompt tokens and are folded into ``input_tokens``; the ledger has no
column for them.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from tools import credit_ledger
from tools.claude_hooks.config import WorkforceConfig

COST_SOURCE_SDK: Final = "ResultMessage.total_cost_usd"
COST_SOURCE_ASSUMED: Final = "cap_assumed"
COST_SOURCE_NOT_STARTED: Final = "not_started"

STATUS_OK: Final = "ok"
STATUS_ERROR: Final = "error"
STATUS_ERROR_API: Final = "error_api"
STATUS_INTERRUPTED: Final = "interrupted"
STATUS_TIMEOUT: Final = "timeout"
STATUS_NO_RESULT: Final = "error:no_result"
STATUS_OUTPUT_WITHHELD: Final = "output_withheld"

_RESULT_SUCCESS: Final = "success"
_TERMINAL_API_ERROR: Final = "api_error"


class LedgerWriteError(RuntimeError):
    """The spend row could not be written after the money was spent.

    Attributes:
        entry: The row that was meant to be appended (without the fields the ledger
            computes itself: ``ts``, ``cap_usd``, ``cumulative_ws_usd``).
    """

    def __init__(self, entry: dict[str, Any], cause: Exception) -> None:
        """Keep the entry and describe the cause."""
        super().__init__(f"{type(cause).__name__}: {cause}")
        self.entry = entry


@dataclass(frozen=True)
class RunAccounting:
    """What one run cost and how it ended, ready for the ledger row."""

    cost_usd: float
    cost_source: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    status: str


def precheck(cfg: WorkforceConfig, cap_usd: float, *, repo_root: Path) -> credit_ledger.Headroom:
    """Authorise a run of up to ``cap_usd`` for the reviewer's workstream, or raise.

    Call this BEFORE any client is built or any API call is made.

    Raises:
        credit_ledger.CapExceeded: Recorded spend plus ``cap_usd`` would pass the cap.
        credit_ledger.LedgerError: The ledger is unreadable or damaged.
        ValueError: ``cap_usd`` is negative, not finite or not a number.
    """
    return credit_ledger.charge(
        cfg.agent_reviewer.ws, cap_usd, config=cfg.credit, repo_root=repo_root
    )


def _usable_cost(value: object) -> float | None:
    """Return ``value`` as a float if it is a finite, non-negative number, else ``None``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or value < 0:
        return None
    return float(value)


def _count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def _tokens_from_model_usage(model_usage: Mapping[str, Any]) -> tuple[int, int, int] | None:
    entries = [entry for entry in model_usage.values() if isinstance(entry, Mapping)]
    if not entries:
        return None
    return (
        sum(
            _count(e.get("inputTokens")) + _count(e.get("cacheCreationInputTokens"))
            for e in entries
        ),
        sum(_count(e.get("outputTokens")) for e in entries),
        sum(_count(e.get("cacheReadInputTokens")) for e in entries),
    )


def _tokens_of(result: Any | None) -> tuple[int, int, int]:
    if result is None:
        return 0, 0, 0
    model_usage = getattr(result, "model_usage", None)
    if isinstance(model_usage, Mapping):
        totals = _tokens_from_model_usage(model_usage)
        if totals is not None:
            return totals
    usage = getattr(result, "usage", None)
    if isinstance(usage, Mapping):
        return (
            _count(usage.get("input_tokens")) + _count(usage.get("cache_creation_input_tokens")),
            _count(usage.get("output_tokens")),
            _count(usage.get("cache_read_input_tokens")),
        )
    return 0, 0, 0


def _cost_of(result: Any | None, *, started: bool, cap_usd: float) -> tuple[float, str]:
    if result is not None:
        reported = _usable_cost(getattr(result, "total_cost_usd", None))
        if reported is not None:
            return reported, COST_SOURCE_SDK
        return cap_usd, COST_SOURCE_ASSUMED
    if not started:
        return 0.0, COST_SOURCE_NOT_STARTED
    return cap_usd, COST_SOURCE_ASSUMED


def _result_status(result: Any) -> str:
    subtype = getattr(result, "subtype", None)
    if isinstance(subtype, str) and subtype and subtype != _RESULT_SUCCESS:
        return subtype
    if getattr(result, "is_error", False):
        if getattr(result, "terminal_reason", None) == _TERMINAL_API_ERROR:
            return STATUS_ERROR_API
        return STATUS_ERROR
    return STATUS_OK


def _error_status(error: BaseException | None) -> str:
    if error is None:
        return STATUS_NO_RESULT
    if isinstance(error, (KeyboardInterrupt, asyncio.CancelledError)):
        return STATUS_INTERRUPTED
    if isinstance(error, (TimeoutError, asyncio.TimeoutError)):
        return STATUS_TIMEOUT
    return f"error:{type(error).__name__}"


def account_run(
    result: Any | None,
    *,
    started: bool,
    cap_usd: float,
    error: BaseException | None,
) -> RunAccounting:
    """Work out what a run cost and how it ended.

    Args:
        result: The SDK's ``ResultMessage`` (duck-typed), or ``None`` if none arrived.
        started: Whether the CLI produced any message, i.e. the session began and
            may have spent money.
        cap_usd: The run's authorised budget, charged when the true cost is unknown.
        error: The exception that ended the run, if any.

    Returns:
        The cost, where it came from, the token counts and a status.
    """
    cost_usd, cost_source = _cost_of(result, started=started, cap_usd=cap_usd)
    input_tokens, output_tokens, cache_read_tokens = _tokens_of(result)
    status = _result_status(result) if result is not None else _error_status(error)
    return RunAccounting(
        cost_usd=cost_usd,
        cost_source=cost_source,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        status=status,
    )


def record_run(
    cfg: WorkforceConfig,
    *,
    accounting: RunAccounting,
    run_id: str,
    target: str,
    model_requested: str,
    model_resolved: str,
    resolved_via: str,
    artifact: str,
    repo_root: Path,
) -> dict[str, Any]:
    """Append this run's row to the ledger and return it as written.

    Raises:
        LedgerWriteError: The row could not be appended (a damaged ledger, a failed
            validation, a disk error). The money is already spent, so the error carries
            the entry for the operator to append by hand.
    """
    entry: dict[str, Any] = {
        "ws": cfg.agent_reviewer.ws,
        "run_id": run_id,
        "target": target,
        "model_requested": model_requested,
        "model_resolved": model_resolved,
        "resolved_via": resolved_via,
        "input_tokens": accounting.input_tokens,
        "output_tokens": accounting.output_tokens,
        "cache_read_tokens": accounting.cache_read_tokens,
        "cost_usd": accounting.cost_usd,
        "cost_source": accounting.cost_source,
        "artifact": artifact,
        "status": accounting.status,
    }
    try:
        return credit_ledger.record(entry, config=cfg.credit, repo_root=repo_root)
    except (credit_ledger.LedgerError, ValueError, OSError) as exc:
        raise LedgerWriteError(entry, exc) from exc
