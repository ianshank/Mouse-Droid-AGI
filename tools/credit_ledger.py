"""F-059 credit ledger: a pre-call spend cap and an append-only spend log.

The credit workstreams (A: request profile + latency probe, B: Agent-SDK peer
reviewer, C: batch triage) spend real money through agents. This module sits
between them and the cap the operator set in ``.claude/workforce.yaml``
(``credit:``), and keeps the evidence that every run stayed under it.

* :func:`charge` - call it BEFORE the API call. It raises :class:`CapExceeded`
  when the workstream's recorded spend plus the estimate would exceed its cap.
  It reads the ledger and never writes it.
* :func:`record` - call it AFTER the run, with what it actually cost. It appends
  ONE JSON object to the ledger (one per line, append-only, LF endings) and fills
  in the fields only the ledger can know: ``ts``, ``cap_usd`` and
  ``cumulative_ws_usd``.
* :func:`spent_usd` - the recorded spend of one workstream.
* ``python -m tools.credit_ledger status`` - cap / spent / remaining per
  workstream; exits 1 when one is over its cap or the ledger is unreadable.

Design rules:

* **No literals.** Caps and the ledger path come from the workforce config
  (:class:`tools.claude_hooks.config.CreditConfig`), read through the same
  loader the hooks use. Nothing here restates a number.
* **Fails closed.** A workstream with no configured cap, an unreadable config, a
  corrupt ledger line, or an estimate that is negative / ``nan`` / ``inf`` all
  stop the spend. ``nan`` compares false against everything, so a naive
  ``spent + estimate > cap`` would let it through.
* **Exact arithmetic.** Sums and the cap comparison use :class:`~decimal.Decimal`
  built from each number's shortest repr, so ``0.1 + 0.2`` is ``0.3`` and a run
  that lands exactly on the cap is allowed.
* **The ledger never stores a credential** (plan risk R8): ``record`` refuses a
  row that contains an Anthropic key anywhere in it. The ledger is a tracked file.
* **A pre-check, not a lock.** Two runs started together can both pass
  :func:`charge`; the workstreams are operator-driven and sequential.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from tools.claude_hooks.config import ConfigError, CreditConfig, load_config
from tools.claude_hooks.paths import resolve_repo_root

#: Prefix of every Anthropic API key. A ledger row containing it is refused.
_KEY_PREFIX = "sk-ant-"

#: Ledger columns in the order they are written (the F-059 plan's example row).
LEDGER_FIELDS: tuple[str, ...] = (
    "ts",
    "ws",
    "run_id",
    "target",
    "model_requested",
    "model_resolved",
    "resolved_via",
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cost_usd",
    "cost_source",
    "cap_usd",
    "cumulative_ws_usd",
    "artifact",
    "status",
)

#: Fields only the ledger can compute; a caller-supplied value is overwritten.
_DERIVED_FIELDS = frozenset({"cap_usd", "cumulative_ws_usd"})
_STRING_FIELDS = (
    "ws",
    "run_id",
    "target",
    "model_requested",
    "model_resolved",
    "resolved_via",
    "cost_source",
    "artifact",
    "status",
)
_TOKEN_FIELDS = ("input_tokens", "output_tokens", "cache_read_tokens")
#: Token columns a caller may omit (recorded as 0).
_OPTIONAL_DEFAULTS: dict[str, Any] = {"cache_read_tokens": 0}
_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
_CURRENCY_QUANTUM = Decimal("0.01")


class CapExceeded(RuntimeError):  # noqa: N818 - a refusal event (cf. InjectionRejected), not a fault
    """Raised BEFORE an API call when it could push a workstream past its cap."""


class LedgerError(RuntimeError):
    """Raised when the ledger file cannot be trusted (spending stops until fixed)."""


@dataclass(frozen=True)
class Headroom:
    """What :func:`charge` found: the cap, the spend so far, and the estimate."""

    ws: str
    cap_usd: float
    spent_usd: float
    estimate_usd: float

    @property
    def remaining_usd(self) -> float:
        """Cap left after the recorded spend AND this estimate (``>= 0`` when allowed)."""
        return self.cap_usd - self.spent_usd - self.estimate_usd


def _resolve(config: CreditConfig | None, repo_root: Path | None) -> tuple[CreditConfig, Path]:
    """Return the credit config and the absolute ledger path.

    Raises:
        ConfigError: The workforce config exists but is unreadable or invalid.
    """
    root = resolve_repo_root() if repo_root is None else repo_root
    cfg = load_config(repo_root=root).credit if config is None else config
    return cfg, root / cfg.ledger


def _decimal(value: object, *, what: str, error: type[Exception]) -> Decimal:
    """Convert a JSON/number value to a finite, non-negative ``Decimal`` or raise ``error``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise error(f"{what} must be a number, got {value!r}")
    if isinstance(value, float) and not math.isfinite(value):
        raise error(f"{what} must be finite, got {value!r}")
    number = Decimal(str(value))
    if number < 0:
        raise error(f"{what} must be >= 0, got {value!r}")
    return number


def _read_rows(path: Path) -> list[dict[str, Any]]:
    """Parse the ledger. A missing file is an empty ledger; a damaged one is an error."""
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LedgerError(f"{path}:{number}: not valid JSON ({exc.msg})") from exc
        if not isinstance(row, dict):
            raise LedgerError(f"{path}:{number}: expected a JSON object, got {type(row).__name__}")
        rows.append(row)
    return rows


def _spent(rows: Sequence[Mapping[str, Any]], ws: str, path: Path) -> Decimal:
    """Sum ``cost_usd`` over the rows of workstream ``ws``."""
    total = Decimal(0)
    for index, row in enumerate(rows, start=1):
        if row.get("ws") != ws:
            continue
        try:
            total += _decimal(row.get("cost_usd"), what="cost_usd", error=LedgerError)
        except LedgerError as exc:
            raise LedgerError(f"{path}: row {index} for workstream {ws!r}: {exc}") from exc
    return total


def _cap_of(cfg: CreditConfig, ws: str, error: type[Exception]) -> Decimal:
    cap = cfg.caps_usd.get(ws)
    if cap is None:
        known = ", ".join(sorted(cfg.caps_usd)) or "none"
        raise error(
            f"no spend cap is configured for workstream {ws!r} in credit.caps_usd "
            f"(.claude/workforce.yaml; configured: {known})"
        )
    return Decimal(str(cap))


def spent_usd(
    ws: str, *, config: CreditConfig | None = None, repo_root: Path | None = None
) -> float:
    """Return the USD recorded so far for workstream ``ws`` (0.0 for an empty ledger)."""
    _cfg, path = _resolve(config, repo_root)
    return float(_spent(_read_rows(path), ws, path))


def charge(
    ws: str,
    estimate_usd: float,
    *,
    config: CreditConfig | None = None,
    repo_root: Path | None = None,
) -> Headroom:
    """Authorise one API call of about ``estimate_usd`` for workstream ``ws``.

    Call this BEFORE the call. It does not write the ledger: the actual cost is
    recorded afterwards with :func:`record`.

    Args:
        ws: Workstream name; must have a cap in ``credit.caps_usd``.
        estimate_usd: Upper-bound estimate of the call's cost in USD.
        config: Credit config; loaded from the repo's workforce config when omitted.
        repo_root: Repository root; resolved when omitted.

    Returns:
        The :class:`Headroom` found, so the caller can log it.

    Raises:
        CapExceeded: ``spent + estimate`` would exceed the cap, the workstream is
            already over it, or it has no configured cap (fail closed).
        ValueError: ``estimate_usd`` is negative, not finite, or not a number.
        LedgerError: The ledger is unreadable or damaged.
        ConfigError: The workforce config is unreadable or invalid.
    """
    estimate = _decimal(estimate_usd, what="estimate_usd", error=ValueError)
    cfg, path = _resolve(config, repo_root)
    cap = _cap_of(cfg, ws, CapExceeded)
    spent = _spent(_read_rows(path), ws, path)
    if spent + estimate > cap:
        raise CapExceeded(
            f"workstream {ws!r} would exceed its ${cap} cap: ${spent} already recorded "
            f"+ ${estimate} estimated (ledger: {path})"
        )
    return Headroom(ws=ws, cap_usd=float(cap), spent_usd=float(spent), estimate_usd=float(estimate))


def _validated_fields(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Check ``entry`` against the ledger schema and return its caller-owned fields."""
    allowed = set(LEDGER_FIELDS)
    unknown = sorted(set(entry) - allowed)
    if unknown:
        raise ValueError(f"unknown ledger field(s): {', '.join(unknown)}")
    fields = {k: v for k, v in entry.items() if k not in _DERIVED_FIELDS}
    for key, default in _OPTIONAL_DEFAULTS.items():
        fields.setdefault(key, default)
    required = (set(LEDGER_FIELDS) - _DERIVED_FIELDS) - {"ts"}
    missing = sorted(required - set(fields))
    if missing:
        raise ValueError(f"missing ledger field(s): {', '.join(missing)}")
    for key in _STRING_FIELDS:
        if not isinstance(fields[key], str) or not fields[key].strip():
            raise ValueError(f"{key} must be a non-empty string, got {fields[key]!r}")
    for key in _TOKEN_FIELDS:
        value = fields[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{key} must be a non-negative integer, got {value!r}")
    _decimal(fields["cost_usd"], what="cost_usd", error=ValueError)
    ts = fields.get("ts")
    if ts is not None and (not isinstance(ts, str) or not ts.strip()):
        raise ValueError(f"ts must be a non-empty string, got {ts!r}")
    return fields


def record(
    entry: Mapping[str, Any],
    *,
    config: CreditConfig | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Append one run to the ledger and return the row as written.

    ``ts`` (UTC, ``Z``-suffixed) is filled in when absent; ``cap_usd`` and
    ``cumulative_ws_usd`` are always computed here — a caller-supplied value is
    overwritten, because only the ledger knows the earlier rows. A run that
    overshoots its cap is still recorded: the money is spent, and hiding it
    would be worse than showing it (``status`` then reports the breach).

    Args:
        entry: The run: ``ws``, ``run_id``, ``target``, ``model_requested``,
            ``model_resolved``, ``resolved_via``, ``input_tokens``,
            ``output_tokens``, optionally ``cache_read_tokens``, ``cost_usd``,
            ``cost_source``, ``artifact``, ``status`` (and optionally ``ts``).
        config: Credit config; loaded from the repo's workforce config when omitted.
        repo_root: Repository root; resolved when omitted.

    Returns:
        The row appended, in :data:`LEDGER_FIELDS` order.

    Raises:
        ValueError: The entry has an unknown or missing field, a malformed value,
            a workstream with no cap, or contains an Anthropic API key.
        LedgerError: The existing ledger is unreadable or damaged.
        ConfigError: The workforce config is unreadable or invalid.
    """
    fields = _validated_fields(entry)
    cfg, path = _resolve(config, repo_root)
    ws = fields["ws"]
    cap = _cap_of(cfg, ws, ValueError)
    cost = _decimal(fields["cost_usd"], what="cost_usd", error=ValueError)
    cumulative = _spent(_read_rows(path), ws, path) + cost

    fields.setdefault("ts", datetime.now(timezone.utc).strftime(_TIMESTAMP_FORMAT))
    fields["cap_usd"] = float(cap)
    fields["cumulative_ws_usd"] = float(cumulative)
    row = {key: fields[key] for key in LEDGER_FIELDS}

    line = json.dumps(row, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    if _KEY_PREFIX in line:
        raise ValueError(
            "refusing to write a ledger row that contains an Anthropic API credential; "
            "the ledger is a tracked file"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    # "a" appends (O_APPEND); newline="\n" keeps a tracked JSONL file LF on Windows.
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return row


def _money(value: Decimal) -> str:
    return f"{value.quantize(_CURRENCY_QUANTUM):.2f}"


def _status(repo_root: Path) -> int:
    """Print cap / spent / remaining for every configured workstream; 1 on a breach."""
    try:
        cfg, path = _resolve(None, repo_root)
        rows = _read_rows(path)
        lines = [f"{'ws':<4} {'cap_usd':>10} {'spent_usd':>10} {'remaining':>10}"]
        breached = False
        for ws in sorted(cfg.caps_usd):
            cap = _cap_of(cfg, ws, ValueError)
            spent = _spent(rows, ws, path)
            over = spent > cap
            breached = breached or over
            marker = "  OVER" if over else ""
            lines.append(
                f"{ws:<4} {_money(cap):>10} {_money(spent):>10} {_money(cap - spent):>10}{marker}"
            )
    except (ConfigError, LedgerError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print("\n".join(lines))
    return 1 if breached else 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: ``python -m tools.credit_ledger status``."""
    parser = argparse.ArgumentParser(
        prog="python -m tools.credit_ledger",
        description="F-059 credit ledger: per-workstream spend against the workforce.yaml caps.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    status = sub.add_parser("status", help="Print cap / spent / remaining per workstream.")
    status.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root (default: discovered from the workforce tooling).",
    )
    args = parser.parse_args(argv)
    repo_root = resolve_repo_root() if args.repo_root is None else args.repo_root.resolve()
    return _status(repo_root)


if __name__ == "__main__":
    sys.exit(main())
