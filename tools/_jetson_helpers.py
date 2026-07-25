"""Shared Jetson-side helpers for operator probe tools.

Promoted out of ``tools/llm_latency_probe.py`` during the F-006 remote-LLM
sprint so the new ``tools/jetson_remote_llm_probe.py`` (and any future
operator probes) can reuse the tegrastats-snapshot logic instead of each
tool reinventing it. Keeping a single source means future fixes (e.g.
parsing a new tegrastats field) land in one place.

Architecture invariants (per CLAUDE.md):

* Structured logging via ``mousedroid.logging.setup.get_logger`` — every
  diagnostic event is operator-dashboard-ingestible.
* Never raises — every fallback path returns a documented sentinel value
  (``None`` for missing-binary / parse-fail / timeout) so the calling
  probe doesn't need to wrap us in another try/except.
* No hardcoded paths — relies on ``shutil.which("tegrastats")`` so any
  ``$PATH``-installed tegrastats binary is found (Jetson hosts always have
  it at ``/usr/bin/tegrastats``; the container does NOT inherit it).
"""

from __future__ import annotations

import re
import shutil
import subprocess

from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)

# tegrastats output line shape (Orin Nano):
#   RAM 2914/7619MB ...  GR3D_FREQ 0%@[306,...]  ... NVRGTX_FREQ ...
# We extract two numbers: used RAM (MB) and total RAM (MB). On non-Jetson hosts
# tegrastats is absent — the helper falls back to ``None`` for the snapshot
# fields. We deliberately don't try to use ``nvidia-smi`` because the Orin
# Nano's iGPU shares system RAM (UMA) — RAM is the right signal here, not
# discrete VRAM.
_TEGRASTATS_RAM_RE = re.compile(r"\bRAM\s+(\d+)/(\d+)MB")


def tegrastats_snapshot(timeout_s: float = 2.0) -> dict[str, int | str | None]:
    """Capture one ``tegrastats`` line and parse the RAM usage.

    Returns a dict with keys ``ram_used_mb``, ``ram_total_mb``, ``raw_line``.
    All values are ``None`` when ``tegrastats`` is absent (non-Jetson host)
    or the parse fails. Never raises.

    Args:
        timeout_s: Max seconds to wait for one tegrastats sample before
            falling back to all-``None``. Default 2.0 covers the Orin
            Nano's typical 100ms sample interval with margin.

    Returns:
        Dict with keys ``ram_used_mb`` (int | None), ``ram_total_mb``
        (int | None), ``raw_line`` (str | None). All three are ``None``
        on any failure mode; the dict shape is stable so callers can
        ``snapshot["ram_used_mb"]`` without ``KeyError``.
    """
    if shutil.which("tegrastats") is None:
        _log.warning("tegrastats_not_available", host_kind="non_jetson")
        return {"ram_used_mb": None, "ram_total_mb": None, "raw_line": None}
    try:
        # ``--interval`` is in ms; ``--count`` exits after N samples.
        # S603/S607: fixed argv list, no shell, executable resolved via PATH
        # (the shutil.which check above guarantees presence). Operator-side
        # diagnostic tool — not exposed to untrusted input.
        result = subprocess.run(
            ["tegrastats", "--interval", "100", "--count", "1"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        _log.warning("tegrastats_invocation_failed", error=f"{type(exc).__name__}: {exc}")
        return {"ram_used_mb": None, "ram_total_mb": None, "raw_line": None}

    raw_line = (result.stdout or "").strip().splitlines()[-1:] or [""]
    line = raw_line[0]
    match = _TEGRASTATS_RAM_RE.search(line)
    if match is None:
        _log.warning("tegrastats_parse_failed", raw_line=line[:120])
        return {"ram_used_mb": None, "ram_total_mb": None, "raw_line": line}
    return {
        "ram_used_mb": int(match.group(1)),
        "ram_total_mb": int(match.group(2)),
        "raw_line": line,
    }
