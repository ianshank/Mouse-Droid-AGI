"""Snapshot exporter that writes ``EpisodicReplay`` to MEMORY.md.

This is a *snapshot* of the in-memory deque, NOT a persistence layer —
the four-tier memory modules
(:mod:`mousedroid.memory.{episodic,semantic,working,consolidation}`)
remain ephemeral. The exporter writes a Markdown summary to a path
shared between the Jetson host and the OpenClaw Mac mini host (typically
a Tailscale shared directory or NFS mount) so the OpenClaw agent can
read recent episodic context between deployments.

Atomicity: writes go to a sibling ``.tmp`` file followed by
:func:`os.replace`, so a crash mid-write leaves the previous valid
snapshot intact. The exporter is async at the surface (callable from
the orchestrator's POST_TICK hook) but delegates the synchronous
``EpisodicReplay.sample`` call to :func:`asyncio.to_thread` so the
30 Hz hot loop is not blocked.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.memory.episodic import EpisodicReplay

_log = get_logger(__name__)

SCHEMA_VERSION = 1

# Per-entry display cap so MEMORY.md stays readable even when episodic
# payloads are large dicts. Configurable per exporter instance via the
# ``entry_truncate_chars`` constructor arg; this constant is the default.
_DEFAULT_ENTRY_TRUNCATE_CHARS = 240


@runtime_checkable
class MemoryExporterProtocol(Protocol):
    """Async snapshot exporter for the in-memory episodic replay buffer."""

    async def export(self, replay: EpisodicReplay) -> Path | None:
        """Write a snapshot and return the path, or ``None`` on a no-op.

        Returns ``None`` when the exporter is disabled (e.g. no shared
        path configured) or when the replay buffer is empty.
        """
        ...


class MarkdownReplayExporter:
    """Default :class:`MemoryExporterProtocol` implementation."""

    __slots__ = ("_entry_truncate_chars", "_max_entries", "_path")

    def __init__(
        self,
        path: Path,
        *,
        max_entries: int = 32,
        entry_truncate_chars: int = _DEFAULT_ENTRY_TRUNCATE_CHARS,
    ) -> None:
        """Initialise the exporter.

        Args:
            path: Destination filename (typically ``MEMORY.md`` on a
                shared volume). Parent directory must exist.
            max_entries: Cap on the number of episodic samples included
                in the snapshot. Bounded so ``MEMORY.md`` stays small
                enough for the OpenClaw agent's context window.
            entry_truncate_chars: Per-entry display cap (in characters).
                Defaults to :data:`_DEFAULT_ENTRY_TRUNCATE_CHARS`.
        """
        if max_entries <= 0:
            msg = "max_entries must be positive"
            raise ValueError(msg)
        if entry_truncate_chars <= 0:
            msg = "entry_truncate_chars must be positive"
            raise ValueError(msg)
        self._path = path
        self._max_entries = max_entries
        self._entry_truncate_chars = entry_truncate_chars

    @property
    def path(self) -> Path:
        """Destination path."""
        return self._path

    async def export(self, replay: EpisodicReplay) -> Path | None:
        """Snapshot the replay buffer and atomically replace MEMORY.md.

        Every disk-touching call is offloaded via :func:`asyncio.to_thread`
        so the 30 Hz orchestrator tick that triggers this hook never
        blocks on filesystem latency — important when the configured
        ``shared_memory_path`` lives on a Tailscale-shared NFS mount or
        SMB share where ``mkdir`` / ``write_text`` / ``replace`` can each
        cost tens of milliseconds.
        """
        if len(replay) == 0:
            _log.debug(
                "memory_export_skipped",
                reason="empty_replay",
                path=str(self._path),
            )
            return None
        start = time.monotonic()
        # ``EpisodicReplay.sample`` is sync; offload to a thread so the
        # orchestrator's hot loop is not blocked during the snapshot.
        samples = await asyncio.to_thread(replay.sample, self._max_entries)
        body = _render_markdown(
            samples,
            replay_size=len(replay),
            entry_truncate_chars=self._entry_truncate_chars,
        )
        try:
            await asyncio.to_thread(self._write_atomic, body)
        except OSError as exc:
            _log.warning(
                "memory_export_failed",
                path=str(self._path),
                error=f"{type(exc).__name__}:{exc}",
            )
            return None
        latency_ms = (time.monotonic() - start) * 1000.0
        _log.info(
            "memory_export_completed",
            path=str(self._path),
            count=len(samples),
            bytes=len(body),
            latency_ms=latency_ms,
        )
        return self._path

    def _write_atomic(self, body: str) -> None:
        """Synchronous atomic write, intended to be run via ``asyncio.to_thread``.

        Splitting this out keeps :meth:`export` free of any sync I/O and
        makes the call site grep-able for callers who want to confirm
        the off-loop guarantee.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(body, encoding="utf-8")
        os.replace(tmp, self._path)


def _render_markdown(samples: list[Any], *, replay_size: int, entry_truncate_chars: int) -> str:
    """Render the snapshot body. Stable format keyed by SCHEMA_VERSION.

    Front-matter is YAML so OpenClaw skills can parse it without a full
    Markdown grammar. Sample bodies are best-effort string conversions
    — whatever ``EpisodicReplay.sample`` returns gets ``repr()``-ed,
    truncated to keep the file readable.
    """
    timestamp_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out: list[str] = [
        "---",
        f"schema_version: {SCHEMA_VERSION}",
        f"timestamp: {timestamp_iso}",
        f"replay_size: {replay_size}",
        f"sample_count: {len(samples)}",
        "---",
        "",
        "# Episodic Replay Snapshot",
        "",
        "This file is a **snapshot**, not durable storage. See `docs/openclaw_skills/README.md`.",
        "",
        "## Recent experiences",
        "",
    ]
    for idx, item in enumerate(samples):
        out.append(f"- `[{idx}]` {_truncate(repr(item), entry_truncate_chars)}")
    out.append("")
    return "\n".join(out)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


__all__ = [
    "SCHEMA_VERSION",
    "MarkdownReplayExporter",
    "MemoryExporterProtocol",
]
