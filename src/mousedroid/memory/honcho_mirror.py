"""Honcho memory mirror — remote long-horizon memory sync.

Mirrors non-sensitive journal entries to Honcho for long-horizon recall.
Recalled text is treated as prompt-injection-capable and sanitized.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema.agents import HonchoConfig
    from mousedroid.harness.journal.protocol import JournalEntry, JournalProtocol
    from mousedroid.security.injection_filter import PromptInjectionFilterProtocol

_log = get_logger(__name__)

# Categories safe to mirror to an external service.
# NEVER include 'safety_state' — those are local-authoritative only.
_SAFE_SYNC_CATEGORIES: frozenset[str] = frozenset({"mission", "operator_preference"})


class HonchoMemoryMirror:
    """Honcho memory mirror — mirrors journal entries to remote memory.

    Reads from a ``JournalProtocol`` source, filters to safe categories,
    and syncs summaries to Honcho. Recalled text is treated as
    prompt-injection-capable and sanitized before return.

    Args:
        cfg: Honcho configuration.
        journal: Optional journal source to read entries from.
        injection_filter: Optional injection filter for sanitizing
            recalled text.
    """

    def __init__(
        self,
        cfg: HonchoConfig,
        *,
        journal: JournalProtocol | None = None,
        injection_filter: PromptInjectionFilterProtocol | None = None,
    ) -> None:
        """Initialize the Honcho mirror.

        Args:
            cfg: Honcho configuration.
            journal: Optional journal source.
            injection_filter: Optional injection filter for
                sanitizing recalled text.
        """
        self._cfg = cfg
        self._journal = journal
        self._injection_filter = injection_filter
        self._honcho: Any = None
        self._client: Any = None
        self._degraded = False
        self._last_sync_ts_ns: int = 0

    async def start(self) -> None:
        """Start the mirror and lazily import Honcho SDK."""
        try:
            import importlib

            honcho = importlib.import_module("honcho")
            if honcho is None or not hasattr(honcho, "Client"):
                self._honcho = None
                self._client = None
                self._degraded = True
                _log.warning("honcho_sdk_missing_degraded", degraded=True)
                return

            self._honcho = honcho
            api_key = self._cfg.api_key.get_secret_value()
            app_name = self._cfg.app_name
            if api_key:
                self._client = honcho.Client(api_key=api_key, app_name=app_name)
            else:
                self._client = honcho.Client(app_name=app_name)
            self._degraded = False
            _log.info("honcho_mirror_started", app_name=app_name)
        except (ImportError, ModuleNotFoundError):
            self._honcho = None
            self._client = None
            self._degraded = True
            _log.warning("honcho_sdk_missing_degraded", degraded=True)
        except Exception as exc:
            self._honcho = None
            self._client = None
            self._degraded = True
            _log.warning(
                "honcho_mirror_start_failed",
                error=str(exc),
                degraded=True,
            )

    async def stop(self) -> None:
        """Stop the mirror and release resources."""
        self._honcho = None
        self._client = None
        self._degraded = False
        _log.info("honcho_mirror_stopped")

    async def sync_to_remote(self) -> int:
        """Sync safe journal entries to Honcho.

        Iterates all entries in the journal via ``read_all()``, filters
        to safe categories, and pushes those written after the last sync
        timestamp.

        Returns:
            The number of entries successfully mirrored.
        """
        if self._degraded or not self._client or not self._journal:
            return 0

        synced_count = 0
        cutoff = self._last_sync_ts_ns
        max_ts: int = cutoff

        try:
            async for entry in self._journal.read_all():
                if entry.ts_ns <= cutoff:
                    continue
                if entry.category not in _SAFE_SYNC_CATEGORIES:
                    continue

                # Mirror entry payload as a serialized summary
                summary = _entry_to_summary(entry)
                await asyncio.to_thread(
                    self._client.add_memory,
                    content=summary,
                    metadata={"category": entry.category},
                )
                synced_count += 1
                max_ts = max(max_ts, entry.ts_ns)

            self._last_sync_ts_ns = max_ts
            _log.info(
                "honcho_mirror_sync_complete",
                count=synced_count,
            )
            return synced_count
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _log.warning(
                "honcho_mirror_sync_failed",
                error=str(exc),
            )
            return synced_count

    async def recall(self, query: str, *, k: int = 5) -> list[str]:
        """Recall memories from Honcho based on a query.

        Args:
            query: The search query.
            k: The number of results to return.

        Returns:
            A list of sanitized recalled text snippets.
        """
        if not query or self._degraded or not self._client:
            return []

        recall_limit = min(k, self._cfg.max_recall_results)

        try:
            results = await asyncio.to_thread(
                self._client.query,
                query=query,
                k=recall_limit,
            )

            recalled: list[str] = []
            for res in results:
                text: str = str(getattr(res, "content", res))
                if self._cfg.sanitize_recalled and self._injection_filter:
                    text = self._injection_filter.sanitize(text)
                recalled.append(text)

            return recalled
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _log.warning(
                "honcho_mirror_recall_failed",
                error=str(exc),
            )
            return []

    @property
    def is_degraded(self) -> bool:
        """Check if the mirror is in a degraded state."""
        return self._degraded


def _entry_to_summary(entry: JournalEntry) -> str:
    """Convert a journal entry to a human-readable summary string.

    Args:
        entry: The journal entry to summarize.

    Returns:
        A one-line summary suitable for remote storage.
    """
    parts = [f"[{entry.category}]"]
    if entry.event:
        parts.append(entry.event)
    if entry.payload:
        # Only include scalar values for safety
        safe_payload = {
            k: v for k, v in entry.payload.items() if isinstance(v, (str, int, float, bool))
        }
        if safe_payload:
            parts.append(json.dumps(safe_payload, sort_keys=True))
    return " ".join(parts)


__all__ = ["HonchoMemoryMirror"]
