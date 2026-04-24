"""Episodic memory synchronisation to Google Cloud Firestore.

Periodically samples recent episodic memory entries and upserts them to a
Firestore collection for cross-run persistence and cloud-side analytics.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import inspect
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from mousedroid.common.async_utils import cancel_and_drain, spawn_tracked
from mousedroid.logging.setup import get_logger

_TRANSIENT_FIRESTORE_EXCEPTIONS: tuple[type[BaseException], ...]

try:
    from google.api_core.exceptions import (
        DeadlineExceeded,
        GoogleAPIError,
        RetryError,
        ServiceUnavailable,
    )

    _TRANSIENT_FIRESTORE_EXCEPTIONS = (
        GoogleAPIError,
        RetryError,
        DeadlineExceeded,
        ServiceUnavailable,
        TimeoutError,
        ConnectionError,
        OSError,
    )
except ImportError:  # pragma: no cover - optional cloud dependency
    _TRANSIENT_FIRESTORE_EXCEPTIONS = (TimeoutError, ConnectionError, OSError)

if TYPE_CHECKING:
    from mousedroid.config.schema import GCPConfig
    from mousedroid.memory.episodic import EpisodicReplay

_log = get_logger(__name__)


class CloudFirestoreSync:
    """Syncs episodic memory entries to Firestore.

    Args:
        cfg: GCP configuration.
        episodic: Episodic replay buffer to sync from.
    """

    def __init__(self, cfg: GCPConfig, episodic: EpisodicReplay) -> None:
        self._cfg = cfg
        self._fs_cfg = cfg.firestore
        self._episodic = episodic
        self._robot_id = cfg.robot_id
        self._interval_s = self._fs_cfg.sync_interval_s

        self._db: Any | None = None
        self._collection_ref: Any | None = None
        self._task: asyncio.Task[None] | None = None
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._running = False

    async def start(self) -> None:
        """Initialise the Firestore client and start the sync loop."""
        from mousedroid.cloud._auth import resolve_credentials

        firestore_module = importlib.import_module("google.cloud.firestore")
        firestore_api = cast(Any, firestore_module)
        creds, _project = resolve_credentials(self._cfg)
        async_client_factory = cast(
            Callable[..., Any],
            firestore_api.AsyncClient,
        )
        self._db = async_client_factory(
            credentials=creds,
            project=self._cfg.project_id,
        )
        self._collection_ref = self._db.collection(self._fs_cfg.collection)
        self._running = True
        self._task = spawn_tracked(
            self._background_tasks,
            self._sync_loop(),
            name=self._sync_loop.__name__,
        )
        _log.info(
            "cloud_firestore_sync_started",
            collection=self._fs_cfg.collection,
            interval_s=self._interval_s,
        )

    async def sync_once(self) -> int:
        """Run a single sync cycle.

        Returns:
            Number of episodes synced.
        """
        if self._collection_ref is None:
            return 0

        batch_size = self._fs_cfg.sync_batch_size
        episodes = self._episodic.sample(min(batch_size, len(self._episodic)))
        if not episodes:
            return 0

        count = 0
        for experience, priority in episodes:
            try:
                doc_data: dict[str, Any] = {
                    "robot_id": self._robot_id,
                    "priority": float(priority),
                    "synced_at": time.time(),
                }
                if hasattr(experience, "timestamp"):
                    doc_data["timestamp"] = experience.timestamp
                if hasattr(experience, "reward"):
                    doc_data["reward"] = float(experience.reward)
                if hasattr(experience, "surprise"):
                    doc_data["surprise"] = float(experience.surprise)
                if hasattr(experience, "distance_m"):
                    doc_data["distance_m"] = float(experience.distance_m)

                doc_id = f"{self._robot_id}_{doc_data.get('timestamp', time.time())}"
                await self._collection_ref.document(doc_id).set(doc_data)
                count += 1
            except _TRANSIENT_FIRESTORE_EXCEPTIONS:
                _log.debug(
                    "cloud_firestore_sync_entry_failed",
                    transient=True,
                    exc_info=True,
                )
            except Exception:
                _log.warning(
                    "cloud_firestore_sync_entry_failed",
                    transient=False,
                    exc_info=True,
                )

        if count > 0:
            _log.debug("cloud_firestore_synced", count=count)
        return count

    async def close(self) -> None:
        """Stop the sync loop and release resources."""
        self._running = False
        if self._task is not None:
            if self._task in self._background_tasks:
                await cancel_and_drain(self._background_tasks)
            elif not self._task.done():
                self._task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._task
            self._background_tasks.discard(self._task)
            self._task = None
        if self._db is not None:
            try:
                close_result = self._db.close()
                if inspect.isawaitable(close_result):
                    await close_result
            except _TRANSIENT_FIRESTORE_EXCEPTIONS:
                _log.debug("cloud_firestore_close_error", transient=True, exc_info=True)
            except Exception:
                _log.warning("cloud_firestore_close_error", transient=False, exc_info=True)
            self._db = None
        _log.info("cloud_firestore_sync_closed")

    async def _sync_loop(self) -> None:
        """Periodic sync loop — runs as a background task."""
        while self._running:
            try:
                await asyncio.sleep(self._interval_s)
                await self.sync_once()
            except asyncio.CancelledError:
                break
            except _TRANSIENT_FIRESTORE_EXCEPTIONS:
                _log.warning("cloud_firestore_sync_error", transient=True, exc_info=True)
            except Exception:
                _log.warning("cloud_firestore_sync_error", transient=False, exc_info=True)
