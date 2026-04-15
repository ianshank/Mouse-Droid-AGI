"""Unit tests for CloudFirestoreSync episodic memory synchronisation."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from mousedroid.config.schema import (
    CircuitBreakerConfig,
    GCPConfig,
    GCPFirestoreConfig,
    RetryConfig,
)


def _make_gcp_cfg(**overrides: Any) -> GCPConfig:
    """Create a GCPConfig with test defaults."""
    return GCPConfig(
        project_id="test-project",
        robot_id="droid-test",
        circuit_breaker=CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout_s=1.0,
            half_open_max_calls=1,
        ),
        retry=RetryConfig(max_attempts=1, base_delay_s=0.01),
        **overrides,
    )


def _make_mock_episodic(entries: list[tuple[Any, float]] | None = None) -> MagicMock:
    """Create a mock EpisodicReplay."""
    mock = MagicMock()
    entries = entries or []
    mock.sample.return_value = entries
    mock.__len__ = MagicMock(return_value=len(entries))
    return mock


def test_firestore_sync_init() -> None:
    """CloudFirestoreSync should be constructable without starting."""
    from mousedroid.cloud.firestore_sync import CloudFirestoreSync

    cfg = _make_gcp_cfg()
    episodic = _make_mock_episodic()
    sync = CloudFirestoreSync(cfg, episodic)
    assert sync._db is None
    assert sync._running is False


@pytest.mark.asyncio
async def test_sync_once_no_collection_returns_zero() -> None:
    """sync_once should return 0 when collection_ref is None (before start)."""
    from mousedroid.cloud.firestore_sync import CloudFirestoreSync

    cfg = _make_gcp_cfg()
    episodic = _make_mock_episodic()
    sync = CloudFirestoreSync(cfg, episodic)
    result = await sync.sync_once()
    assert result == 0


@pytest.mark.asyncio
async def test_sync_once_empty_episodes() -> None:
    """sync_once should return 0 when episodic buffer is empty."""
    from mousedroid.cloud.firestore_sync import CloudFirestoreSync

    cfg = _make_gcp_cfg()
    episodic = _make_mock_episodic([])
    sync = CloudFirestoreSync(cfg, episodic)
    sync._collection_ref = MagicMock()
    result = await sync.sync_once()
    assert result == 0


@pytest.mark.asyncio
async def test_sync_once_with_episodes() -> None:
    """sync_once should sync episodes and return count."""
    from mousedroid.cloud.firestore_sync import CloudFirestoreSync

    cfg = _make_gcp_cfg()

    experience = MagicMock()
    experience.timestamp = 1234567890.0
    experience.reward = 0.5
    experience.surprise = 0.3
    experience.distance_m = 1.5

    episodic = _make_mock_episodic([(experience, 0.8)])
    sync = CloudFirestoreSync(cfg, episodic)

    mock_doc = AsyncMock()
    mock_collection = MagicMock()
    mock_collection.document.return_value = mock_doc
    sync._collection_ref = mock_collection

    result = await sync.sync_once()
    assert result == 1
    mock_collection.document.assert_called_once()
    mock_doc.set.assert_awaited_once()


@pytest.mark.asyncio
async def test_sync_once_handles_exception_per_entry() -> None:
    """sync_once should continue syncing even when individual entries fail."""
    from mousedroid.cloud.firestore_sync import CloudFirestoreSync

    cfg = _make_gcp_cfg()

    exp1 = MagicMock()
    exp1.timestamp = 1.0
    exp2 = MagicMock()
    exp2.timestamp = 2.0

    episodic = _make_mock_episodic([(exp1, 0.5), (exp2, 0.6)])
    sync = CloudFirestoreSync(cfg, episodic)

    mock_doc_fail = AsyncMock()
    mock_doc_fail.set.side_effect = RuntimeError("network error")
    mock_doc_ok = AsyncMock()

    mock_collection = MagicMock()
    mock_collection.document.side_effect = [mock_doc_fail, mock_doc_ok]
    sync._collection_ref = mock_collection

    result = await sync.sync_once()
    assert result == 1


@pytest.mark.asyncio
async def test_close_without_start() -> None:
    """close() should be safe to call before start()."""
    from mousedroid.cloud.firestore_sync import CloudFirestoreSync

    cfg = _make_gcp_cfg()
    episodic = _make_mock_episodic()
    sync = CloudFirestoreSync(cfg, episodic)
    await sync.close()
    assert sync._db is None


@pytest.mark.asyncio
async def test_close_with_db() -> None:
    """close() should close the database client."""
    from mousedroid.cloud.firestore_sync import CloudFirestoreSync

    cfg = _make_gcp_cfg()
    episodic = _make_mock_episodic()
    sync = CloudFirestoreSync(cfg, episodic)
    sync._db = MagicMock()
    sync._running = True
    await sync.close()
    assert sync._db is None
    assert sync._running is False


def test_uses_config_batch_size() -> None:
    """sync_once should use sync_batch_size from config."""
    from mousedroid.cloud.firestore_sync import CloudFirestoreSync

    cfg = _make_gcp_cfg(firestore=GCPFirestoreConfig(sync_batch_size=5))
    episodic = _make_mock_episodic()
    sync = CloudFirestoreSync(cfg, episodic)
    assert sync._fs_cfg.sync_batch_size == 5


@pytest.mark.asyncio
async def test_start_initialises_firestore_client() -> None:
    """start() should create a Firestore client and start the sync loop."""
    import sys
    from unittest.mock import patch

    from mousedroid.cloud.firestore_sync import CloudFirestoreSync

    cfg = _make_gcp_cfg()
    episodic = _make_mock_episodic()
    sync = CloudFirestoreSync(cfg, episodic)

    mock_client_cls = MagicMock()
    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_client.collection.return_value = mock_collection
    mock_client_cls.return_value = mock_client

    mock_firestore_module = MagicMock()
    mock_firestore_module.AsyncClient = mock_client_cls

    with (
        patch.dict(
            sys.modules,
            {
                "google": MagicMock(),
                "google.cloud": MagicMock(),
                "google.cloud.firestore": mock_firestore_module,
            },
        ),
        patch("mousedroid.cloud._auth.resolve_credentials") as mock_creds,
    ):
        mock_creds.return_value = (MagicMock(), "test-project")
        await sync.start()

    assert sync._db is not None
    assert sync._collection_ref is not None
    assert sync._running is True
    assert sync._task is not None
    # Clean up
    await sync.close()


@pytest.mark.asyncio
async def test_sync_loop_runs_and_stops() -> None:
    """_sync_loop should periodically call sync_once and stop on cancel."""
    from mousedroid.cloud.firestore_sync import CloudFirestoreSync

    cfg = _make_gcp_cfg(firestore=GCPFirestoreConfig(sync_interval_s=0.01))
    episodic = _make_mock_episodic()
    sync = CloudFirestoreSync(cfg, episodic)
    sync._running = True

    call_count = 0

    async def counting_sync() -> int:
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            sync._running = False
        return 0

    sync.sync_once = counting_sync
    await sync._sync_loop()
    assert call_count >= 2


@pytest.mark.asyncio
async def test_sync_loop_handles_exceptions() -> None:
    """_sync_loop should catch and continue on exceptions."""
    from mousedroid.cloud.firestore_sync import CloudFirestoreSync

    cfg = _make_gcp_cfg(firestore=GCPFirestoreConfig(sync_interval_s=0.01))
    episodic = _make_mock_episodic()
    sync = CloudFirestoreSync(cfg, episodic)
    sync._running = True

    call_count = 0

    async def failing_sync() -> int:
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            sync._running = False
            return 0
        raise RuntimeError("test error")

    sync.sync_once = failing_sync
    await sync._sync_loop()
    assert call_count >= 2


@pytest.mark.asyncio
async def test_close_handles_db_close_error() -> None:
    """close() should handle errors from db.close() gracefully."""
    from mousedroid.cloud.firestore_sync import CloudFirestoreSync

    cfg = _make_gcp_cfg()
    episodic = _make_mock_episodic()
    sync = CloudFirestoreSync(cfg, episodic)
    mock_db = MagicMock()
    mock_db.close.side_effect = RuntimeError("close error")
    sync._db = mock_db
    await sync.close()
    assert sync._db is None


def test_uses_config_interval() -> None:
    """CloudFirestoreSync should use sync_interval_s from config."""
    from mousedroid.cloud.firestore_sync import CloudFirestoreSync

    cfg = _make_gcp_cfg(firestore=GCPFirestoreConfig(sync_interval_s=60.0))
    episodic = _make_mock_episodic()
    sync = CloudFirestoreSync(cfg, episodic)
    assert sync._interval_s == 60.0
