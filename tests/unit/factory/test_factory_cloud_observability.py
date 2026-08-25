"""Unit tests for the F-032 cloud observability builders.

Covers ``build_cloud_logging_sink``, ``build_cloud_metrics_exporter``, and
``build_cloud_firestore_sync`` — the disabled-by-default path, the two
null-collaborator guards (``metrics_registry``/``episodic`` can each
independently be ``None`` even when their ``gcp.*.enabled`` flag is
``True``), and the simulated-``ImportError`` degrade path.

Construction of the three concrete classes is plain attribute assignment —
none imports ``google.cloud.*`` until ``.start()`` runs (confirmed by
reading ``cloud/logging_sink.py``, ``cloud/monitoring_exporter.py``,
``cloud/firestore_sync.py`` directly) — so the "enabled" happy-path tests
below need no SDK mocking at all. The ``except ImportError`` branch in every
builder guards the lightweight ``mousedroid.cloud.<x>`` wrapper module
import, not the heavy SDK import, and is not exercised by ordinary CI (no
job installs the ``[gcp]`` extra) — proven here via
``monkeypatch.setitem(sys.modules, "mousedroid.cloud.<x>", None)``, which
forces the next import of that exact module name to raise ``ImportError``.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

from mousedroid.cloud.firestore_sync import CloudFirestoreSync
from mousedroid.cloud.logging_sink import CloudLoggingSink
from mousedroid.cloud.monitoring_exporter import CloudMetricsExporter
from mousedroid.config.schema.root import Settings
from mousedroid.factory import (
    build_cloud_firestore_sync,
    build_cloud_logging_sink,
    build_cloud_metrics_exporter,
)


def _settings(**gcp_overrides: object) -> Settings:
    """Settings with a minimal GCP block, matching test_gcp_egress_defaults_aqa.py."""
    return Settings(mock_hardware=True, gcp={"project_id": "test-project", **gcp_overrides})


# ---------------------------------------------------------------------------
# build_cloud_logging_sink
# ---------------------------------------------------------------------------


def test_logging_sink_none_when_gcp_not_configured() -> None:
    settings = Settings(mock_hardware=True)
    assert settings.gcp is None
    assert build_cloud_logging_sink(settings) is None


def test_logging_sink_none_when_disabled_by_default() -> None:
    settings = _settings()
    assert settings.gcp is not None
    assert settings.gcp.logging.enabled is False
    assert build_cloud_logging_sink(settings) is None


def test_logging_sink_built_when_enabled() -> None:
    settings = _settings(logging={"enabled": True})
    sink = build_cloud_logging_sink(settings)
    assert isinstance(sink, CloudLoggingSink)


def test_logging_sink_none_when_module_not_importable() -> None:
    settings = _settings(logging={"enabled": True})
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(sys.modules, "mousedroid.cloud.logging_sink", None)
        assert build_cloud_logging_sink(settings) is None


# ---------------------------------------------------------------------------
# build_cloud_metrics_exporter
# ---------------------------------------------------------------------------


def test_metrics_exporter_none_when_gcp_not_configured() -> None:
    settings = Settings(mock_hardware=True)
    assert build_cloud_metrics_exporter(settings, metrics_registry=MagicMock()) is None


def test_metrics_exporter_none_when_disabled_by_default() -> None:
    settings = _settings()
    assert settings.gcp is not None
    assert settings.gcp.monitoring.enabled is False
    assert build_cloud_metrics_exporter(settings, metrics_registry=MagicMock()) is None


def test_metrics_exporter_none_when_registry_missing() -> None:
    """gcp.monitoring.enabled=True + metrics_registry=None must degrade, not crash.

    ``CloudMetricsExporter.__init__``'s ``registry`` param is non-Optional --
    without this guard the orchestrator would crash on an unvalidated but
    legal config combination (metrics.enabled=False + monitoring.enabled=True).
    """
    settings = _settings(monitoring={"enabled": True})
    assert build_cloud_metrics_exporter(settings, metrics_registry=None) is None


def test_metrics_exporter_built_when_enabled_and_registry_present() -> None:
    settings = _settings(monitoring={"enabled": True})
    exporter = build_cloud_metrics_exporter(settings, metrics_registry=MagicMock())
    assert isinstance(exporter, CloudMetricsExporter)


def test_metrics_exporter_none_when_module_not_importable() -> None:
    settings = _settings(monitoring={"enabled": True})
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(sys.modules, "mousedroid.cloud.monitoring_exporter", None)
        assert build_cloud_metrics_exporter(settings, metrics_registry=MagicMock()) is None


# ---------------------------------------------------------------------------
# build_cloud_firestore_sync
# ---------------------------------------------------------------------------


def test_firestore_sync_none_when_gcp_not_configured() -> None:
    settings = Settings(mock_hardware=True)
    assert build_cloud_firestore_sync(settings, episodic=MagicMock()) is None


def test_firestore_sync_none_when_disabled_by_default() -> None:
    settings = _settings()
    assert settings.gcp is not None
    assert settings.gcp.firestore.enabled is False
    assert build_cloud_firestore_sync(settings, episodic=MagicMock()) is None


def test_firestore_sync_none_when_episodic_missing() -> None:
    """gcp.firestore.enabled=True + episodic=None must degrade, not crash.

    This is the default-``memory.enabled=False`` scenario in practice --
    ``CloudFirestoreSync.__init__``'s ``episodic`` param is non-Optional, and
    ``build_memory_tier`` returns ``None`` whenever ``memory.enabled`` is
    ``False``, the schema default -- independent of the GCP toggle.
    """
    settings = _settings(firestore={"enabled": True})
    assert build_cloud_firestore_sync(settings, episodic=None) is None


def test_firestore_sync_built_when_enabled_and_episodic_present() -> None:
    settings = _settings(firestore={"enabled": True})
    sync = build_cloud_firestore_sync(settings, episodic=MagicMock())
    assert isinstance(sync, CloudFirestoreSync)


def test_firestore_sync_none_when_module_not_importable() -> None:
    settings = _settings(firestore={"enabled": True})
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(sys.modules, "mousedroid.cloud.firestore_sync", None)
        assert build_cloud_firestore_sync(settings, episodic=MagicMock()) is None
