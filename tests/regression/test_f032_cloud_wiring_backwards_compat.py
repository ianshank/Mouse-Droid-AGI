"""Backwards-compat pins for F-032 — adding three builders changes nothing by default.

``config/default.yaml`` and every shipped overlay except
``config/gcp_digital_twin.yaml`` declare no ``gcp:`` block at all, so the
three new builders (and the new ``cloud_logging_sink`` threading through
``main.py``) must be provably inert for them — the same shape of proof
``test_gcp_egress_defaults_backwards_compat.py`` already applies to F-029.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from mousedroid.config.schema.root import Settings
from mousedroid.factory import (
    build_cloud_firestore_sync,
    build_cloud_logging_sink,
    build_cloud_metrics_exporter,
)


def test_absent_gcp_block_yields_none_for_all_three_new_builders() -> None:
    """No gcp: block means no GCP config at all — the pre-existing baseline."""
    settings = Settings(mock_hardware=True)
    assert settings.gcp is None
    assert build_cloud_logging_sink(settings) is None
    assert build_cloud_metrics_exporter(settings, metrics_registry=object()) is None
    assert build_cloud_firestore_sync(settings, episodic=object()) is None


async def test_run_without_a_cloud_logging_sink_argument_is_unaffected() -> None:
    """A caller that omits the new kwarg entirely still gets pre-F-032 behaviour.

    ``cloud_logging_sink`` defaults to ``None`` on both ``_run``/
    ``_health_check`` specifically so that calling them exactly as they were
    called before this feature landed keeps working — this pin calls ``_run``
    with only the pre-existing ``cfg`` positional argument.
    """
    from mousedroid.main import _run

    class _FakeOrchestrator:
        def __init__(self) -> None:
            self.start = AsyncMock()
            self.run = AsyncMock()
            self.stop = AsyncMock()

    fake = _FakeOrchestrator()
    with (
        patch("mousedroid.orchestrator.orchestrator.MouseDroidOrchestrator", _FakeOrchestrator),
        patch("mousedroid.factory.build_orchestrator", return_value=fake),
    ):
        await _run(Settings(mock_hardware=True))  # no second argument

    fake.start.assert_called_once()
    fake.run.assert_called_once()
    fake.stop.assert_called_once()
