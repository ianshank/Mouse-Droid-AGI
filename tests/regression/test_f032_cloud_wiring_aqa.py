"""AQA pins for F-032 — GCP observability wiring stays structurally intact.

Complements the unit-tier coverage in
``tests/unit/factory/test_factory_cloud_observability.py`` and
``tests/unit/orchestrator/test_cloud_subsystems_wiring.py``, which prove
*behaviour*. These pins prove *shape* — the kind of drift a refactor could
introduce without any behavioural test going red (e.g. renaming a
constructor parameter that nothing else in the tree references yet).
"""

from __future__ import annotations

import inspect


def test_cloud_firestore_sync_protocol_has_the_expected_members() -> None:
    """CloudFirestoreSyncProtocol declares exactly start/sync_once/close.

    New in F-032 — no prior pin exists for this protocol's shape.
    """
    from mousedroid.cloud.protocol import CloudFirestoreSyncProtocol

    members = {
        name
        for name in vars(CloudFirestoreSyncProtocol)
        if not name.startswith("_") or name == "__call__"
    }
    assert members == {"start", "sync_once", "close"}


def test_cloud_logging_sink_protocol_was_widened_to_include_lifecycle() -> None:
    """CloudLoggingSinkProtocol covers start/__call__/close, not just __call__.

    Widened during F-032 so main.py can drive the sink's lifecycle directly
    (design.md D-5 in the F-032 openspec bundle) — main.py's own type
    annotations depend on this shape under mypy --strict.
    """
    from mousedroid.cloud.protocol import CloudLoggingSinkProtocol

    members = {
        name
        for name in vars(CloudLoggingSinkProtocol)
        if not name.startswith("_") or name == "__call__"
    }
    assert members == {"start", "__call__", "close"}


def test_orchestrator_accepts_the_two_new_cloud_collaborators() -> None:
    """MouseDroidOrchestrator.__init__ has cloud_metrics_exporter/cloud_firestore_sync.

    A signature-level pin, not a behavioural one — the behavioural contract
    (start/stop wiring, LIFO order) is covered in
    tests/unit/orchestrator/test_cloud_subsystems_wiring.py.
    """
    from mousedroid.orchestrator.orchestrator import MouseDroidOrchestrator

    params = inspect.signature(MouseDroidOrchestrator.__init__).parameters
    assert "cloud_metrics_exporter" in params
    assert "cloud_firestore_sync" in params
    assert params["cloud_metrics_exporter"].default is None
    assert params["cloud_firestore_sync"].default is None


def test_main_entry_points_accept_cloud_logging_sink() -> None:
    """main.py's _run/_health_check both carry an optional cloud_logging_sink param.

    Pinned at the signature level because main.py is ``# pragma: no cover``
    and excluded from the coverage gate (``pyproject.toml``'s ``omit``) — a
    behavioural regression here would not show up in the coverage report,
    only in tests/unit/test_main.py or this pin.
    """
    from mousedroid.main import _health_check, _run

    for fn in (_run, _health_check):
        params = inspect.signature(fn).parameters
        assert "cloud_logging_sink" in params, f"{fn.__name__} lost its cloud_logging_sink param"
        assert params["cloud_logging_sink"].default is None
