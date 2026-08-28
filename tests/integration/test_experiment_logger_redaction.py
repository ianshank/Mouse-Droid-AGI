"""Integration: no tracking-URI credential reaches a log event (F-034).

Why this tier and not unit: ``tests/unit/logging/test_redaction.py`` proves
the helper masks correctly, but it would stay green if nobody ever called it.
What actually protects an operator is that ``build_experiment_logger`` --
reached through the factory, from a real ``Settings`` -- redacts before
logging. Per ``.claude/skills/test-tier-mirror/SKILL.md``: "Does it go
through ``factory.py``? If yes it is at least integration."

Background: ``ExperimentLoggerConfig.tracking_uri`` is a plain ``str``, not
a ``SecretStr``, and a remote store is legitimately spelled
``http://user:password@host:5000`` (the field's own description advertises
``http://host:port``). There is no redaction processor in the structlog
chain -- ``mousedroid.logging.setup`` builds the processor list and none of
its entries scrub values -- so whatever a log call is handed is emitted
verbatim to stdout, to the telemetry ring buffer, and, where enabled, to
Cloud Logging. Redaction therefore has to happen at the call site, and this
pins that it does.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import structlog

from mousedroid.config.schema import (
    ExperimentLoggerConfig,
    ObservabilityConfig,
    Settings,
)
from mousedroid.factory import build_experiment_logger

# Distinctive enough that a substring search cannot collide with anything
# structlog itself emits.
_SECRET = "sup3rS3cretTrackingPassw0rd"  # noqa: S105 - test fixture, not a credential
_CREDENTIALED_URI = f"http://mlflow-user:{_SECRET}@mlflow.internal.example:5000"


@pytest.fixture
def captured_events() -> Any:
    """Capture structlog events without touching the global config.

    ``structlog.testing.capture_logs`` swaps the processor chain for the
    duration of the block and restores it afterwards, so this cannot leak
    configuration into sibling tests -- which matters here because the
    module under test is reached through the factory.
    """
    return structlog.testing.capture_logs


def _settings_with_uri(uri: str) -> Settings:
    return Settings(mock_hardware=True).model_copy(
        update={
            "observability": ObservabilityConfig(
                experiment_logger=ExperimentLoggerConfig(
                    backend="mlflow",
                    tracking_uri=uri,
                    experiment_name="redaction-test",
                ),
            ),
        }
    )


def _all_text(events: list[dict[str, Any]]) -> str:
    """Flatten every key and value of every captured event into one string."""
    return " ".join(f"{k}={v!r}" for event in events for k, v in event.items())


@pytest.fixture
def offline_mlflow(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralise ``MlflowExperimentLogger`` so no test here touches a socket.

    Without this, a credentialed ``http://`` tracking URI sends the real
    client into DNS resolution and retry backoff -- minutes per test, and
    dependent on the sandbox having no network. The URI still travels the
    full factory path (resolution, the pre-construction log event, the
    branch selection); only the client construction is stubbed.

    A no-op when mlflow is absent: the factory then takes its
    ``ImportError`` branch, which logs the URI too and must equally redact.
    """
    try:
        from mousedroid.training.observability import mlflow_logger
    except ImportError:  # pragma: no cover - depends on the [mlflow] extra
        return

    class _StubLogger:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    monkeypatch.setattr(mlflow_logger, "MlflowExperimentLogger", _StubLogger)


def test_no_log_event_carries_the_tracking_uri_password(
    captured_events: Any, offline_mlflow: None
) -> None:
    """The password must appear in no captured event, on any code path.

    Deliberately asserts the security property over the *whole* event stream
    rather than checking one field of one event: the factory logs the URI
    from three different places (resolved, extras-missing, init-failed) and
    a future fourth must not be able to reintroduce the leak unnoticed.

    This drives the real ``build_experiment_logger``. Whether mlflow is
    installed decides which branch runs -- resolved-then-constructed, or
    extras-missing -- and the assertion holds either way, which is the
    point: every branch redacts.
    """
    with captured_events() as events:
        build_experiment_logger(_settings_with_uri(_CREDENTIALED_URI))

    assert events, "no log events captured -- the pin would be vacuous"
    haystack = _all_text(events)
    assert _SECRET not in haystack, f"tracking_uri password leaked into a log event: {haystack}"
    # The host must survive, or redaction has destroyed the diagnostic value
    # the event exists for and a weaker fix (dropping the field) would pass.
    assert "mlflow.internal.example" in haystack
    assert "***" in haystack


def test_init_failure_path_also_redacts(
    captured_events: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The failure paths are the ones an operator pastes into a ticket.

    Before redaction these were the worst case: the URI was logged *only*
    on failure (it was previously absent from these two events entirely),
    which is exactly when a wrong-but-valid-elsewhere credential is most
    likely to be copied somewhere public.

    The failure is injected rather than provoked with an unreachable host:
    a real bad endpoint makes mlflow retry with backoff, which turned this
    into a multi-minute network-dependent test. Patching the constructor
    reaches the same ``except Exception`` branch in one step, offline and
    deterministically.
    """
    mlflow_logger = pytest.importorskip("mousedroid.training.observability.mlflow_logger")

    def _explode(**_kwargs: Any) -> None:
        raise RuntimeError("unable to open database file")

    monkeypatch.setattr(mlflow_logger, "MlflowExperimentLogger", _explode)

    with captured_events() as events:
        build_experiment_logger(_settings_with_uri(_CREDENTIALED_URI))

    assert any(e.get("event") == "experiment_logger_mlflow_init_failed" for e in events), (
        "the init-failed branch did not run -- this pin would be vacuous"
    )
    haystack = _all_text(events)
    assert _SECRET not in haystack, f"password leaked on a failure path: {haystack}"


def test_the_local_default_is_not_mangled_by_redaction(
    captured_events: Any,
    offline_mlflow: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Redaction must be invisible for the credential-free default.

    Guards the obvious over-correction: a blanket mask would "fix" the leak
    while destroying the resolved-path reporting that
    ``experiment_logger_tracking_uri_resolved`` was added to provide.

    ``chdir`` into ``tmp_path`` first, because the default is CWD-relative
    and ``_resolve_tracking_uri`` pins it against wherever the process
    happens to be: without this the test writes a real SQLite database into
    the repository root and leaves it there to grow across runs.
    ``monkeypatch.chdir`` rather than ``os.chdir`` + try/finally so pytest
    restores it during teardown even if the body raises.
    """
    monkeypatch.chdir(tmp_path)
    with captured_events() as events:
        build_experiment_logger(_settings_with_uri("sqlite:///mlflow.db"))

    haystack = _all_text(events)
    assert "mlflow.db" in haystack, "the resolved local path stopped being reportable"
    assert "***" not in haystack, "a credential-free URI must not be masked"
