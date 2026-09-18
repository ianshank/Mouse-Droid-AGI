"""Backwards-compat pin — the reserved-key guard must not rename existing fields.

``safe_log_extra`` renames only keys that collide with something owned.  Every
other caller key passes through untouched, which is what keeps existing
Grafana panels and alert rules keyed on ``elapsed_s``, ``tick``, ``h_norm``,
``strategy`` and friends working after the guard landed.

The payloads below are transcribed from the live ``FailureRecorder.record``
call sites in ``src/`` — if a future change starts prefixing them, dashboards
break silently and this pin fails loudly first.
"""

from __future__ import annotations

import inspect

import pytest
import structlog.testing

from mousedroid.config.schema import MetricsConfig
from mousedroid.logging.setup import EXTRA_KEY_PREFIX, safe_log_extra
from mousedroid.telemetry.failure_recorder import (
    FailureRecorder,
    NullFailureRecorder,
    PrometheusFailureRecorder,
)
from mousedroid.telemetry.metrics import MetricsRegistry

# (call site, extra payload) for every production record() call that passes one.
_PRODUCTION_PAYLOADS = [
    ("orchestrator/_action_mixin.py::vla_exception", {"error": "RuntimeError"}),
    ("orchestrator/_action_mixin.py::vla_timeout", {"elapsed_s": 0.04, "budget_s": 0.033}),
    ("orchestrator/_action_mixin.py::vla_wrong_shape", {"expected": "(3,)", "got": "(1, 3)"}),
    ("orchestrator/_action_mixin.py::cognitive_core_exception", {"error": "ValueError"}),
    ("orchestrator/_world_model_state_mixin.py::latent_nan", {"tick": 42}),
    ("orchestrator/_world_model_state_mixin.py::latent_saturated", {"h_norm": 12.5}),
    ("telemetry/server/_lifecycle.py::bind_failed", {"strategy": "fixed", "host": "127.0.0.1"}),
    ("telemetry/server/_lifecycle.py::bind_exhausted", {"port_start": 9090, "attempts": 10}),
    ("voice/rocky.py::cooldown", {"cooldown_s": 2.0, "elapsed_s": 0.01}),
    ("voice/rocky.py::queue_full", {"queue_full": 1}),
]


@pytest.mark.parametrize(("site", "payload"), _PRODUCTION_PAYLOADS, ids=lambda v: str(v)[:60])
def test_production_payload_keys_are_unchanged(site: str, payload: dict[str, object]) -> None:
    """Existing call-site field names survive the guard verbatim."""
    result = safe_log_extra(payload, occupied=("subsystem", "reason", "log_level"))

    assert result == payload, f"{site} payload was rewritten by the guard"


@pytest.mark.parametrize(("site", "payload"), _PRODUCTION_PAYLOADS, ids=lambda v: str(v)[:60])
def test_production_payloads_gain_no_prefixed_keys(site: str, payload: dict[str, object]) -> None:
    """No spurious ``extra_``-prefixed field appears for well-behaved callers."""
    result = safe_log_extra(payload, occupied=("subsystem", "reason", "log_level"))

    assert not [k for k in result if k.startswith(EXTRA_KEY_PREFIX)], site


class TestRecorderLogShapeUnchanged:
    """The emitted log event keeps its established field names."""

    def test_event_name_unchanged(self) -> None:
        """The structlog event name is still subsystem_failure_recorded."""
        rec = PrometheusFailureRecorder(MetricsRegistry(MetricsConfig()))

        with structlog.testing.capture_logs() as logs:
            rec.record("voice", "piper_timeout")

        assert logs[0]["event"] == "subsystem_failure_recorded"

    def test_owned_field_names_unchanged(self) -> None:
        """subsystem / reason / log_level remain the recorder's field names."""
        rec = PrometheusFailureRecorder(MetricsRegistry(MetricsConfig()))

        with structlog.testing.capture_logs() as logs:
            rec.record("telemetry", "bind_failed", level="error")

        log = logs[0]
        assert log["subsystem"] == "telemetry"
        assert log["reason"] == "bind_failed"
        assert log["log_level"] == "error"

    def test_extra_none_still_supported(self) -> None:
        """extra=None remains a valid call."""
        rec = PrometheusFailureRecorder(MetricsRegistry(MetricsConfig()))

        with structlog.testing.capture_logs() as logs:
            rec.record("voice", "retry_exhausted", extra=None)

        assert logs[0]["event"] == "subsystem_failure_recorded"


class TestProtocolSignatureUnchanged:
    """The FailureRecorder protocol surface did not shift."""

    def test_record_parameter_names(self) -> None:
        """record() still takes (subsystem, reason, *, level, extra)."""
        params = list(inspect.signature(PrometheusFailureRecorder.record).parameters)

        assert params == ["self", "subsystem", "reason", "level", "extra"]

    def test_level_and_extra_remain_keyword_only(self) -> None:
        """level and extra stay keyword-only with their defaults."""
        sig = inspect.signature(PrometheusFailureRecorder.record)

        assert sig.parameters["level"].kind is inspect.Parameter.KEYWORD_ONLY
        assert sig.parameters["level"].default == "warning"
        assert sig.parameters["extra"].kind is inspect.Parameter.KEYWORD_ONLY
        assert sig.parameters["extra"].default is None

    def test_both_implementations_still_satisfy_the_protocol(self) -> None:
        """Guard did not break structural conformance."""
        prometheus = PrometheusFailureRecorder(MetricsRegistry(MetricsConfig()))

        assert isinstance(prometheus, FailureRecorder)
        assert isinstance(NullFailureRecorder(), FailureRecorder)
