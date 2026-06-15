"""AQA: schema-field hygiene + counter cardinality for Phase-6 WS1.

Architectural-quality assertions locking the WS1 contracts a future refactor
could silently break: config field descriptions/defaults, the registry helper
name (rename guard), the fixed low-cardinality ``reason`` set, the
``track_on_device_learning`` gate flag, and the byte-identical ``/metrics``
render when the family is unused.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mousedroid.config.schema import MetricsConfig, OnDeviceLearningConfig, Settings
from mousedroid.telemetry import metrics as metrics_mod
from mousedroid.telemetry.metrics import MetricsRegistry


def test_on_device_learning_field_is_optional_default_none() -> None:
    field = Settings.model_fields["on_device_learning"]
    assert field.default is None
    assert field.description, "on_device_learning must carry an operator description"


def test_config_fields_all_documented() -> None:
    for name, field in OnDeviceLearningConfig.model_fields.items():
        assert field.description, f"{name} must carry an operator description"


def test_ws_e0_enablement_fields_present_typed_and_documented() -> None:
    """The WS-E0 enablement fields exist, are typed/defaulted, and documented."""
    fields = OnDeviceLearningConfig.model_fields
    for name, expected_default in (
        ("enable_hot_swap", False),
        ("refine_sequence_length", 16),
        ("refine_batch_episodes", 4),
    ):
        assert name in fields, f"WS-E0 field {name} must be present"
        assert fields[name].default == expected_default
        assert fields[name].description, f"{name} must carry an operator description"


def test_ws_e0_model_validator_enforced() -> None:
    """The cross-field validator rejects hot-swap without the master switch."""
    with pytest.raises(ValidationError):
        OnDeviceLearningConfig(enable_hot_swap=True, enabled=False)
    # ... and accepts it once enabled.
    assert OnDeviceLearningConfig(enable_hot_swap=True, enabled=True).enable_hot_swap is True


def test_metrics_gate_flag_documented_and_default_on() -> None:
    field = MetricsConfig.model_fields["track_on_device_learning"]
    assert field.default is True
    assert field.description


def test_registry_helper_exists_rename_guard() -> None:
    assert callable(getattr(MetricsRegistry, "inc_on_device_learning_reverted", None))


def test_reason_label_value_set_fixed() -> None:
    """Render reflects only the fixed low-cardinality reason enum."""
    reg = MetricsRegistry(MetricsConfig())
    for reason in ("regression_bound", "integrity_mismatch", "exception"):
        reg.inc_on_device_learning_reverted(reason)
    out = reg.render_prometheus()
    assert out.count("on_device_learning_reverted_total{") == 3


def test_reason_constant_set_pinned() -> None:
    assert (
        frozenset({"regression_bound", "integrity_mismatch", "exception"})
        == metrics_mod._ON_DEVICE_REVERT_REASONS
    )


def test_metrics_byte_identical_when_family_unused() -> None:
    """A registry with no revert write renders nothing for the family."""
    out = MetricsRegistry(MetricsConfig()).render_prometheus()
    assert "on_device_learning_reverted_total" not in out
