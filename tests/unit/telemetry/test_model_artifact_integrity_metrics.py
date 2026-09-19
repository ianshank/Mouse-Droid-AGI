"""``mousedroid_model_artifact_sha256_mismatches_total`` (Phase 7, task 7.1).

The boot-path sibling of ``inc_cloud_weight_update_sha256_mismatch``: every
increment is a refused artifact, so the family has to behave like the rest of
the registry — pure-add render, ``_total`` appended once at render time, and a
closed label set that a config edit cannot widen.
"""

from __future__ import annotations

from mousedroid.config.schema import MetricsConfig
from mousedroid.telemetry.metrics import _MODEL_ARTIFACT_KINDS, generate_metrics_sample
from mousedroid.telemetry.metrics.registry import MetricsRegistry

_FAMILY = "model_artifact_sha256_mismatches"


def _registry() -> MetricsRegistry:
    return MetricsRegistry(MetricsConfig.model_validate({}))


class TestRenderShape:
    def test_absent_until_the_first_refusal(self) -> None:
        """Pure-add: a healthy rover's exposition output is unchanged."""
        assert _FAMILY not in _registry().render_prometheus()

    def test_rendered_once_a_refusal_lands(self) -> None:
        registry = _registry()
        registry.inc_model_artifact_sha256_mismatch("world_model_onnx")
        ns = MetricsConfig().namespace
        rendered = registry.render_prometheus()
        assert f'{ns}_{_FAMILY}_total{{artifact="world_model_onnx"}} 1' in rendered

    def test_the_total_suffix_is_appended_exactly_once(self) -> None:
        registry = _registry()
        registry.inc_model_artifact_sha256_mismatch("bdi_weights")
        rendered = registry.render_prometheus()
        assert f"{_FAMILY}_total_total" not in rendered

    def test_the_registry_field_name_carries_no_total_suffix(self) -> None:
        """``primitives.py`` appends ``_total`` at render; the field must not."""
        ns = MetricsConfig().namespace
        registry = _registry()
        assert registry._name_model_artifact_sha256_mismatches == f"{ns}_{_FAMILY}"

    def test_declares_a_counter_type_and_help_text(self) -> None:
        registry = _registry()
        registry.inc_model_artifact_sha256_mismatch("world_model_onnx")
        rendered = registry.render_prometheus()
        assert f"# TYPE {MetricsConfig().namespace}_{_FAMILY}_total counter" in rendered
        assert f"# HELP {MetricsConfig().namespace}_{_FAMILY}_total " in rendered


class TestWriterGuards:
    def test_accumulates_per_artifact_kind(self) -> None:
        registry = _registry()
        registry.inc_model_artifact_sha256_mismatch("world_model_onnx")
        registry.inc_model_artifact_sha256_mismatch("world_model_onnx")
        registry.inc_model_artifact_sha256_mismatch("bdi_weights")
        rendered = registry.render_prometheus()
        assert f'{_FAMILY}_total{{artifact="world_model_onnx"}} 2' in rendered
        assert f'{_FAMILY}_total{{artifact="bdi_weights"}} 1' in rendered

    def test_a_non_positive_amount_is_a_no_op(self) -> None:
        registry = _registry()
        registry.inc_model_artifact_sha256_mismatch("world_model_onnx", 0)
        registry.inc_model_artifact_sha256_mismatch("world_model_onnx", -3)
        assert _FAMILY not in registry.render_prometheus()

    def test_an_out_of_set_label_value_is_dropped(self) -> None:
        """A free-text string must never open a new time series."""
        registry = _registry()
        registry.inc_model_artifact_sha256_mismatch("mission: go to the hangar")  # type: ignore[arg-type]
        assert _FAMILY not in registry.render_prometheus()

    def test_the_label_set_is_closed_at_two_values(self) -> None:
        assert frozenset({"world_model_onnx", "bdi_weights"}) == _MODEL_ARTIFACT_KINDS


class TestSampleSeeding:
    def test_generate_metrics_sample_seeds_both_label_values(self) -> None:
        """Required: alert / dashboard tests resolve names against the sample."""
        sample = generate_metrics_sample()
        ns = MetricsConfig().namespace
        assert f'{ns}_{_FAMILY}_total{{artifact="world_model_onnx"}}' in sample
        assert f'{ns}_{_FAMILY}_total{{artifact="bdi_weights"}}' in sample
