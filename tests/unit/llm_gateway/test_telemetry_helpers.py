"""Unit tests for the shared LLM-gateway telemetry helpers.

``extract_token_pair`` + ``record_round_trip_metrics`` are the single place all
three backends pull token counts and write per-round-trip metrics, so they are
covered directly here (in addition to the per-backend integration coverage).
"""

from __future__ import annotations

from types import SimpleNamespace

from mousedroid.config.schema import MetricsConfig
from mousedroid.llm_gateway._telemetry import extract_token_pair, record_round_trip_metrics
from mousedroid.telemetry.metrics import MetricsRegistry


# --------------------------------------------------------------------------- #
# extract_token_pair
# --------------------------------------------------------------------------- #
def test_extract_token_pair_none_usage() -> None:
    assert extract_token_pair(None, input_key="input_tokens", output_key="output_tokens") == (
        None,
        None,
    )


def test_extract_token_pair_dict_usage() -> None:
    usage = {"prompt_tokens": 12, "completion_tokens": 4}
    assert extract_token_pair(usage, input_key="prompt_tokens", output_key="completion_tokens") == (
        12,
        4,
    )


def test_extract_token_pair_object_usage() -> None:
    usage = SimpleNamespace(input_tokens=9, output_tokens=3)
    assert extract_token_pair(usage, input_key="input_tokens", output_key="output_tokens") == (9, 3)


def test_extract_token_pair_missing_key_is_none() -> None:
    usage = {"prompt_tokens": 7}  # completion_tokens absent
    assert extract_token_pair(usage, input_key="prompt_tokens", output_key="completion_tokens") == (
        7,
        None,
    )


def test_extract_token_pair_non_integer_field_is_none() -> None:
    usage = {"prompt_tokens": "lots", "completion_tokens": 2.5}  # non-int → dropped
    assert extract_token_pair(usage, input_key="prompt_tokens", output_key="completion_tokens") == (
        None,
        None,
    )


def test_extract_token_pair_key_names_are_parameterised() -> None:
    """The same helper serves Anthropic and OpenAI/llama key conventions."""
    usage = SimpleNamespace(input_tokens=1, output_tokens=2, prompt_tokens=3, completion_tokens=4)
    assert extract_token_pair(usage, input_key="input_tokens", output_key="output_tokens") == (1, 2)
    assert extract_token_pair(usage, input_key="prompt_tokens", output_key="completion_tokens") == (
        3,
        4,
    )


# --------------------------------------------------------------------------- #
# record_round_trip_metrics
# --------------------------------------------------------------------------- #
def _registry() -> MetricsRegistry:
    return MetricsRegistry(MetricsConfig())


def test_record_round_trip_metrics_none_registry_is_noop() -> None:
    # Must not raise when no registry is wired.
    record_round_trip_metrics(
        None, model="m", elapsed_ms=12.0, over_budget=True, input_tokens=5, output_tokens=5
    )


def test_record_round_trip_metrics_records_latency_and_tokens() -> None:
    reg = _registry()
    record_round_trip_metrics(
        reg,
        model="claude-haiku-4-5",
        elapsed_ms=42.0,
        over_budget=False,
        input_tokens=11,
        output_tokens=6,
    )
    out = reg.render_prometheus()
    assert "llm_gateway_latency_ms_count 1" in out
    assert 'model="claude-haiku-4-5",token_type="input"} 11' in out
    assert 'model="claude-haiku-4-5",token_type="output"} 6' in out
    assert "llm_latency_budget_exceeded" not in out  # over_budget=False


def test_record_round_trip_metrics_budget_counter_on_over_budget() -> None:
    reg = _registry()
    record_round_trip_metrics(
        reg, model="m", elapsed_ms=9000.0, over_budget=True, input_tokens=None, output_tokens=None
    )
    out = reg.render_prometheus()
    assert 'llm_latency_budget_exceeded_total{model="m"} 1' in out
    # None token counts must not fabricate a token series.
    assert "_llm_tokens_total" not in out
