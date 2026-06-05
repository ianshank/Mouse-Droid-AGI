"""Shared LLM-gateway telemetry helpers.

Centralises the two operations every backend (``llama_cpp``, ``anthropic``,
``openai_compatible``) repeats for each request: pulling token counts out of a
provider ``usage`` block, and writing the per-round-trip Prometheus metrics
(latency histogram + budget-exceeded counter + token counters). Keeping them
here means the defensive parsing and the metric family / label conventions live
in one tested place instead of being re-implemented — and drifting — per
backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mousedroid.telemetry.metrics import MetricsRegistry


def extract_token_pair(
    usage: Any,
    *,
    input_key: str,
    output_key: str,
) -> tuple[int | None, int | None]:
    """Read ``(input_tokens, output_tokens)`` from a provider ``usage`` object.

    Defensive against every shape the backends encounter: a missing/``None``
    container, attribute access (live SDK objects) vs item access (mock dicts
    and JSON bodies), and non-integer fields. Any of those yields ``None`` for
    that slot so :meth:`MetricsRegistry.inc_llm_tokens` no-ops rather than
    fabricating a count.

    The field names differ per provider, so they are parameters: Anthropic
    reports ``input_tokens`` / ``output_tokens``; OpenAI-compatible servers and
    llama-cpp report ``prompt_tokens`` / ``completion_tokens``.

    Args:
        usage: The provider usage container (object, dict, or ``None``).
        input_key: Field name carrying the prompt/input token count.
        output_key: Field name carrying the completion/output token count.

    Returns:
        ``(input_tokens, output_tokens)`` — each ``int`` when present and
        integral, else ``None``.
    """
    if usage is None:
        return (None, None)

    def _field(name: str) -> int | None:
        raw = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
        return raw if isinstance(raw, int) else None

    return (_field(input_key), _field(output_key))


def record_round_trip_metrics(
    metrics: MetricsRegistry | None,
    *,
    model: str,
    elapsed_ms: float,
    over_budget: bool,
    input_tokens: int | None,
    output_tokens: int | None,
) -> None:
    """Record one gateway round-trip's latency, budget breach, and token usage.

    No-op when ``metrics`` is ``None`` so a gateway built without a registry
    behaves byte-identically to the pre-telemetry path. Used by every backend
    so the metric families and label conventions never diverge.

    Args:
        metrics: Shared registry, or ``None`` to disable recording.
        model: Prometheus ``model`` label (Claude id, Ollama tag, or GGUF name).
        elapsed_ms: Wall-clock round-trip latency in milliseconds.
        over_budget: Whether ``elapsed_ms`` exceeded the configured target.
        input_tokens: Prompt/input token count, or ``None`` if unreported.
        output_tokens: Completion/output token count, or ``None`` if unreported.
    """
    if metrics is None:
        return
    metrics.observe_llm_gateway_latency_ms(elapsed_ms)
    if over_budget:
        metrics.inc_llm_latency_budget_exceeded(model)
    metrics.inc_llm_tokens(model, "input", input_tokens)
    metrics.inc_llm_tokens(model, "output", output_tokens)


__all__ = ["extract_token_pair", "record_round_trip_metrics"]
