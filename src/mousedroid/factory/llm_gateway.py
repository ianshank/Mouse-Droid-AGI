"""Factory builders — LLM backend selection and prompt-injection filtering."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mousedroid.llm_gateway.protocol import LLMGatewayProtocol
from mousedroid.logging.setup import get_logger
from mousedroid.security.injection_filter import PromptInjectionFilterProtocol, RegexInjectionFilter

if TYPE_CHECKING:
    from mousedroid.config.schema import (
        LLMConfig,
        Settings,
    )
    from mousedroid.telemetry.metrics import MetricsRegistry

_log = get_logger(__name__)


def build_injection_filter(cfg: Settings) -> PromptInjectionFilterProtocol:
    """Build the shared :class:`PromptInjectionFilterProtocol` instance.

    Combines patterns from ``cfg.llm.injection_patterns`` (the historical
    source) so existing YAML/env behaviour is preserved. The same filter
    is later threaded into both :func:`build_llm_gateway` and the
    OpenClaw :class:`MissionDispatcher` so REST + MCP + LLM ingress all
    enforce the same envelope.

    The length cap defers to ``cfg.openclaw.max_command_len`` only when
    OpenClaw is **enabled** — so the dispatcher is the single source of
    truth for the cap on production deployments. Disabled (or absent)
    OpenClaw blocks fall back to ``cfg.llm.max_command_len`` so a YAML
    block like ``openclaw: {enabled: false, max_command_len: 128}``
    cannot silently lower the LLM gateway's length cap.
    """
    if cfg.openclaw is not None and cfg.openclaw.enabled:
        max_len = cfg.openclaw.max_command_len
    else:
        max_len = cfg.llm.max_command_len
    return RegexInjectionFilter(cfg.llm.injection_patterns, max_len=max_len)


def _build_single_llm_gateway(
    llm_cfg: LLMConfig,
    *,
    injection_filter: PromptInjectionFilterProtocol | None = None,
    metrics: MetricsRegistry | None = None,
) -> LLMGatewayProtocol:
    """Build ONE concrete gateway for ``llm_cfg.backend`` (no failover wrap).

    Extracted so :func:`build_llm_gateway` can reuse the same dispatch for
    both the primary and the optional ``fallback_backend`` secondary.

    Args:
        llm_cfg: The :class:`LLMConfig` to build from. The secondary path
            passes a ``model_copy`` with ``backend`` (and optionally
            ``model_name``) overridden.
        injection_filter: Optional shared prompt-injection filter. Applied to
            ALL three backends — ``llama_cpp`` (the GGUF model runs the
            command verbatim), ``anthropic`` (ships NL to a third-party
            cloud), and ``openai_compatible`` (the f006-remote-llm sprint
            closed the prior gap where the HTTP path silently discarded the
            filter, trusting the upstream provider's guardrails — wrong once
            the operator runbook started teaching mission text via the
            remote-LLM probe). Each backend sanitises the mission-translation
            path before egress.
        metrics: Optional shared :class:`MetricsRegistry`. Forwarded to ALL
            three backends so successful translations / queries record
            latency / token / budget metrics regardless of which backend
            ``cfg.llm.backend`` selects (the ``openai_compatible`` token
            counts come from the response ``usage`` block; the ``llama_cpp``
            counts from the llama-cpp ``usage`` block when the build reports it).

    Returns:
        A gateway conforming to :class:`LLMGatewayProtocol`.
    """
    if llm_cfg.backend == "openai_compatible":
        from mousedroid.llm_gateway.openai_compatible import OpenAICompatibleLLMGateway

        _log.info(
            "llm_gateway_built",
            backend="openai_compatible",
            base_url=llm_cfg.base_url,
            model=llm_cfg.model_name,
            enabled=llm_cfg.enabled,
        )
        return OpenAICompatibleLLMGateway(
            llm_cfg, injection_filter=injection_filter, metrics=metrics
        )

    if llm_cfg.backend == "anthropic":
        from mousedroid.llm_gateway.anthropic_gateway import AnthropicLLMGateway

        _log.info(
            "llm_gateway_built",
            backend="anthropic",
            model=llm_cfg.model_name,
            enabled=llm_cfg.enabled,
        )
        return AnthropicLLMGateway(llm_cfg, injection_filter=injection_filter, metrics=metrics)

    # Default / legacy ``llama_cpp`` path.
    from mousedroid.llm_gateway.config import GatewayConfig
    from mousedroid.llm_gateway.gateway import LLMGateway

    gateway_cfg = GatewayConfig(
        enabled=llm_cfg.enabled,
        model_path=llm_cfg.model_path,
        model_url=llm_cfg.model_url,
        model_checksum=llm_cfg.model_checksum,
        context_length=llm_cfg.context_length,
        n_threads=llm_cfg.n_threads,
        n_gpu_layers=llm_cfg.n_gpu_layers,
        n_batch=llm_cfg.n_batch,
        max_tokens=llm_cfg.max_tokens,
        temperature=llm_cfg.temperature,
        latency_target_ms=llm_cfg.latency_target_ms,
        stop_tokens=llm_cfg.stop_tokens,
        max_vx_norm_mps=llm_cfg.max_vx_norm_mps,
        max_vy_norm_mps=llm_cfg.max_vy_norm_mps,
        max_omega_norm_rads=llm_cfg.max_omega_norm_rads,
        max_command_len=llm_cfg.max_command_len,
        system_prompt=llm_cfg.system_prompt,
        query_system_prompt=llm_cfg.query_system_prompt,
        query_max_tokens=llm_cfg.query_max_tokens,
        injection_patterns=llm_cfg.injection_patterns,
    )
    _log.info("llm_gateway_built", backend="llama_cpp", enabled=llm_cfg.enabled)
    return LLMGateway(gateway_cfg, injection_filter=injection_filter, metrics=metrics)


def build_llm_gateway(
    cfg: Settings,
    *,
    injection_filter: PromptInjectionFilterProtocol | None = None,
    metrics: MetricsRegistry | None = None,
) -> LLMGatewayProtocol:
    """Build the LLM gateway selected by ``cfg.llm.backend``.

    Three backends ship (all conform to :class:`LLMGatewayProtocol`):

    * ``llama_cpp`` (default, pre-Tier-C2.3): in-process GGUF loader via
      ``llama-cpp-python``. Loads from ``cfg.llm.model_path``.
    * ``openai_compatible`` (Tier C2.3): async HTTP client talking to
      ``{cfg.llm.base_url}/v1/chat/completions``. Default targets the
      local Ollama daemon at ``http://127.0.0.1:11434``. The same
      endpoint is served by Ollama, LM Studio, OpenAI, and most
      OpenAI-compatible local-LLM tooling — operators swap deployments
      by changing only ``cfg.llm.base_url`` (and ``cfg.llm.model_name``).
    * ``anthropic`` (Tier C-rover): async Claude Messages API client for
      cloud deliberative mission translation. Reads the Claude model id
      from ``cfg.llm.model_name`` and the key from ``cfg.llm.api_key`` (or
      the ``ANTHROPIC_API_KEY`` env var).

    When ``cfg.llm.fallback_backend != "none"`` the primary is wrapped with
    the selected LOCAL secondary in a :class:`FallbackLLMGateway` composite,
    so an off-network rover transparently degrades from cloud Claude to a
    local model. Setting ``fallback_backend == backend`` is treated as a
    no-op (the composite is skipped) — falling back to the same backend
    serves no purpose.

    The ``injection_filter`` is now shared across ALL three backends
    (``llama_cpp``, ``anthropic``, AND ``openai_compatible``) and both tiers
    of the fallback composite. The f006-remote-llm sprint closed the prior
    gap where the HTTP (``openai_compatible``) backend silently discarded the
    filter ("upstream provider expected to enforce its own guardrails") — a
    documented attack surface once the operator runbook started teaching
    mission text via the new ``jetson_remote_llm_probe``. The HTTP path now
    calls ``injection_filter.sanitize(nl)`` inside ``translate_mission``,
    mirroring ``LLMGateway._sanitize_command`` at
    ``llm_gateway/gateway.py:148`` so every backend applies the same local
    rejection envelope before NL leaves the rover.

    Args:
        cfg: Root settings.
        injection_filter: Optional shared :class:`PromptInjectionFilterProtocol`.
            When ``None``, each filter-aware gateway constructs its own filter
            from ``cfg.llm.injection_patterns`` (legacy behaviour) and the
            ``openai_compatible`` gateway skips local sanitisation; when
            supplied (the default in :func:`build_orchestrator`), the same
            filter is reused by all three backends + the OpenClaw mission
            dispatcher.
        metrics: Optional shared :class:`MetricsRegistry`, forwarded to both
            tiers (every backend records latency/token/budget metrics; the
            composite additionally records the per-tier served counter).
            ``None`` (default) is a no-op — the gateway behaves byte-identically.

    Returns:
        LLM gateway conforming to :class:`LLMGatewayProtocol`.
    """
    primary = _build_single_llm_gateway(cfg.llm, injection_filter=injection_filter, metrics=metrics)

    fallback_backend = cfg.llm.fallback_backend
    if fallback_backend == "none":
        return primary
    if fallback_backend == cfg.llm.backend:
        # Falling back to the same backend is pointless — skip the composite
        # so we don't double-instantiate an identical gateway.
        _log.warning(
            "llm_gateway_fallback_same_as_primary",
            backend=cfg.llm.backend,
        )
        return primary

    # Build the secondary from a copy of the LLM config with the backend
    # (and optional model name) overridden, so the local fallback can use a
    # different model identifier than the cloud primary without a second
    # config block.
    secondary_overrides: dict[str, object] = {"backend": fallback_backend}
    if cfg.llm.fallback_model_name is not None:
        secondary_overrides["model_name"] = cfg.llm.fallback_model_name
    secondary_cfg = cfg.llm.model_copy(update=secondary_overrides)
    secondary = _build_single_llm_gateway(
        secondary_cfg, injection_filter=injection_filter, metrics=metrics
    )

    from mousedroid.llm_gateway.fallback_gateway import FallbackLLMGateway

    _log.info(
        "llm_gateway_fallback_wired",
        primary=cfg.llm.backend,
        secondary=fallback_backend,
        fallback_model_name=cfg.llm.fallback_model_name,
        retry_cooldown_s=cfg.llm.fallback_retry_cooldown_s,
    )
    return FallbackLLMGateway(
        primary,
        secondary,
        retry_cooldown_s=cfg.llm.fallback_retry_cooldown_s,
        metrics=metrics,
    )
