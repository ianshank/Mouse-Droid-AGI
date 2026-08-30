"""Factory builders — MCP server, agent harness, and skills.

Harness: task tracker/journal/approval/hooks. Skills: loaders/registry/delegator.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, cast

from mousedroid.factory.arm import build_llm_replanner
from mousedroid.factory.llm_gateway import build_injection_filter
from mousedroid.logging.setup import get_logger
from mousedroid.safety.protocol import SafetyMonitorProtocol

if TYPE_CHECKING:
    from mousedroid.common.tools.registry import ToolRegistry
    from mousedroid.config.schema import (
        Settings,
    )
    from mousedroid.harness.approval.protocol import ApprovalGateProtocol
    from mousedroid.mcp.protocol import MCPServerProtocol
    from mousedroid.memory.tier import MemoryTier
    from mousedroid.telemetry.log_buffer import LogRingBuffer
    from mousedroid.telemetry.metrics import MetricsRegistry
    from mousedroid.telemetry.protocol import TelemetryPublisherProtocol

_log = get_logger(__name__)


def build_mcp_server(
    cfg: Settings,
    tool_registry: ToolRegistry,
    safety_monitor: SafetyMonitorProtocol,
    publisher: TelemetryPublisherProtocol | None = None,
    log_buffer: LogRingBuffer | None = None,
    metrics_registry: MetricsRegistry | None = None,
    memory_tier: MemoryTier | None = None,
) -> MCPServerProtocol | None:
    """Build the MCP server when ``cfg.mcp`` is enabled.

    Returns ``None`` (and logs a structured warning) when the optional
    ``mcp`` package is not installed, so missing extras never break a
    boot. The server itself runs without the SDK for in-process tests
    but only binds a real transport when the package is present.

    Args:
        cfg: Root settings.
        tool_registry: Shared tool registry instance.
        safety_monitor: Live safety monitor for actuation gates.
        publisher: Optional telemetry publisher (for the telemetry
            resource).
        log_buffer: Optional log ring buffer (for the logs resource).
        metrics_registry: Optional metrics registry.
        memory_tier: Optional memory tier (for the memory resource).

    Returns:
        Server implementing :class:`MCPServerProtocol`, or ``None`` when
        disabled / unavailable.
    """
    if cfg.mcp is None or not cfg.mcp.enabled:
        return None
    if (
        cfg.telemetry.enabled
        and cfg.mcp.transport != "stdio"
        and cfg.mcp.port == cfg.telemetry.port
    ):
        msg = (
            f"mcp.port ({cfg.mcp.port}) collides with telemetry.port "
            f"({cfg.telemetry.port}); pick distinct ports"
        )
        raise ValueError(msg)
    from mousedroid.mcp.server import MouseDroidMCPServer

    _log.info(
        "mcp_server_built",
        transport=cfg.mcp.transport,
        host=cfg.mcp.host,
        port=cfg.mcp.port,
        memory_enabled=cfg.mcp.resources.memory_enabled and memory_tier is not None,
    )
    return MouseDroidMCPServer(
        cfg=cfg.mcp,
        root_cfg=cfg,
        tool_registry=tool_registry,
        safety_monitor=safety_monitor,
        telemetry_publisher=publisher,
        log_buffer=log_buffer,
        metrics_registry=metrics_registry,
        memory_tier=memory_tier,
    )


def build_task_tracker(cfg: Settings) -> Any:
    """Build the harness task tracker, or ``None`` when the harness is off.

    Args:
        cfg: Root settings.

    Returns:
        ``InMemoryTaskTracker`` when ``cfg.harness.tracker.enabled`` is
        ``True``; otherwise ``None`` so the orchestrator can short-circuit.
    """
    if cfg.harness is None or not cfg.harness.tracker.enabled:
        return None
    from mousedroid.harness.task_tracker import InMemoryTaskTracker

    return InMemoryTaskTracker(cfg.harness.tracker)


def build_journal(cfg: Settings) -> Any:
    """Build the harness journal backend.

    Args:
        cfg: Root settings.

    Returns:
        A concrete journal implementing ``JournalProtocol``. ``NullJournal``
        is the default — never raises and never writes to disk.
    """
    from mousedroid.harness.journal.null_journal import NullJournal

    if cfg.harness is None:
        return NullJournal()
    backend = cfg.harness.journal.backend
    if backend == "jsonl":
        from mousedroid.harness.journal.jsonl_journal import JSONLJournal

        return JSONLJournal(cfg.harness.journal)
    if backend == "lmdb":
        from mousedroid.harness.journal.lmdb_journal import LMDBJournal

        return LMDBJournal(cfg.harness.journal)
    return NullJournal()


def _resolve_approval_callback(
    dotted_path: str | None,
) -> Callable[[Any], Awaitable[bool]]:
    """Import an async ``(ApprovalRequest) -> bool`` callable from a dotted path.

    When the path is ``None`` or the import fails, returns a fail-closed
    fallback that denies every request — explicit configuration is
    required to grant approvals through the callback gate. The error is
    logged at WARNING so misconfigured deployments are visible without
    silently permitting actions.
    """

    async def _deny(_request: Any) -> bool:
        return False

    if not dotted_path:
        _log.warning(
            "approval_callback_dotted_path_missing",
            note="callback gate will deny all requests until configured",
        )
        return _deny

    module_path, _, attr = dotted_path.rpartition(".")
    if not module_path or not attr:
        _log.warning(
            "approval_callback_dotted_path_invalid",
            dotted_path=dotted_path,
        )
        return _deny

    try:
        import importlib

        module = importlib.import_module(module_path)
        target = getattr(module, attr)
    except (ImportError, AttributeError) as exc:
        _log.warning(
            "approval_callback_resolution_failed",
            dotted_path=dotted_path,
            error=str(exc),
        )
        return _deny

    if not callable(target):
        _log.warning(
            "approval_callback_not_callable",
            dotted_path=dotted_path,
        )
        return _deny

    _log.info("approval_callback_resolved", dotted_path=dotted_path)
    return cast("Callable[[Any], Awaitable[bool]]", target)


def build_approval_gate(cfg: Settings) -> ApprovalGateProtocol:
    """Build the configured :class:`ApprovalGateProtocol`.

    The default ``"auto"`` gate approves every request (the
    ``PolicyApprovalGate`` wrapper is unnecessary when no patterns are
    configured). Other modes use an inner gate matched to ``cfg.harness.approval``.

    Args:
        cfg: Root settings.

    Returns:
        A concrete approval gate implementing ``ApprovalGateProtocol``.
    """
    # PR-105b: tighten the `inner` annotation from ``object`` to the
    # protocol so ``PolicyApprovalGate(inner, ...)`` typechecks under
    # ``mypy --strict`` (closes the PR-98-introduced type hole reported
    # at factory.py:2262). All concrete branches below assign a
    # protocol-conforming instance — the previous ``object`` annotation
    # was an over-broad placeholder.
    from mousedroid.harness.approval.auto import AutoApproveGate
    from mousedroid.harness.approval.policy import PolicyApprovalGate

    if cfg.harness is None:
        return AutoApproveGate()
    approval = cfg.harness.approval
    inner: ApprovalGateProtocol
    if approval.gate == "auto":
        inner = AutoApproveGate()
    elif approval.gate == "cli":
        from mousedroid.harness.approval.cli import CLIApprovalGate

        inner = CLIApprovalGate(
            timeout_s=approval.cli_timeout_s,
            on_timeout=approval.on_timeout,
        )
    elif approval.gate == "callback":
        from mousedroid.harness.approval.callback import (
            AsyncCallbackApprovalGate,
        )

        callback = _resolve_approval_callback(approval.callback_dotted_path)
        inner = AsyncCallbackApprovalGate(
            callback,
            timeout_s=approval.cli_timeout_s,
            on_timeout=approval.on_timeout,
        )
    else:  # "policy" — caller is expected to supply patterns; default to auto
        inner = AutoApproveGate()

    if approval.require_approval_tool_patterns or approval.require_approval_skill_patterns:
        inner = PolicyApprovalGate(
            inner,
            tool_patterns=tuple(approval.require_approval_tool_patterns),
            skill_patterns=tuple(approval.require_approval_skill_patterns),
        )

    if cfg.openclaw is not None and cfg.openclaw.enabled:
        from mousedroid.harness.approval.openclaw_gate import OpenClawSafetyGate
        from mousedroid.harness.approval.sandbox_gate import SandboxPolicyGate

        filter_impl = build_injection_filter(cfg)
        inner = OpenClawSafetyGate(inner, filter_impl, cfg.openclaw)
        inner = SandboxPolicyGate(inner, cfg.openclaw.policy)

    return inner


def build_skill_loaders(cfg: Settings) -> tuple[Any, ...]:
    """Build the configured tuple of :class:`SkillLoaderProtocol` instances.

    Args:
        cfg: Root settings.

    Returns:
        Tuple of skill loaders to drain into the registry. Empty when
        ``cfg.harness.skills.enabled`` is False.
    """
    if cfg.harness is None or not cfg.harness.skills.enabled:
        return ()
    from mousedroid.skills.loaders import (
        MarkdownAgentLoader,
        YAMLManifestLoader,
    )

    skills_cfg = cfg.harness.skills
    return (
        YAMLManifestLoader(skills_cfg.manifest_glob),
        MarkdownAgentLoader(skills_cfg.markdown_agent_dirs),
    )


def build_memory_exporter(cfg: Settings) -> Any | None:
    """Build the OpenClaw MEMORY.md exporter when configured.

    Returns ``None`` when OpenClaw is disabled OR when
    ``cfg.openclaw.shared_memory_path`` is unset; the orchestrator hook
    is gated on a non-None return so disabled deployments incur zero
    runtime cost.

    Tunable parameters (``max_entries``, ``entry_truncate_chars``) come
    from :class:`OpenClawConfig` so the exporter has zero hardcoded
    knobs at construction time (per CLAUDE.md rule #3).
    """
    if cfg.openclaw is None or not cfg.openclaw.enabled or cfg.openclaw.shared_memory_path is None:
        return None
    from mousedroid.memory.exporter import MarkdownReplayExporter

    _log.info(
        "memory_exporter_built",
        path=str(cfg.openclaw.shared_memory_path),
        max_entries=cfg.openclaw.export_max_entries,
        entry_truncate_chars=cfg.openclaw.export_entry_truncate_chars,
    )
    return MarkdownReplayExporter(
        cfg.openclaw.shared_memory_path,
        max_entries=cfg.openclaw.export_max_entries,
        entry_truncate_chars=cfg.openclaw.export_entry_truncate_chars,
    )


def build_builtin_skills(cfg: Settings) -> tuple[Any, ...]:
    """Return the OpenClaw-publishable :class:`SkillSpec` tuple.

    Returns the four builtin specs (``mousedroid-navigate``,
    ``mousedroid-sensor-report``, ``mousedroid-voice``,
    ``mousedroid-world-model``) when OpenClaw is enabled; otherwise an
    empty tuple so existing deployments still see an empty registry.
    """
    if cfg.openclaw is None or not cfg.openclaw.enabled:
        return ()
    from mousedroid.skills.builtin import all_builtin_specs

    return all_builtin_specs()


def build_skill_registry(cfg: Settings, loaders: tuple[Any, ...] = ()) -> Any:
    """Build the skill registry pre-populated from ``loaders`` and builtins.

    Args:
        cfg: Root settings — drives whether the OpenClaw builtin specs
            (``mousedroid-navigate`` etc.) are auto-registered.
        loaders: Additional skill loaders to drain at construction time.

    Returns:
        A populated ``SkillRegistry``.
    """
    from mousedroid.skills.registry import SkillRegistry

    registry = SkillRegistry()
    if loaders:
        registry.load_all(loaders)
    for spec in build_builtin_skills(cfg):
        registry.register(spec)
    return registry


def _build_sub_agent_factory(
    cfg: Settings,
    skill_registry: Any,
    journal: Any,
    llm_gateway: Any,
) -> Callable[[str], Any]:
    """Return a ``(skill_name) -> SubAgentProtocol`` factory honouring config.

    The configured ``cfg.harness.skills.backend`` selects the concrete
    sub-agent class:

    * ``"noop"`` (default) — a deterministic :class:`NoOpSubAgent` per
      skill so tests and dry-runs stay free of external dependencies.
    * ``"llm_gateway"`` — :class:`LLMBackedSubAgent` wired to the local
      LLM gateway when available, falling back to the no-op.
    * ``"anthropic"`` — :class:`LLMBackedSubAgent` backed by an
      :class:`AnthropicReplanner` when ``arm_planning.llm_replanner.backend``
      points at Anthropic; otherwise falls back to ``noop`` with a warning.

    The journal is threaded through so sub-agents can record their own
    lifecycle events alongside the delegator's.
    """
    from mousedroid.skills.sub_agent import LLMBackedSubAgent, NoOpSubAgent

    backend = "noop" if cfg.harness is None else cfg.harness.skills.backend

    async def _journal_append(entry: Any) -> None:
        await journal.append(entry)

    def _factory(skill_name: str) -> Any:
        skill = skill_registry.get(skill_name) if skill_registry is not None else None
        if backend == "llm_gateway" and llm_gateway is not None and skill is not None:
            return LLMBackedSubAgent(
                skill,
                llm_gateway=llm_gateway,
                journal_append=_journal_append,
            )
        if backend == "anthropic":
            anthropic_gateway = build_llm_replanner(cfg)
            if skill is not None and anthropic_gateway is not None:
                return LLMBackedSubAgent(
                    skill,
                    llm_gateway=anthropic_gateway,
                    journal_append=_journal_append,
                )
            _log.warning(
                "skill_backend_anthropic_unavailable_fallback_noop",
                skill=skill_name,
            )
        if backend != "noop":
            _log.debug(
                "skill_backend_falling_back_to_noop",
                skill=skill_name,
                backend=backend,
            )
        return NoOpSubAgent(skill_name)

    return _factory


def build_skill_delegator(
    cfg: Settings,
    skill_registry: Any,
    approval_gate: Any,
    journal: Any,
    task_tracker: Any,
    *,
    llm_gateway: Any = None,
) -> Any:
    """Wire the :class:`SkillDelegator` once all dependencies are built.

    Args:
        cfg: Root settings.
        skill_registry: The populated skill registry.
        approval_gate: Approval gate to consult before delegation.
        journal: Journal that receives delegation events.
        task_tracker: Tracker that owns task lifecycle.
        llm_gateway: Optional local LLM gateway used when the configured
            ``skills.backend`` is ``"llm_gateway"``.

    Returns:
        Configured ``SkillDelegator``, or ``None`` when the harness or the
        skills sub-config is disabled.
    """
    if cfg.harness is None or not cfg.harness.skills.enabled or task_tracker is None:
        return None
    from mousedroid.skills.delegator import SkillDelegator

    agent_factory = _build_sub_agent_factory(cfg, skill_registry, journal, llm_gateway)

    return SkillDelegator(
        skill_registry,
        approval_gate,
        journal,
        task_tracker,
        agent_factory=agent_factory,
    )


def build_hook_registry(cfg: Settings, journal: Any) -> Any:
    """Build the hook registry, optionally seeded with default hooks.

    When ``cfg.harness.hooks.journal_events`` is True (the default), a
    journal-append hook is registered on every phase so the ledger
    captures tick activity without further wiring.

    Args:
        cfg: Root settings.
        journal: Journal used by the seeded ``journal:*`` hooks.

    Returns:
        Concrete ``HookRegistry`` (or ``NullHookRegistry`` when the
        harness is disabled).
    """
    from mousedroid.harness.hooks import HookRegistry, NullHookRegistry

    if cfg.harness is None:
        # Harness disabled — return the no-op registry so the 30 Hz hot
        # loop pays no cost for hook dispatch.
        return NullHookRegistry()

    hooks_cfg = cfg.harness.hooks
    registry = HookRegistry()
    enabled_set = frozenset(hooks_cfg.enabled_hooks)

    # ``fail_fast=True`` overrides per-hook ``error_policy`` so any failure
    # propagates and aborts the tick. Otherwise the per-hook policy is used.
    effective_policy = "raise" if hooks_cfg.fail_fast else hooks_cfg.error_policy

    if hooks_cfg.journal_events:
        from mousedroid.harness.journal.protocol import JournalEntry
        from mousedroid.harness.protocol import HookPhase, HookSpec

        async def _append_for(phase_value: str, ctx: Any) -> None:
            await journal.append(
                JournalEntry(
                    phase=phase_value,
                    event=f"orchestrator_{phase_value}",
                    payload={"tick": ctx.tick_index},
                )
            )

        def _make_handler(
            phase_value: str,
        ) -> Callable[[Any], Awaitable[None]]:
            async def _handler(ctx: Any) -> None:
                await _append_for(phase_value, ctx)

            return _handler

        for phase in HookPhase:
            spec_name = f"journal:{phase.value}"
            # When ``enabled_hooks`` is non-empty it acts as an opt-in
            # allowlist; otherwise every default hook is registered.
            if enabled_set and spec_name not in enabled_set:
                continue
            registry.register(
                HookSpec(
                    name=spec_name,
                    phase=phase,
                    handler=_make_handler(phase.value),
                    error_policy=effective_policy,
                )
            )
    return registry
