"""Factory builders — natural-language mission parsing, VLM progress, replanning, lifecycle."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mousedroid.llm_gateway.protocol import LLMGatewayProtocol
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import (
        Settings,
    )
    from mousedroid.harness.protocol import TaskTrackerProtocol
    from mousedroid.llm_gateway.mission_parser import MissionParserProtocol
    from mousedroid.orchestrator.mission_lifecycle import (
        MissionLifecycle,
        MissionReplannerProtocol,
    )
    from mousedroid.reward.vlm_progress import VLMProgressHead
    from mousedroid.telemetry.metrics import MetricsRegistry

_log = get_logger(__name__)


def build_mission_parser(cfg: Settings) -> MissionParserProtocol:
    """Build NL mission parser with configurable speed/confidence mappings.

    Args:
        cfg: Root settings.

    Returns:
        Rule-based mission parser conforming to ``MissionParserProtocol``.
    """
    from mousedroid.llm_gateway.mission_parser import RuleBasedMissionParser

    parser = RuleBasedMissionParser(cfg.mission_parser)
    _log.info("mission_parser_built")
    return parser


def build_vlm_progress(cfg: Settings) -> VLMProgressHead | None:
    """Build the optional Tier C2.3 :class:`VLMProgressHead`.

    Returns ``None`` when ``cfg.mission.vlm_progress_enabled is False``
    (the default) — :func:`build_mission_lifecycle` then short-circuits
    and the orchestrator's POST_TICK seam stays a no-op so pre-Tier-C2.3
    deployments are byte-identical.

    When enabled the head wraps a :class:`MockVLMProgress` backend whose
    constant value comes from ``cfg.mission.vlm_mock_progress_value``. A
    real VLM backend (HF-hosted, BLIP-2, …) is a separate sprint — the
    protocol surface this factory targets is identical for either.

    Args:
        cfg: Root settings.

    Returns:
        A :class:`VLMProgressHead` instance or ``None`` when disabled.
    """
    if not cfg.mission.vlm_progress_enabled:
        _log.debug("vlm_progress_disabled")
        return None

    from mousedroid.reward.vlm_progress import MockVLMProgress, VLMProgressHead

    # Reuse the existing ``cfg.reward.vlm_progress`` block (cache size,
    # instruction, hash precision) — Tier C2.3 only adds the mock-value
    # gate inside ``MissionConfig`` so we can choose a value tuned to
    # the success threshold without disturbing the reward-module config.
    backend = MockVLMProgress(cfg.mission.vlm_mock_progress_value)
    head = VLMProgressHead(cfg=cfg.reward.vlm_progress, backend=backend)
    _log.info(
        "vlm_progress_built",
        backend="MockVLMProgress",
        mock_value=cfg.mission.vlm_mock_progress_value,
    )
    return head


def build_mission_replanner(
    cfg: Settings,
    *,
    llm_gateway: LLMGatewayProtocol | None,
    metrics: MetricsRegistry | None = None,
) -> MissionReplannerProtocol | None:
    """Build the optional Tier C2.3 LLM-backed mission replanner.

    Returns ``None`` in two cases (both preserve the defensive null path
    that :func:`build_mission_lifecycle` already handles):

    * ``cfg.mission.llm_replanner_enabled`` is ``False`` (the default).
    * ``llm_gateway`` is ``None`` — typically because
      ``cfg.llm.enabled`` is False. A warning is logged so an operator
      who enabled the replanner without enabling the gateway sees the
      misconfiguration at boot.

    The adapter is backend-agnostic: it wraps any
    :class:`LLMGatewayProtocol`-conforming instance (in-process
    llama-cpp OR the new HTTP ``OpenAICompatibleLLMGateway``), so the
    same wiring covers both deployment topologies — local Ollama, host-
    PC Ollama via 192.168.55.1, or OpenAI cloud.

    Args:
        cfg: Root settings.
        llm_gateway: Wired :class:`LLMGatewayProtocol`-conformant
            instance, or ``None`` when the gateway is disabled.
        metrics: Optional :class:`MetricsRegistry` for the
            ``mission_replan_llm_calls_total`` counter.

    Returns:
        An :class:`LLMGatewayMissionReplanner` or ``None``.
    """
    if not cfg.mission.llm_replanner_enabled:
        _log.debug("mission_replanner_disabled")
        return None
    if llm_gateway is None:
        _log.warning(
            "mission_replanner_no_gateway",
            hint=(
                "cfg.mission.llm_replanner_enabled=True but no LLM gateway "
                "is wired (cfg.llm.enabled likely False). Enable the "
                "gateway or leave the replanner disabled."
            ),
        )
        return None

    from mousedroid.orchestrator.llm_replanner import LLMGatewayMissionReplanner

    _log.info("mission_replanner_built", gateway_type=type(llm_gateway).__name__)
    return LLMGatewayMissionReplanner(
        gateway=llm_gateway,
        cfg=cfg.mission.replanner,
        metrics=metrics,
    )


def build_mission_lifecycle(
    cfg: Settings,
    *,
    task_tracker: TaskTrackerProtocol | None = None,
    vlm_progress: VLMProgressHead | None = None,
    replanner: MissionReplannerProtocol | None = None,
    metrics: MetricsRegistry | None = None,
) -> MissionLifecycle | None:
    """Build the optional :class:`MissionLifecycle` (Tier C2 / C2.2).

    Returns ``None`` when ``cfg.mission.replan_enabled`` is ``False`` so
    pre-C2 deployments produce byte-identical behaviour (no lifecycle,
    no replans, no new structured events).

    Also returns ``None`` (defensively) when ``replan_enabled=True`` but
    either ``vlm_progress`` or ``replanner`` is missing — in that
    configuration the lifecycle would stall on every tick (no VLM head
    means ``_score_progress`` is constant ``0.0``, which trips
    ``stall_window_ticks`` and then fails with
    ``reason='llm_replan_unavailable'`` because no replanner is wired).
    Returning ``None`` is strictly safer than instantiating a
    self-failing state machine; the orchestrator's tick seam becomes a
    no-op exactly as in the disabled case. The decision is logged at
    warning level so operators can spot the missing dependency at boot
    rather than after the first stall window elapses.

    Args:
        cfg: Root settings.
        task_tracker: Optional :class:`TaskTrackerProtocol`. When wired,
            :class:`MissionLifecycle` submits a synthetic task on
            ``start_mission`` and forwards terminal lifecycle states
            (SUCCEEDED → COMPLETED, FAILED → FAILED) via
            ``tracker.update`` so the unified active-task list reflects
            mission outcomes alongside skill / OpenClaw tasks.
        vlm_progress: :class:`VLMProgressHead` providing goal-progress
            feedback per tick. Required for the lifecycle to make
            forward progress; ``None`` triggers the defensive ``None``
            return described above.
        replanner: :class:`MissionReplannerProtocol`-compliant object.
            Required so the lifecycle has a recovery path when stalls
            fire; ``None`` triggers the defensive ``None`` return.
        metrics: Optional shared metrics registry. When supplied, every
            transition + replan + terminal duration increments the
            corresponding Tier C2 metric family.

    Returns:
        :class:`MissionLifecycle` when ``cfg.mission.replan_enabled`` is
        True AND both ``vlm_progress`` and ``replanner`` are wired,
        otherwise ``None``.
    """
    if not cfg.mission.replan_enabled:
        _log.debug("mission_lifecycle_disabled")
        return None

    # Defensive dependency check (Copilot HIGH): wiring the lifecycle
    # without a VLM progress head and an LLM-backed replanner produces a
    # state machine that can only ever fail with ``llm_replan_unavailable``.
    # Skip construction and surface the missing-dependency warning instead.
    missing_deps: list[str] = []
    if vlm_progress is None:
        missing_deps.append("vlm_progress")
    if replanner is None:
        missing_deps.append("replanner")
    if missing_deps:
        _log.warning(
            "mission_lifecycle_dependencies_missing",
            missing=missing_deps,
            hint=(
                "Wire VLMProgressHead + MissionReplannerProtocol before "
                "setting cfg.mission.replan_enabled=True, or leave the "
                "lifecycle disabled to keep the pre-C2.2 byte-identical path."
            ),
        )
        return None

    from mousedroid.orchestrator.mission_lifecycle import MissionLifecycle

    _log.info("mission_lifecycle_built")
    return MissionLifecycle(
        cfg.mission,
        task_tracker=task_tracker,
        vlm_progress=vlm_progress,
        replanner=replanner,
        metrics=metrics,
    )
