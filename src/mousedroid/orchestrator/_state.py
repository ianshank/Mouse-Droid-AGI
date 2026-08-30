"""Shared attribute/method type declarations for the orchestrator mixins.

`MouseDroidOrchestrator` is composed from `orchestrator.py` (`__init__` +
`tick()`) plus 7 sibling `_*_mixin.py` files (ADR-017). mypy --strict
type-checks each mixin file independently against its own class body, so a
mixin method that reads `self._cfg` or calls `self._compute_curiosity_scores()`
-- both defined elsewhere, on the concrete class or a *different* mixin --
has no visibility into where that name actually lives.

`telemetry/metrics/_registry_*.py` (an earlier, smaller mixin split)
solved the same problem by having each mixin declare, as a bare
class-level annotation, only the handful of attributes *it itself*
touches (e.g. `_LidarMetricsMixin` declares just `_cfg: MetricsConfig`).
That works well when the overlap between mixins is small. Here it is not:
`MouseDroidOrchestrator.__init__` carries 45+ attributes and all 7 mixins
collectively touch nearly every one of them, so per-mixin duplication
would mean re-declaring most of this file's contents seven times with no
single source of truth. Instead every mixin inherits from `_OrchestratorState`
(a pure type-declaration class -- no `__init__`, no production
implementations -- only fail-loud `raise NotImplementedError` stubs a real
mixin must override -- never instantiated on its own) alongside its real
base. Python's MRO
handles the resulting diamond (all 7 mixins -> `_OrchestratorState` ->
`object`) the same way it already handles `MouseDroidOrchestrator`
inheriting from all 7 mixins -- consistent C3 linearization, not a
conflict.

This file is intentionally NEVER imported by `orchestrator.py` itself:
`__init__` already assigns concrete values with their own inferred types,
and re-declaring them there would be redundant, not additive.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from mousedroid.agents.base import AgentProtocol
    from mousedroid.cloud.protocol import (
        CloudExperienceExporterProtocol,
        CloudFirestoreSyncProtocol,
        CloudMetricsExporterProtocol,
        CloudTelemetrySinkProtocol,
        PendingWeightUpdate,
        WeightUpdatePollerProtocol,
    )
    from mousedroid.cognitive.cognitive_core import CognitiveCore
    from mousedroid.common.time.protocol import ClockProtocol
    from mousedroid.comms.protocol import ESP32CommProtocol
    from mousedroid.config.schema import Settings
    from mousedroid.curiosity.protocol import CuriosityProtocol
    from mousedroid.experience.logger import ExperienceLogger
    from mousedroid.growth.coordinator import GrowthDistillationCoordinator
    from mousedroid.hardware.accelerator.hailo_runtime import HailoRuntimeProtocol
    from mousedroid.harness.journal.protocol import JournalProtocol
    from mousedroid.harness.protocol import HookRegistryProtocol, TaskTrackerProtocol
    from mousedroid.health.watchdog import WatchdogProtocol
    from mousedroid.learning.on_device.replay_trigger import ReplayTriggerCoordinator
    from mousedroid.llm_gateway.mission_parser import MissionParserProtocol
    from mousedroid.llm_gateway.protocol import LLMGatewayProtocol
    from mousedroid.mcp.protocol import MCPServerProtocol
    from mousedroid.memory.exporter import MemoryExporterProtocol
    from mousedroid.memory.tier import MemoryTier
    from mousedroid.orchestrator.face_controller import FaceController
    from mousedroid.orchestrator.mission_dispatcher import MissionDispatcherProtocol
    from mousedroid.orchestrator.mission_lifecycle import MissionLifecycle
    from mousedroid.safety.context import SafetyContext
    from mousedroid.safety.projector_protocol import SafetyActionProjectorProtocol
    from mousedroid.safety.protocol import SafetyMonitorProtocol
    from mousedroid.sensing.manager import SensorManager
    from mousedroid.skills.delegator import SkillDelegator
    from mousedroid.telemetry.failure_recorder import FailureRecorder
    from mousedroid.telemetry.metrics import MetricsRegistry
    from mousedroid.telemetry.protocol import TelemetryPublisherProtocol, TelemetryServerProtocol
    from mousedroid.vla.policy import VLAPolicyProtocol
    from mousedroid.voice.greeting import GreeterProtocol
    from mousedroid.voice.protocol import VoiceEngineProtocol
    from mousedroid.world_model.protocol import LatentContextProtocol, WorldModelProtocol


class _OrchestratorState:
    """Bare type declarations plus fail-loud stubs. Never instantiated.

    Never given a production implementation -- do not "clean up" a stub's
    `raise NotImplementedError` body into a silent no-op. Every real mixin
    implementation wins over its stub today (this class sits last in the
    MRO, right before `object`), so the raise never executes in production;
    it exists as a runtime backstop for the one case
    `tests/regression/test_orchestrator_mixin_surface.py::test_every_state_stub_is_overridden_somewhere_reachable`
    cannot catch by static inspection alone -- a caller reaching this stub
    despite that test staying green (e.g. a method removed and the test
    itself skipped or deleted in the same change).

    One block per constructor argument/derived attribute in
    `orchestrator.py::MouseDroidOrchestrator.__init__`, grouped by category
    (constructor passthroughs, then derived/resolved-from-default attributes,
    then latent state) rather than strictly mirroring `__init__`'s own
    assignment order -- `_supports_lateral`, for instance, is assigned
    between two passthroughs in `__init__` but declared with the other
    derived attributes here, because it groups better than a positional
    match would. Keep this file's declarations in sync with `__init__` in
    the same PR that changes either.
    """

    # Constructor passthroughs (identical to their __init__ parameter type)
    _world_model: WorldModelProtocol
    _agents: list[AgentProtocol]
    _safety_monitor: SafetyMonitorProtocol
    _esp32: ESP32CommProtocol
    _sensor_manager: SensorManager
    _cognitive_core: CognitiveCore | None
    _telemetry_publisher: TelemetryPublisherProtocol | None
    _telemetry_server: TelemetryServerProtocol | None
    _voice_engine: VoiceEngineProtocol | None
    _hailo_runtime: HailoRuntimeProtocol | None
    _memory_tier: MemoryTier | None
    _experience_logger: ExperienceLogger | None
    _curiosity_module: CuriosityProtocol | None
    _llm_gateway: LLMGatewayProtocol | None
    _mission_parser: MissionParserProtocol | None
    _watchdog: WatchdogProtocol | None
    _cloud_sink: CloudTelemetrySinkProtocol | None
    _cloud_experience_exporter: CloudExperienceExporterProtocol | None
    _cloud_metrics_exporter: CloudMetricsExporterProtocol | None
    _cloud_firestore_sync: CloudFirestoreSyncProtocol | None
    _tool_registry: Any | None
    _face_controller: FaceController | None
    _mcp_server: MCPServerProtocol | None
    _vla_policy: VLAPolicyProtocol | None
    _cfg: Settings

    # Derived / resolved-from-default attributes
    _supports_lateral: bool
    _hook_registry: HookRegistryProtocol
    _task_tracker: TaskTrackerProtocol | None
    _journal: JournalProtocol
    _skill_delegator: SkillDelegator | None
    _memory_exporter: MemoryExporterProtocol | None
    _mission_dispatcher: MissionDispatcherProtocol | None
    _memory_export_every_n: int
    _clock: ClockProtocol
    _failure_recorder: FailureRecorder
    _liveness_tracker: Any | None
    _mock_telemetry_source: Any | None
    _metrics: MetricsRegistry | None
    _weight_update_pollers: dict[str, WeightUpdatePollerProtocol]
    _weight_update_loader: Callable[[PendingWeightUpdate], object] | None
    _safety_projector: SafetyActionProjectorProtocol | None
    _mission_lifecycle: MissionLifecycle | None
    _on_device_coordinator: ReplayTriggerCoordinator | None
    _on_device_task: asyncio.Task[Any] | None
    _on_device_tasks: set[asyncio.Task[Any]]
    _growth_coordinator: GrowthDistillationCoordinator | None
    _growth_task: asyncio.Task[Any] | None
    _growth_tasks: set[asyncio.Task[Any]]
    _greeter: GreeterProtocol | None
    _prev_obs_for_vlm: torch.Tensor | None
    _mission_seq: int
    _running: bool
    _tick_count: int
    _consolidation_task: asyncio.Task[Any] | None
    _consolidation_tasks: set[asyncio.Task[Any]]
    _cloud_publish_tasks: set[asyncio.Task[Any]]

    # Latent state
    _h: torch.Tensor
    _z: torch.Tensor
    _prev_action: torch.Tensor
    _latent_buffer: deque[tuple[torch.Tensor, torch.Tensor]]
    _latent_context: LatentContextProtocol | None

    # Cross-mixin method calls. Declared here (not just the attributes
    # above) for the same reason: mypy checks each mixin file in
    # isolation and cannot otherwise see a method defined on a sibling
    # mixin -- including ``tick()``, which per ADR-014 lives directly on
    # the concrete ``MouseDroidOrchestrator`` class, not on any mixin.
    # Signatures must match the real implementation exactly -- this is a
    # type-only redeclaration, not a second source of truth for behavior
    # (mypy does not re-check the body against this signature beyond the
    # normal override-compatibility rules).
    #
    # Every stub raises NotImplementedError, including the ``-> None``
    # ones (mypy accepts an unconditional raise against any return type).
    # `_OrchestratorState` sits last in the MRO, right before `object`, so
    # today a real mixin's implementation always wins and these bodies
    # never execute. A silent ``...`` body would still be true if a mixin
    # method were ever accidentally deleted or renamed: the call would
    # fall through to this stub and no-op instead of raising, which for a
    # safety-critical 30 Hz control loop is a worse failure mode than a
    # loud crash -- so every stub here fails loudly, on purpose.
    async def tick(self) -> None:
        raise NotImplementedError

    def _compute_curiosity_scores(self) -> dict[str, float]:
        raise NotImplementedError

    async def _consolidation_loop(self) -> None:
        raise NotImplementedError

    async def _growth_distill_loop(self) -> None:
        raise NotImplementedError

    def _growth_enabled(self) -> bool:
        raise NotImplementedError

    def _on_device_learning_enabled(self) -> bool:
        raise NotImplementedError

    async def _on_device_update_loop(self) -> None:
        raise NotImplementedError

    def _validate_latent(
        self, h: torch.Tensor, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        raise NotImplementedError

    async def _voice_lifecycle(self, event: str) -> None:
        raise NotImplementedError

    async def _try_sensor_recovery(self, safety_ctx: SafetyContext) -> bool:
        raise NotImplementedError
