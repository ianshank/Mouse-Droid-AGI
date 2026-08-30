"""Factory builders — expressive voice/face output layer (face controller, speaker, greeter, voice engine)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from mousedroid.logging.setup import get_logger
from mousedroid.hardware.protocols import (
    FaceDisplayProtocol,
    SpeakerProtocol,
    VisionProtocol,
)
from mousedroid.voice.protocol import VoiceEngineProtocol


if TYPE_CHECKING:
    import torch
    from torch import Tensor

    from mousedroid.agents.base import AgentProtocol
    from mousedroid.arm.protocols import (
        ArmControllerProtocol,
        ArmDriverProtocol,
        ArmEnvironmentProtocol,
        ArmPerceptionProtocol,
        ArmPlannerProtocol,
        SymbolicPlannerBackend,
    )
    from mousedroid.cloud.protocol import (
        CloudExperienceExporterProtocol,
        CloudFirestoreSyncProtocol,
        CloudLoggingSinkProtocol,
        CloudMetricsExporterProtocol,
        CloudTelemetrySinkProtocol,
        PendingWeightUpdate,
        WeightUpdatePollerProtocol,
    )
    from mousedroid.cognitive.bdi_model import NeuralBDI
    from mousedroid.cognitive.cognitive_core import CognitiveCore
    from mousedroid.common.time.protocol import ClockProtocol
    from mousedroid.common.tools.registry import ToolRegistry
    from mousedroid.config.schema import (
        ESP32Config,
        LLMConfig,
        ModelConfig,
        Settings,
        UltrasonicConfig,
    )
    from mousedroid.curiosity.protocol import CuriosityProtocol
    from mousedroid.efficiency.tensorrt import TensorRTCompilerProtocol
    from mousedroid.experience.logger import ExperienceLogger
    from mousedroid.experience.record import MouseDroidExperienceRecord
    from mousedroid.growth.coordinator import GrowthDistillationCoordinator
    from mousedroid.hardware.accelerator.hailo_runtime import HailoRuntimeProtocol
    from mousedroid.hardware.camera.feature_extractor import FeatureExtractorProtocol
    from mousedroid.harness.approval.protocol import ApprovalGateProtocol
    from mousedroid.harness.protocol import TaskTrackerProtocol
    from mousedroid.health.monitor import HealthMonitor
    from mousedroid.learning.on_device.hot_swap import OnDeviceWeightUpdateSource
    from mousedroid.learning.on_device.replay_trigger import ReplayTriggerCoordinator
    from mousedroid.learning.on_device.slot_store import CandidateSlot, OnDeviceSlotStore
    from mousedroid.llm_gateway.mission_parser import MissionParserProtocol
    from mousedroid.mcp.protocol import MCPServerProtocol
    from mousedroid.memory.episodic import EpisodicReplay
    from mousedroid.memory.tier import MemoryTier
    from mousedroid.orchestrator.face_controller import FaceController
    from mousedroid.orchestrator.mission_dispatcher import MissionDispatcherProtocol
    from mousedroid.orchestrator.mission_lifecycle import (
        MissionLifecycle,
        MissionReplannerProtocol,
    )
    from mousedroid.reward.protocol import RewardModelProtocol
    from mousedroid.reward.vlm_progress import VLMProgressHead
    from mousedroid.sensing.manager import SensorManager
    from mousedroid.sim.protocols import RoverEnvProtocol
    from mousedroid.telemetry.failure_recorder import FailureRecorder
    from mousedroid.telemetry.log_buffer import LogRingBuffer
    from mousedroid.telemetry.metrics import MetricsRegistry
    from mousedroid.telemetry.protocol import TelemetryPublisherProtocol, TelemetryServerProtocol
    from mousedroid.training.observability import ExperimentLoggerProtocol
    from mousedroid.training.replay import ReplayReaderProtocol
    from mousedroid.training.replay.lmdb_reader import LMDBReplayReader
    from mousedroid.vla.policy import VLAPolicyProtocol
    from mousedroid.voice.greeting import Greeter, GreeterProtocol
    from mousedroid.voice.mock_tts import MockTTS
    from mousedroid.voice.tts import PiperTTS
    from mousedroid.world_model.encoder import MultimodalEncoder
    from mousedroid.world_model.protocol import LatentContextProtocol, WorldModelProtocol
    from mousedroid.world_model.rssm import RSSM

_log = get_logger(__name__)


def build_face_controller(
    cfg: Settings, driver: FaceDisplayProtocol | None
) -> FaceController | None:
    """Wrap a face-display driver in a :class:`FaceController`.

    Returns ``None`` when ``driver`` is ``None`` (subsystem disabled) so the
    orchestrator can simply skip face-display calls.

    Args:
        cfg: Root settings (must have ``face_display`` populated when
            ``driver`` is not ``None``).
        driver: Optional face-display driver from :func:`build_face_display`.

    Returns:
        :class:`FaceController` instance or ``None``.
    """
    if driver is None or cfg.face_display is None:
        return None

    from mousedroid.orchestrator.face_controller import FaceController

    return FaceController(driver, cfg.face_display)


def build_speaker(
    cfg: Settings, *, metrics: MetricsRegistry | None = None
) -> SpeakerProtocol | None:
    """Build USB speaker driver based on config.

    Args:
        cfg: Root settings.
        metrics: Optional shared metrics registry, threaded keyword-only so the
            USB speaker's retry-exhaustion path surfaces
            ``voice_speaker_degraded_total`` on ``/metrics``.

    Returns:
        Speaker driver conforming to ``SpeakerProtocol``, or None if disabled.
    """
    if cfg.speaker is None or not cfg.speaker.enabled:
        _log.info("speaker_disabled")
        return None

    if cfg.mock_hardware:
        from mousedroid.hardware.audio.mock_speaker import MockSpeaker

        _log.info("speaker_mock_built", sample_rate=cfg.speaker.sample_rate)
        return MockSpeaker(cfg.speaker)

    from mousedroid.hardware.audio.usb_speaker import UsbSpeaker

    _log.info(
        "speaker_built",
        sample_rate=cfg.speaker.sample_rate,
        device_name=cfg.speaker.device_name,
    )
    return UsbSpeaker(cfg.speaker, metrics=metrics)


def build_greeter(
    cfg: Settings,
    *,
    voice_engine: VoiceEngineProtocol | None = None,
) -> Greeter:
    """Build the operator-tools greeting subsystem.

    Opt-in: raises :class:`ValueError` when ``cfg.greeting is None`` or
    when ``cfg.greeting.enabled is False``. This makes mis-invocations
    surface as a clear error rather than a silently-skipped no-op
    (mirrors the discipline of every other ``build_*`` builder that
    handles disabled subsystems by returning ``None`` — the greeter
    is operator-tools, NOT orchestrator wiring, so a None return
    would obscure operator intent).

    Args:
        cfg: Root settings. ``cfg.greeting`` MUST be a non-None
            enabled :class:`GreetingConfig`.
        voice_engine: Optional pre-built voice engine. Test seam so
            unit tests can inject a mock without depending on the
            ``build_voice_engine`` factory chain. Production callers
            (the ``scripts/greet_intro.py`` CLI) pass ``None`` so
            this builder routes through the standard factory path.

    Returns:
        Configured :class:`Greeter` ready to be ``await``ed via
        :meth:`Greeter.greet`. Caller still owns the voice-engine
        lifecycle (``await voice_engine.start()`` before, ``stop()``
        after) — see :class:`Greeter` docstring.

    Raises:
        ValueError: When greeting is disabled / unconfigured, or when
            the voice engine cannot be built (e.g. ``cfg.voice.enabled``
            is False).
    """
    if cfg.greeting is None or not cfg.greeting.enabled:
        msg = (
            "build_greeter requires Settings.greeting to be a non-None "
            "GreetingConfig with enabled=True (see "
            "config/greeting_pilot.yaml.example for the canonical overlay)"
        )
        raise ValueError(msg)

    engine = voice_engine if voice_engine is not None else build_voice_engine(cfg)
    if engine is None:
        msg = (
            "build_greeter could not obtain a voice engine — cfg.voice.enabled "
            "must be True for the greeting subsystem to play audio"
        )
        raise ValueError(msg)

    # Source the intensity threshold from VoiceConfig so an operator
    # tuning ``voice.intensity_threshold`` doesn't get silently
    # shadowed by the rocky_transform default (CLAUDE.md "no hardcoded
    # values"). Concrete ``Greeter`` import deferred per Invariant 1.
    from mousedroid.voice.greeting import Greeter

    intensity_threshold = cfg.voice.intensity_threshold

    _log.info(
        "greeter_built",
        names_count=len(cfg.greeting.names),
        pre_chirp_event=cfg.greeting.pre_chirp_event or "(none)",
        excitement_intensity=cfg.greeting.excitement_intensity,
        intensity_threshold=intensity_threshold,
    )
    return Greeter(engine, cfg.greeting, intensity_threshold=intensity_threshold)


def _build_orchestrator_greeter(
    cfg: Settings,
    voice_engine: VoiceEngineProtocol | None,
) -> GreeterProtocol | None:
    """Wire the startup greeter onto the orchestrator (Issue #109).

    Returns a :class:`Greeter` when ``cfg.greeting`` is enabled AND a
    voice engine is available, reusing the orchestrator's already-built
    ``voice_engine`` so a single engine instance serves both the greeting
    one-shot and the normal voice output (no competing start/stop). The
    orchestrator owns that engine's lifecycle.

    Returns ``None`` — never raises — when greeting is unconfigured /
    disabled or no voice engine was built, so this helper is safe to call
    unconditionally from ``build_orchestrator``. (``build_greeter`` raises
    on misconfiguration because it is the operator-tools entry point; the
    orchestrator path is wiring, where a ``None`` cleanly disables the
    seam.)

    Args:
        cfg: Root settings.
        voice_engine: The orchestrator's voice engine (or ``None`` when
            ``cfg.voice.enabled`` is ``False``).

    Returns:
        A :class:`Greeter` conforming to :class:`GreeterProtocol`, or
        ``None``.
    """
    if cfg.greeting is None or not cfg.greeting.enabled or voice_engine is None:
        return None
    greeter = build_greeter(cfg, voice_engine=voice_engine)
    _log.info("orchestrator_startup_greeter_wired", fire_on_startup=cfg.greeting.fire_on_startup)
    return greeter


def build_voice_engine(
    cfg: Settings,
    speaker: SpeakerProtocol | None = None,
    failure_recorder: FailureRecorder | None = None,
    clock: ClockProtocol | None = None,
    *,
    metrics: MetricsRegistry | None = None,
) -> VoiceEngineProtocol | None:
    """Build Rocky voice engine based on config.

    Args:
        cfg: Root settings.
        speaker: Pre-built speaker driver (built if not provided).
        failure_recorder: Optional ``FailureRecorder`` injected so the voice
            engine can emit observable signals (Prometheus counter +
            structured log) whenever events are dropped due to per-event
            cooldown or token-bucket backpressure. Defaults to a no-op when
            unspecified.
        clock: Optional :class:`ClockProtocol`. Defaults to
            :class:`RealClock` (production); tests inject
            :class:`MockClock` so cooldown / token-bucket logic is
            deterministic without wall-clock waits.
        metrics: Optional shared metrics registry, threaded keyword-only into
            the speaker, TTS, and engine so speaker degradations and TTS
            synthesis failures surface the ``voice_speaker_degraded_total`` /
            ``voice_tts_synthesize_failures_total`` counters on ``/metrics``.

    Returns:
        Voice engine conforming to ``VoiceEngineProtocol``, or None if disabled.
    """
    if not cfg.voice.enabled:
        _log.info("voice_engine_disabled")
        return None

    if speaker is None:
        speaker = build_speaker(cfg, metrics=metrics)
    if speaker is None:
        _log.warning("voice_engine_disabled_no_speaker")
        return None

    tts: MockTTS | PiperTTS
    try:
        if cfg.mock_hardware:
            from mousedroid.voice.mock_tts import MockTTS

            tts = MockTTS(cfg.voice)
        else:
            from mousedroid.voice.tts import PiperTTS

            tts = PiperTTS(cfg.voice, metrics=metrics)
    except Exception:
        _log.warning("voice_engine_tts_init_failed", exc_info=True)
        return None

    # Validate sample rate compatibility
    if speaker.sample_rate != cfg.voice.tts_sample_rate:
        _log.warning(
            "voice_engine_sample_rate_mismatch",
            speaker_rate=speaker.sample_rate,
            tts_rate=cfg.voice.tts_sample_rate,
        )
        return None

    from mousedroid.voice.rocky import RockyVoiceEngine

    engine = RockyVoiceEngine(
        cfg.voice,
        speaker,
        tts,
        failure_recorder=failure_recorder,
        clock=clock,
        metrics=metrics,
    )
    _log.info(
        "voice_engine_built",
        personality=cfg.voice.personality,
        cooldown_s=cfg.voice.cooldown_s,
        sample_rate=speaker.sample_rate,
        resolved_model_path=cfg.voice.resolved_tts_model_path(),
    )
    return engine
