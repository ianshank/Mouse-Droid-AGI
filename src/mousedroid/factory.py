"""Platform factory functions — build all components via dependency injection.

Factory functions eliminate platform branching. Each ``build_*()`` function
returns the correct implementation based on ``Settings``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mousedroid.cloud.protocol import (
    ENGINE_TYPE_POLICY,
    ENGINE_TYPE_WORLD_MODEL,
)
from mousedroid.common.imports import module_available
from mousedroid.comms.protocol import ESP32CommProtocol
from mousedroid.hardware.protocols import (
    AudioProtocol,
    DistanceSensorProtocol,
    FaceDisplayProtocol,
    LidarProtocol,
    SpeakerProtocol,
    VisionProtocol,
)
from mousedroid.health.watchdog import WatchdogProtocol
from mousedroid.llm_gateway.protocol import LLMGatewayProtocol
from mousedroid.logging.setup import get_logger
from mousedroid.safety.projector_protocol import SafetyActionProjectorProtocol
from mousedroid.safety.protocol import SafetyMonitorProtocol
from mousedroid.security.injection_filter import (
    PromptInjectionFilterProtocol,
    RegexInjectionFilter,
)
from mousedroid.vla.policy import VLAPolicyProtocol
from mousedroid.voice.protocol import VoiceEngineProtocol

if TYPE_CHECKING:
    from mousedroid.agents.base import AgentProtocol
    from mousedroid.arm.protocols import (
        ArmControllerProtocol,
        ArmDriverProtocol,
        ArmEnvironmentProtocol,
        ArmPerceptionProtocol,
        ArmPlannerProtocol,
    )
    from mousedroid.cloud.protocol import (
        CloudExperienceExporterProtocol,
        CloudTelemetrySinkProtocol,
        PendingWeightUpdate,
        WeightUpdatePollerProtocol,
    )
    from mousedroid.cognitive.bdi_model import NeuralBDI
    from mousedroid.cognitive.cognitive_core import CognitiveCore
    from mousedroid.common.time.protocol import ClockProtocol
    from mousedroid.common.tools.registry import ToolRegistry
    from mousedroid.config.schema import ESP32Config, LLMConfig, Settings, UltrasonicConfig
    from mousedroid.curiosity.protocol import CuriosityProtocol
    from mousedroid.efficiency.tensorrt import TensorRTCompilerProtocol
    from mousedroid.experience.logger import ExperienceLogger
    from mousedroid.hardware.accelerator.hailo_runtime import HailoRuntimeProtocol
    from mousedroid.harness.approval.protocol import ApprovalGateProtocol
    from mousedroid.harness.protocol import TaskTrackerProtocol
    from mousedroid.health.monitor import HealthMonitor
    from mousedroid.llm_gateway.mission_parser import MissionParserProtocol
    from mousedroid.mcp.protocol import MCPServerProtocol
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
    from mousedroid.training.replay import ReplayReaderProtocol
    from mousedroid.voice.greeting import Greeter
    from mousedroid.voice.mock_tts import MockTTS
    from mousedroid.voice.tts import PiperTTS
    from mousedroid.world_model.protocol import WorldModelProtocol


_log = get_logger(__name__)


def build_esp32_driver(cfg: Settings) -> ESP32CommProtocol:
    """Build ESP32 communication driver based on config.

    Wraps the underlying driver with circuit breaker + retry for
    fault tolerance.  The wrapper implements ``ESP32CommProtocol``
    so the orchestrator doesn't need to know about it.

    When ``cfg.usbc_discovery`` is enabled and declares a ``rover_esp32``
    endpoint, that endpoint's live by-id path supersedes the literal
    ``cfg.esp32.serial_port``. This keeps the config stable across rover
    swaps (CP2102N serial numbers differ per unit).

    Args:
        cfg: Root settings.

    Returns:
        ESP32 driver conforming to ``ESP32CommProtocol``.
    """
    inner: ESP32CommProtocol

    # ``cfg.esp32.enabled = False`` is the schema-driven dev escape hatch
    # for running the orchestrator on hardware where the ESP32 isn't
    # plugged in (e.g. Jetson + camera + LiDAR + Hailo for dashboard
    # verification — see PR #104 harden-2). The mock driver short-circuits
    # connect / send_velocity / emergency_stop without touching any serial
    # port, so the orchestrator's start() doesn't crash and tick rate isn't
    # dragged down by ResilientESP32Driver's open-circuit timeouts.
    if cfg.mock_hardware or not cfg.esp32.enabled:
        from mousedroid.comms.mock_driver import MockESP32Driver

        inner = MockESP32Driver(cfg.esp32)
    elif cfg.esp32.protocol == "serial":
        from mousedroid.comms.serial_driver import SerialESP32Driver

        esp32_cfg = _resolve_esp32_serial_via_usbc_discovery(cfg)
        inner = SerialESP32Driver(esp32_cfg)
    else:
        from mousedroid.comms.wifi_driver import WiFiESP32Driver

        inner = WiFiESP32Driver(cfg.esp32)

    from mousedroid.resilience.resilient_driver import ResilientESP32Driver

    return ResilientESP32Driver(inner, cfg.retry, cfg.circuit_breaker)


def _resolve_esp32_serial_via_usbc_discovery(cfg: Settings) -> ESP32Config:
    """Override ``esp32.serial_port`` with the live rover_esp32 by-id path.

    Returns the original ESP32Config when discovery is disabled, the
    ``rover_esp32`` endpoint is absent, or the literal serial_port path
    already exists on disk (an exact match wins — avoids surprise
    overrides when the operator pinned a specific path).
    """
    from pathlib import Path as _Path

    if cfg.usbc_discovery is None or not cfg.usbc_discovery.enabled:
        return cfg.esp32
    if _Path(cfg.esp32.serial_port).exists():
        return cfg.esp32

    from mousedroid.diagnostics.usbc import resolve_endpoint

    resolved = resolve_endpoint(cfg.usbc_discovery, "rover_esp32")
    if resolved is None:
        _log.warning(
            "esp32_serial_port_unresolved",
            literal=cfg.esp32.serial_port,
            hint="usbc_discovery has no rover_esp32 endpoint matching the bus",
        )
        return cfg.esp32

    _log.info(
        "esp32_serial_port_overridden",
        literal=cfg.esp32.serial_port,
        resolved=str(resolved),
    )
    return cfg.esp32.model_copy(update={"serial_port": str(resolved)})


def build_camera(
    cfg: Settings,
    hailo_runtime: HailoRuntimeProtocol | None = None,
) -> VisionProtocol:
    """Build camera driver based on config.

    When a Hailo-8 runtime is provided, it is passed through to the
    camera constructor so that ``build_feature_extractor`` can select
    the :class:`HailoFeatureExtractor` backend at construction time.

    Args:
        cfg: Root settings.
        hailo_runtime: Optional Hailo-8 runtime for accelerated feature extraction.

    Returns:
        Camera driver conforming to ``VisionProtocol``.
    """
    if cfg.mock_hardware:
        from mousedroid.hardware.camera.mock_camera import MockCamera

        return MockCamera(cfg.camera)

    if cfg.camera.backend == "jetson_csi":
        from mousedroid.hardware.camera.jetson_csi import JetsonCSICamera

        return JetsonCSICamera(cfg.camera, hailo_runtime=hailo_runtime)

    if cfg.camera.backend == "picamera2":
        from mousedroid.hardware.camera.imx500 import IMX500Camera

        return IMX500Camera(cfg.camera, hailo_runtime=hailo_runtime)

    # auto: prefer picamera2 when its stack is installed, else fall back to jetson_csi
    if module_available("picamera2"):
        from mousedroid.hardware.camera.imx500 import IMX500Camera

        return IMX500Camera(cfg.camera, hailo_runtime=hailo_runtime)

    from mousedroid.hardware.camera.jetson_csi import JetsonCSICamera

    return JetsonCSICamera(cfg.camera, hailo_runtime=hailo_runtime)


def build_distance_sensor(cfg: Settings) -> DistanceSensorProtocol:
    """Build distance sensor driver based on config.

    Args:
        cfg: Root settings.

    Returns:
        Distance sensor conforming to ``DistanceSensorProtocol``.
    """
    if cfg.mock_hardware:
        from mousedroid.config.schema import UltrasonicConfig as UltraCfg
        from mousedroid.hardware.sensors.mock_ultrasonic import MockUltrasonic

        ultrasonic_cfg: UltrasonicConfig = cfg.ultrasonic or UltraCfg.model_validate(
            {"trigger_pin": 0, "echo_pin": 0}
        )
        return MockUltrasonic(ultrasonic_cfg)

    if cfg.ultrasonic is None:
        msg = "ultrasonic config required for real hardware"
        raise ValueError(msg)

    from mousedroid.hardware.sensors.ultrasonic import HcSr04

    return HcSr04(cfg.ultrasonic)


def build_microphone(cfg: Settings) -> AudioProtocol | None:
    """Build USB microphone driver based on config.

    Args:
        cfg: Root settings.

    Returns:
        Microphone driver conforming to ``AudioProtocol``, or None if disabled.
    """
    if cfg.microphone is None or not cfg.microphone.enabled:
        return None

    if cfg.mock_hardware:
        from mousedroid.hardware.audio.mock_microphone import MockMicrophone

        return MockMicrophone(cfg.microphone)

    from mousedroid.hardware.audio.usb_microphone import UsbMicrophone

    return UsbMicrophone(cfg.microphone)


def build_face_display(cfg: Settings) -> FaceDisplayProtocol | None:
    """Build the SSD1306 face-display driver based on config.

    Returns ``None`` when the subsystem is omitted from config or explicitly
    disabled, mirroring the other optional-hardware factories. The factory
    eagerly probes the I²C bus + address so that
    ``fallback_to_mock_on_error`` covers both:

    * import failures (``luma.oled`` / ``smbus2`` unavailable), and
    * runtime probe failures (panel disconnected, wrong address, missing
      I²C device node).

    When the probe fails and ``fallback_to_mock_on_error=True``, returns a
    :class:`MockFaceDriver` so the orchestrator can still come up. When the
    flag is ``False``, the failure is re-raised.

    Args:
        cfg: Root settings.

    Returns:
        Driver conforming to :class:`FaceDisplayProtocol`, or ``None``.
    """
    if cfg.face_display is None or not cfg.face_display.enabled:
        return None

    from mousedroid.hardware.display.mock_face_driver import MockFaceDriver

    if cfg.mock_hardware:
        _log.info("face_display_mock_built")
        return MockFaceDriver(cfg.face_display)

    try:
        from mousedroid.hardware.display.ssd1306_face_driver import SSD1306FaceDriver

        SSD1306FaceDriver.probe(cfg.face_display)
        _log.info(
            "face_display_real_built",
            i2c_bus=cfg.face_display.i2c_bus,
            i2c_address=cfg.face_display.i2c_address,
        )
        return SSD1306FaceDriver(cfg.face_display)
    except (ImportError, OSError):
        # ImportError → luma.oled/smbus2 missing; OSError → bus/addr/panel
        # unreachable.  All other exceptions propagate so programming errors
        # are never silently swallowed.
        if cfg.face_display.fallback_to_mock_on_error:
            _log.warning(
                "face_display_falling_back_to_mock",
                i2c_bus=cfg.face_display.i2c_bus,
                i2c_address=cfg.face_display.i2c_address,
                exc_info=True,
            )
            return MockFaceDriver(cfg.face_display)
        raise


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


def build_speaker(cfg: Settings) -> SpeakerProtocol | None:
    """Build USB speaker driver based on config.

    Args:
        cfg: Root settings.

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
    return UsbSpeaker(cfg.speaker)


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


def build_voice_engine(
    cfg: Settings,
    speaker: SpeakerProtocol | None = None,
    failure_recorder: FailureRecorder | None = None,
    clock: ClockProtocol | None = None,
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

    Returns:
        Voice engine conforming to ``VoiceEngineProtocol``, or None if disabled.
    """
    if not cfg.voice.enabled:
        _log.info("voice_engine_disabled")
        return None

    if speaker is None:
        speaker = build_speaker(cfg)
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

            tts = PiperTTS(cfg.voice)
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
    )
    _log.info(
        "voice_engine_built",
        personality=cfg.voice.personality,
        cooldown_s=cfg.voice.cooldown_s,
        sample_rate=speaker.sample_rate,
        resolved_model_path=cfg.voice.resolved_tts_model_path(),
    )
    return engine


def build_world_model(cfg: Settings) -> WorldModelProtocol:
    """Build world model for configured platform.

    Dispatch order:

    1. ``cfg.world_model.engine == "onnx_trt"`` — construct
       :class:`~mousedroid.world_model.dual_stream_rssm_onnx.DualStreamRSSMOnnx`
       backed by the exported ``.onnx`` at ``cfg.world_model.onnx_path``.
       The runtime is constructed cheaply (no ORT import at this point)
       and warms up on first ``observe_step()`` call. Requires
       ``cfc_hidden_dim > 0`` because the ONNX export is built from
       :class:`DualStreamRSSM`.
    2. ``cfg.world_model.engine == "torch"`` (default) AND
       ``cfc_hidden_dim > 0`` — construct :class:`DualStreamRSSM`.
    3. Fallback — construct the classic :class:`~mousedroid.world_model.rssm.RSSM`.

    Default behavior (``engine="torch"``) is byte-identical to pre-B2:
    existing ``config/*.yaml`` files that omit the ``world_model:`` block
    load unchanged.

    Args:
        cfg: Root settings.

    Returns:
        World model conforming to ``WorldModelProtocol``.
    """
    engine = cfg.world_model.engine
    if engine == "onnx_trt":
        return _build_onnx_world_model(cfg)
    if engine != "torch":
        # Pydantic Literal["torch", "onnx_trt"] should catch this earlier,
        # but defend in depth so dynamic instantiation doesn't drift past
        # the dispatcher.
        msg = f"Unknown world_model.engine {engine!r}"
        raise ValueError(msg)

    if cfg.model.cfc_hidden_dim > 0:
        try:
            from mousedroid.world_model.dual_stream_rssm import DualStreamRSSM
        except ImportError:
            _log.warning(
                "dual_stream_unavailable_falling_back_to_rssm",
                reason="ncps package not installed (pip install ncps)",
                requested_cfc_dim=cfg.model.cfc_hidden_dim,
            )
        else:
            _log.info(
                "world_model_engine_selected",
                engine="torch",
                gru_dim=cfg.model.hidden_dim,
                cfc_dim=cfg.model.cfc_hidden_dim,
            )
            return DualStreamRSSM(cfg.model)

    from mousedroid.world_model.rssm import RSSM

    return RSSM(cfg.model)


def _build_onnx_world_model(cfg: Settings) -> WorldModelProtocol:
    """Construct the ONNX runtime world model.

    Resolves ``cfg.world_model.onnx_path`` (filesystem-first, HF Hub
    fallback) and hands it to :class:`DualStreamRSSMOnnx`. The runtime
    is constructed lazily — ``onnxruntime`` is not imported here.
    """
    if cfg.model.cfc_hidden_dim <= 0:
        msg = (
            "world_model.engine='onnx_trt' requires model.cfc_hidden_dim > 0 "
            "(the ONNX export is built from DualStreamRSSM, which requires "
            "the CfC stream). Set cfc_hidden_dim in your config YAML or "
            "switch to engine='torch'."
        )
        raise ValueError(msg)

    from mousedroid.world_model.composite import CompositeWorldModel
    from mousedroid.world_model.dual_stream_rssm import DualStreamRSSM
    from mousedroid.world_model.dual_stream_rssm_onnx import DualStreamRSSMOnnx

    model_path = _resolve_world_model_onnx_path(cfg)

    # observe_step path: ONNX-accelerated (the hot 30Hz tick benefit).
    observe_engine = DualStreamRSSMOnnx(
        model_path=model_path,
        cfg=cfg.model,
        warmup_iterations=cfg.world_model.onnx_warmup_iterations,
    )
    # imagine_step + get_safety_trace path: PyTorch DualStreamRSSM. The
    # ONNX export (B2 Story 1) is scoped to observe_step only, so MCTS
    # rollouts and the safety monitor's CfC inspection need the PyTorch
    # graph. Both engines share the same ModelConfig so dimensions stay
    # consistent across the composition boundary.
    imagine_engine = DualStreamRSSM(cfg.model)
    imagine_engine.train(False)

    _log.info(
        "world_model_engine_selected",
        engine="onnx_trt",
        model_path=str(model_path),
        cfc_dim=cfg.model.cfc_hidden_dim,
        composite=True,
        observe_engine=type(observe_engine).__name__,
        imagine_engine=type(imagine_engine).__name__,
    )
    return CompositeWorldModel(
        observe_engine=observe_engine,
        imagine_engine=imagine_engine,
    )


def _resolve_world_model_onnx_path(cfg: Settings) -> Path:
    """Resolve the .onnx artifact path for the ONNX runtime engine.

    Resolution order:

    1. ``cfg.world_model.onnx_path`` when set — use it directly. If the
       file is missing, the runtime's :meth:`warmup` will raise
       ``FileNotFoundError`` so operators get a clear error from the
       runtime, not a confusing ``hf_hub_download`` traceback.
    2. HF Hub download via
       ``cfg.world_model.onnx_repo_id``/``cfg.world_model.onnx_filename``.
       Mirrors the [vla] pattern at ``_build_distilled_onnx_vla``.
       Cached under ``weights/dual_stream_rssm/`` so the same file is
       reused across runs without re-downloading.
    """
    explicit = cfg.world_model.onnx_path
    if explicit is not None:
        return Path(explicit)

    # HF Hub auto-download fallback. Reuses the same
    # ``download_weights_from_huggingface`` helper the VLA path uses, so
    # retries / auth tokens / progress bars work identically.
    from mousedroid.utils.weights_manager import (
        download_weights_from_huggingface,
    )

    # Operator-tunable per deployment via ``cfg.world_model.onnx_cache_dir``
    # (default ``weights/dual_stream_rssm``). Mirrors the VLA pattern at
    # ``_build_distilled_onnx_vla`` so Jetson deployments can repoint both
    # caches under ``/opt/mousedroid/weights/...`` in one place.
    cache_dir = Path(cfg.world_model.onnx_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / cfg.world_model.onnx_filename

    if model_path.is_file():
        _log.info(
            "world_model_onnx_cache_hit",
            cache_path=str(model_path),
            repo_id=cfg.world_model.onnx_repo_id,
        )
        return model_path

    _log.info(
        "world_model_onnx_download_start",
        repo_id=cfg.world_model.onnx_repo_id,
        filename=cfg.world_model.onnx_filename,
        cache_dir=str(cache_dir),
    )
    success = download_weights_from_huggingface(
        repo_id=cfg.world_model.onnx_repo_id,
        filenames=[cfg.world_model.onnx_filename],
        cache_dir=cache_dir,
        # Force flat layout so model_path.is_file() check succeeds.
        # Without local_dir, hf_hub_download uses its blob/snapshot
        # cache layout and the file would not be at the expected path.
        local_dir=cache_dir,
    )
    if not success or not model_path.is_file():
        msg = (
            f"failed to download world-model ONNX artifact "
            f"({cfg.world_model.onnx_repo_id}/{cfg.world_model.onnx_filename}) "
            f"into {cache_dir}. Set world_model.onnx_path to a local path "
            f"or run scripts/export_dual_stream_rssm_onnx.py --push-to-hf "
            f"to publish a fresh artifact first."
        )
        raise FileNotFoundError(msg)
    _log.info(
        "world_model_onnx_downloaded",
        path=str(model_path),
        repo_id=cfg.world_model.onnx_repo_id,
    )
    return model_path


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
            the ``llama_cpp`` and ``anthropic`` backends (both forward NL
            commands that warrant local filtering — the GGUF model runs the
            command verbatim, and ``anthropic`` ships it to a third-party
            cloud). The ``openai_compatible`` backend skips it, trusting the
            upstream provider's guardrails.
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
        return OpenAICompatibleLLMGateway(llm_cfg, metrics=metrics)

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

    The ``injection_filter`` is shared with the ``llama_cpp`` and
    ``anthropic`` backends (and both tiers of the composite); the
    ``openai_compatible`` backend skips local injection filtering because the
    upstream provider is expected to enforce its own guardrails.

    Args:
        cfg: Root settings.
        injection_filter: Optional shared :class:`PromptInjectionFilterProtocol`.
            When ``None``, each filter-aware gateway constructs its own filter
            from ``cfg.llm.injection_patterns`` (legacy behaviour); when
            supplied (the default in :func:`build_orchestrator`), the same
            filter is reused by the OpenClaw mission dispatcher.
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


def build_vla_policy(
    cfg: Settings,
    *,
    metrics: MetricsRegistry | None = None,
) -> VLAPolicyProtocol | None:
    """Build the VLA policy if configured.

    Returns ``None`` when ``cfg.vla.backend == "none"`` so callers (the
    orchestrator) can skip the VLA branch without inspecting backend
    strings.

    Backends:
        - ``"none"`` (default): VLA path disabled; returns ``None``.
        - ``"mock"`` (Phase 3a): in-tree zero-dependency :class:`MockVLA`.
        - ``"distilled_onnx"`` (Phase 3b): :class:`DistilledVLAOnnx`
          backed by ONNX Runtime. Optionally pulls weights from
          HuggingFace via
          :func:`mousedroid.utils.weights_manager.download_weights_from_huggingface`.
          Requires the ``[vla]`` extra (``onnxruntime`` /
          ``onnxruntime-gpu``); the import is deferred to
          :meth:`DistilledVLAOnnx.warmup`.

    Args:
        cfg: Root settings.
        metrics: Optional :class:`MetricsRegistry` forwarded to the chosen
            backend so ``predict()`` calls populate
            ``mousedroid_vla_inference_seconds``. ``None`` (default)
            preserves byte-identical pre-PR-A2.1 behavior.

    Returns:
        A :class:`VLAPolicyProtocol` instance, or ``None`` when disabled.

    Raises:
        ValueError: When ``vla.canned_action`` length disagrees with
            ``model.action_dim``, or the configured ``distilled_onnx``
            cache directory cannot be located and no ``model_repo_id``
            is configured for download.
    """
    backend = cfg.vla.backend
    if backend == "none":
        _log.info("vla_policy_disabled")
        return None

    import torch as _torch  # runtime import; torch is a project dependency

    action_dim = cfg.model.action_dim
    canned: _torch.Tensor | None = None
    if cfg.vla.canned_action is not None:
        if len(cfg.vla.canned_action) != action_dim:
            msg = (
                f"vla.canned_action length {len(cfg.vla.canned_action)} "
                f"!= model.action_dim ({action_dim})"
            )
            raise ValueError(msg)
        canned = _torch.tensor(cfg.vla.canned_action, dtype=_torch.float32)

    if backend == "mock":
        from mousedroid.vla.policy import MockVLA

        _log.info("vla_policy_built", backend="mock", action_dim=action_dim)
        return MockVLA(
            action_dim=action_dim,
            canned_action=canned,
            confidence=cfg.vla.confidence,
            metrics=metrics,
        )

    if backend == "distilled_onnx":
        return _build_distilled_onnx_vla(cfg, action_dim, metrics=metrics)

    msg = f"Unknown VLA backend {backend!r}"
    raise ValueError(msg)


def _build_distilled_onnx_vla(
    cfg: Settings,
    action_dim: int,
    *,
    metrics: MetricsRegistry | None = None,
) -> VLAPolicyProtocol:
    """Resolve the ONNX model and instantiate :class:`DistilledVLAOnnx`.

    Args:
        cfg: Root settings.
        action_dim: Configured action dimensionality.
        metrics: Optional :class:`MetricsRegistry` forwarded to
            :class:`DistilledVLAOnnx` so each inference call populates
            ``mousedroid_vla_inference_seconds``.

    Returns:
        An un-warmed :class:`DistilledVLAOnnx`. Warmup happens on first
        :meth:`predict` call (or eagerly if ``warmup_iterations > 0``
        and the file exists at construction time).

    Raises:
        ValueError: When neither a downloadable ``model_repo_id`` nor a
            local ``cache_dir`` containing ``model_filename`` is
            available.
    """
    from pathlib import Path as _Path

    from mousedroid.vla.policy import DistilledVLAOnnx

    # ``VLAConfig.cache_dir`` is required (Pydantic-defaulted to
    # ``"weights/vla"``); guard against an explicit ``None`` override that
    # may slip in via legacy YAML.
    if cfg.vla.cache_dir is None:
        msg = "vla.cache_dir must be set (default is 'weights/vla')"
        raise ValueError(msg)
    cache_dir = _Path(cfg.vla.cache_dir)
    model_path = cache_dir / cfg.vla.model_filename

    if not model_path.is_file():
        if cfg.vla.model_repo_id is None:
            msg = (
                f"distilled_onnx model not found at {model_path} and "
                f"vla.model_repo_id is unset; either provide a local "
                f"file or configure a HuggingFace repo for download"
            )
            raise ValueError(msg)
        from mousedroid.utils.weights_manager import (
            download_weights_from_huggingface,
        )

        cache_dir.mkdir(parents=True, exist_ok=True)
        success = download_weights_from_huggingface(
            repo_id=cfg.vla.model_repo_id,
            filenames=[cfg.vla.model_filename],
            cache_dir=cache_dir,
            # Force flat layout (cache_dir/model_filename) so the
            # ``model_path.is_file()`` check below sees the file. Without
            # ``local_dir``, hf_hub_download uses its blob/snapshot cache
            # layout and the file would not be at the expected path.
            local_dir=cache_dir,
        )
        if not success or not model_path.is_file():
            msg = (
                f"failed to download distilled_onnx weights "
                f"({cfg.vla.model_repo_id}/{cfg.vla.model_filename}) "
                f"into {cache_dir}"
            )
            raise ValueError(msg)

    policy = DistilledVLAOnnx(
        model_path=model_path,
        action_dim=action_dim,
        providers=list(cfg.vla.providers) if cfg.vla.providers is not None else None,
        h_input_name=cfg.vla.h_input_name,
        z_input_name=cfg.vla.z_input_name,
        action_output_name=cfg.vla.action_output_name,
        warmup_iterations=cfg.vla.warmup_iterations,
        confidence=cfg.vla.confidence,
        metrics=metrics,
    )
    _log.info(
        "vla_policy_built",
        backend="distilled_onnx",
        action_dim=action_dim,
        model_path=str(model_path),
    )
    return policy


def build_reward_model(
    cfg: Settings,
    *,
    metrics: MetricsRegistry | None = None,
) -> RewardModelProtocol:
    """Build the multi-objective reward model with optional VLM progress head.

    The Three Laws head is constructed inside
    :class:`MultiObjectiveRewardModel` whenever ``cfg.three_laws.enabled``.
    The Phase 4 VLM progress head is attached only when both
    ``cfg.reward.vlm_progress.enabled`` and
    ``cfg.reward.weight_vlm_progress > 0`` so that, by default, behaviour is
    byte-identical to the pre-Phase 4 reward path.

    Args:
        cfg: Root settings.
        metrics: Optional :class:`MetricsRegistry` forwarded to
            :class:`VLMProgressHead` so cache hit/miss decisions populate
            ``mousedroid_vlm_progress_cache_hits_total`` / ``..._misses_total``.
            ``None`` (default) preserves byte-identical pre-PR-A2.1 behavior.

    Returns:
        Configured reward model.
    """
    from mousedroid.reward.model import MultiObjectiveRewardModel
    from mousedroid.reward.vlm_progress import VLMProgressHead

    vlm_head: VLMProgressHead | None = None
    if cfg.reward.vlm_progress.enabled and cfg.reward.weight_vlm_progress > 0.0:
        vlm_head = VLMProgressHead(cfg.reward.vlm_progress, metrics=metrics)

    model = MultiObjectiveRewardModel(
        cfg.model,
        cfg.reward,
        law_cfg=cfg.three_laws,
        vlm_head=vlm_head,
    )
    _log.info(
        "reward_model_built",
        vlm_progress_enabled=vlm_head is not None,
        three_laws_enabled=cfg.three_laws.enabled,
    )
    return model


def build_replay_reader(
    cfg: Settings,
    *,
    metrics: MetricsRegistry | None = None,
) -> ReplayReaderProtocol:
    """Build the Phase 2 LMDB replay reader.

    Args:
        cfg: Root settings. Reads ``cfg.experience`` (LMDB path + map size)
            and respects ``cfg.training.replay.source_path`` as a path
            override when set.
        metrics: Optional :class:`MetricsRegistry` forwarded to the reader
            so each decoded record / schema-mismatch drop populates
            ``mousedroid_replay_records_total{outcome=...}``. ``None``
            (default) preserves byte-identical pre-PR-A2.1 behavior.

    Returns:
        Reader conforming to :class:`ReplayReaderProtocol`. The concrete
        type is hidden behind the protocol so callers cannot couple to
        LMDB internals (CLAUDE.md invariants 1+2).
    """
    from mousedroid.training.replay import LMDBReplayReader

    reader = LMDBReplayReader(
        cfg.experience,
        path_override=cfg.training.replay.source_path,
        debug_log_every_n=cfg.training.replay_mixer.debug_log_every_n,
        metrics=metrics,
    )
    _log.info(
        "replay_reader_built",
        path=str(reader.path),
        debug_log_every_n=cfg.training.replay_mixer.debug_log_every_n,
        metrics_enabled=metrics is not None,
    )
    return reader


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


def build_metrics_registry(cfg: Settings) -> MetricsRegistry | None:
    """Build the shared Prometheus metrics registry when metrics are enabled.

    Args:
        cfg: Root settings.

    Returns:
        Shared ``MetricsRegistry`` or ``None`` when metrics are disabled.
    """
    if not cfg.metrics.enabled:
        return None

    from mousedroid.telemetry.metrics import MetricsRegistry

    return MetricsRegistry(cfg.metrics)


def build_weight_update_poller(
    cfg: Settings,
    *,
    metrics: MetricsRegistry | None = None,
) -> WeightUpdatePollerProtocol | None:
    """Build the optional Tier C1 OTA weight-update poller (legacy single-engine shim).

    Deprecated: prefer :func:`build_weight_update_pollers` (Tier C1.2)
    which returns a ``Mapping[str, WeightUpdatePollerProtocol]`` keyed by
    ``engine_type`` and supports a second world-model poller alongside
    the policy poller. Retained for backwards compatibility with external
    callers for one minor-version window.

    Returns ``None`` (poller disabled) when
    ``cfg.cloud.weight_update.poll_interval_s <= 0.0`` — the default — so
    deployments without OTA configured produce byte-identical pre-Tier-C1
    behavior. Always builds a single ``policy`` poller when polling is
    enabled; world-model OTA is now reachable via the plural
    :func:`build_weight_update_pollers` factory + the
    ``cfg.cloud.weight_update.world_model_enabled`` schema flag (Tier
    C1.2). New callers should migrate.

    Args:
        cfg: Root settings.
        metrics: Shared metrics registry; forwarded to the poller for
            download / mismatch / latency observability.

    Returns:
        A :class:`WeightUpdatePollerProtocol` implementation or ``None``.
    """
    if cfg.cloud.weight_update.poll_interval_s <= 0.0:
        return None

    from mousedroid.cloud.weight_update_poller import HuggingFaceWeightUpdatePoller

    poller = HuggingFaceWeightUpdatePoller(
        cfg.cloud.weight_update,
        repo_id=cfg.cloud.weight_update.policy_repo_id,
        filename=cfg.cloud.weight_update.policy_filename,
        engine_type=ENGINE_TYPE_POLICY,
        metrics=metrics,
    )
    _log.info(
        "weight_update_poller_built",
        repo_id=cfg.cloud.weight_update.policy_repo_id,
        poll_interval_s=cfg.cloud.weight_update.poll_interval_s,
    )
    return poller


def build_weight_update_pollers(
    cfg: Settings,
    *,
    metrics: MetricsRegistry | None = None,
) -> Mapping[str, WeightUpdatePollerProtocol]:
    """Build a mapping of ``engine_type`` -> poller (Tier C1.2).

    Returns ``{}`` when ``cfg.cloud.weight_update.poll_interval_s <= 0.0``,
    preserving byte-identical pre-Tier-C1 behaviour. Always includes the
    ``"policy"`` entry when polling is enabled; includes ``"world_model"``
    only when ``cfg.cloud.weight_update.world_model_enabled is True``.

    Dict insertion order is ``policy`` -> ``world_model`` so the orchestrator
    consumes pending updates deterministically.

    Args:
        cfg: Root settings.
        metrics: Shared metrics registry; forwarded to each poller for
            download / mismatch / latency observability.

    Returns:
        Mapping from ``engine_type`` to its
        :class:`WeightUpdatePollerProtocol` implementation. Empty when OTA
        polling is disabled. Insertion order is guaranteed to be ``policy``
        before ``world_model`` so the orchestrator consumes updates
        deterministically.
    """
    if cfg.cloud.weight_update.poll_interval_s <= 0.0:
        return {}

    from pathlib import Path as _Path

    from mousedroid.cloud.weight_update_poller import HuggingFaceWeightUpdatePoller

    # Per-engine cache subdirectory layout (Copilot MED): both pollers
    # download a ``sha256.txt`` manifest into their cache dir each cycle.
    # When both share the same root, the world-model poller's manifest
    # writer races the policy poller's writer and produces spurious
    # mismatches. Giving each poller a per-engine subdir under the
    # configured root preserves the operator-facing
    # ``cfg.cloud.weight_update.cache_dir`` knob (root unchanged) while
    # eliminating the collision. The subdir name reuses the typed
    # ``EngineType`` literal so a future engine addition only needs to
    # extend the enum.
    cache_root = _Path(cfg.cloud.weight_update.cache_dir)
    pollers: dict[str, WeightUpdatePollerProtocol] = {
        ENGINE_TYPE_POLICY: HuggingFaceWeightUpdatePoller(
            cfg.cloud.weight_update,
            repo_id=cfg.cloud.weight_update.policy_repo_id,
            filename=cfg.cloud.weight_update.policy_filename,
            engine_type=ENGINE_TYPE_POLICY,
            metrics=metrics,
            cache_dir_override=cache_root / ENGINE_TYPE_POLICY,
        ),
    }
    if cfg.cloud.weight_update.world_model_enabled:
        pollers[ENGINE_TYPE_WORLD_MODEL] = HuggingFaceWeightUpdatePoller(
            cfg.cloud.weight_update,
            repo_id=cfg.cloud.weight_update.world_model_repo_id,
            filename=cfg.cloud.weight_update.world_model_filename,
            engine_type=ENGINE_TYPE_WORLD_MODEL,
            metrics=metrics,
            cache_dir_override=cache_root / ENGINE_TYPE_WORLD_MODEL,
        )
    _log.info(
        "weight_update_pollers_built",
        engines=list(pollers.keys()),
        poll_interval_s=cfg.cloud.weight_update.poll_interval_s,
        cache_root=str(cache_root),
    )
    return pollers


def build_weight_update_loader(
    cfg: Settings,
) -> Callable[[PendingWeightUpdate], object] | None:
    """Build the optional Tier C1 OTA artifact loader.

    The loader is invoked by the orchestrator inside
    ``_apply_pending_weight_update`` to materialise the downloaded artifact
    into a live engine BEFORE the reference swap.

    Returns ``None`` when the OTA poller is disabled or when no production
    loader is wired (the test suite injects its own loader). When the
    poller IS enabled but the loader returns ``None``, the orchestrator
    emits ``cloud_weight_update_swap_skipped_no_loader`` and leaves the
    live model untouched — operators decide what to do.

    Returns:
        Callable that maps :class:`PendingWeightUpdate` to a new engine
        object, or ``None``.
    """
    if cfg.cloud.weight_update.poll_interval_s <= 0.0:
        return None
    # Production loader wiring is engine-specific (ONNX runtime / TensorRT
    # session reload). Tier C1 ships the seam; the operator pulls the
    # concrete loader through configuration in a follow-up PR.
    return None


def build_failure_recorder(
    cfg: Settings,
    metrics: MetricsRegistry | None = None,
) -> FailureRecorder:
    """Build a failure recorder wired to the given metrics registry.

    Returns a :class:`~mousedroid.telemetry.failure_recorder.PrometheusFailureRecorder`
    when *metrics* is non-None (telemetry enabled), otherwise a
    :class:`~mousedroid.telemetry.failure_recorder.NullFailureRecorder`.

    Args:
        cfg: Root settings (reserved for future per-subsystem gating).
        metrics: Shared metrics registry, or ``None`` when telemetry is
            disabled.  Pass the result of :func:`build_metrics_registry`.

    Returns:
        A ``FailureRecorder`` implementation appropriate for the environment.
    """
    from mousedroid.telemetry.failure_recorder import NullFailureRecorder, PrometheusFailureRecorder

    if metrics is not None:
        _log.debug("failure_recorder_built", backend="prometheus")
        return PrometheusFailureRecorder(metrics)

    _log.debug("failure_recorder_built", backend="null")
    return NullFailureRecorder()


def build_safety_monitor(cfg: Settings) -> SafetyMonitorProtocol:
    """Build safety monitor for configured platform.

    Args:
        cfg: Root settings.

    Returns:
        Safety monitor conforming to ``SafetyMonitorProtocol``.
    """
    from mousedroid.safety.monitor import MouseDroidSafetyMonitor

    return MouseDroidSafetyMonitor(cfg.safety)


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


def build_safety_projector(
    cfg: Settings,
    *,
    metrics: MetricsRegistry | None = None,
) -> SafetyActionProjectorProtocol | None:
    """Build the optional geometric safety action projector (Tier C2 / C2.1).

    Returns ``None`` when ``cfg.safety.projector.enabled`` is ``False`` —
    the orchestrator skips the projection seam entirely in that case, so
    pre-C2 deployments produce byte-identical actions.

    Args:
        cfg: Root settings.
        metrics: Optional shared metrics registry. When supplied, the
            projector increments ``mousedroid_safety_action_clamps_total``
            with one of ``forward_velocity`` / ``human_proximity`` /
            ``tight_quarters`` on every materially different clamp.

    Returns:
        :class:`SafetyActionProjectorProtocol` implementation when enabled,
        ``None`` otherwise.
    """
    if not cfg.safety.projector.enabled:
        _log.debug("safety_projector_disabled")
        return None

    from mousedroid.safety.projector import GeometricSafetyProjector

    _log.info("safety_projector_built", backend="geometric")
    return GeometricSafetyProjector(cfg.safety.projector, metrics=metrics)


def build_agent(cfg: Settings, world_model: WorldModelProtocol) -> AgentProtocol:
    """Build navigation agent for configured platform.

    Args:
        cfg: Root settings.
        world_model: World model for planning.

    Returns:
        Agent conforming to ``AgentProtocol``.
    """
    from mousedroid.agents.navigation import MouseDroidNavigationAgent
    from mousedroid.world_model.mcts import MCTSPlanner

    planner = MCTSPlanner(cfg.mcts, world_model, action_dim=cfg.model.action_dim)
    return MouseDroidNavigationAgent(planner, cfg)


def _resolve_bdi_weights(cfg: Settings) -> tuple[NeuralBDI, str]:
    """Resolve BDI model weights: local, HuggingFace, or random.

    Args:
        cfg: Root settings.

    Returns:
        Tuple of ``(NeuralBDI instance, weights_source_label)``.
    """
    from pathlib import Path

    from mousedroid.cognitive.bdi_model import NeuralBDI
    from mousedroid.utils import (
        download_weights_from_huggingface,
        weights_exist_locally,
    )

    weights_dir = Path(cfg.cognitive.weights_dir)
    bdi_filenames = ["belief.npz", "desire.npz", "intention.npz", "affect.npz"]

    if weights_exist_locally(weights_dir, bdi_filenames):
        _log.info("cognitive_core_loading_local_weights", weights_dir=str(weights_dir))
        return NeuralBDI(weights_dir=weights_dir), "local"

    if cfg.cognitive.auto_download:
        success = download_weights_from_huggingface(
            repo_id=cfg.cognitive.huggingface_repo,
            filenames=bdi_filenames,
            cache_dir=weights_dir,
            subfolder=cfg.cognitive.huggingface_subfolder,
            local_dir=weights_dir.parent,
            max_retries=cfg.cognitive.download_max_retries,
            backoff_base=cfg.cognitive.download_backoff_base,
        )
        if success:
            _log.info(
                "cognitive_core_loaded_from_huggingface",
                repo_id=cfg.cognitive.huggingface_repo,
                weights_dir=str(weights_dir),
            )
            return NeuralBDI(weights_dir=weights_dir), "huggingface"

    _log.warning(
        "weights_not_found_using_random_initialization",
        weights_dir=str(weights_dir),
        auto_download=cfg.cognitive.auto_download,
    )
    return NeuralBDI(), "random"


def build_cognitive_core(cfg: Settings) -> CognitiveCore:
    """Build cognitive core with optional weight loading from HuggingFace.

    Args:
        cfg: Root settings.

    Returns:
        Fully configured ``CognitiveCore``.
    """
    from mousedroid.cognitive.cognitive_core import CognitiveCore
    from mousedroid.cognitive.constitutional_rl import (
        ConstitutionalChecker,
        ConstitutionalRLConfig,
        PolicyMLP,
    )
    from mousedroid.cognitive.metacognitive import MetacognitiveModel

    _log.info(
        "cognitive_core_init_starting",
        weights_dir=str(cfg.cognitive.weights_dir),
        auto_download=cfg.cognitive.auto_download,
    )

    bdi, weights_source = _resolve_bdi_weights(cfg)

    policy = PolicyMLP(
        action_dim=cfg.model.action_dim,
        input_dim=cfg.model.belief_dim,
    )
    core = CognitiveCore(
        bdi=bdi,
        metacog=MetacognitiveModel(
            n_capabilities=cfg.metacognitive.n_capabilities,
            loop_score_scale=cfg.metacognitive.loop_score_scale,
        ),
        checker=ConstitutionalChecker(
            config=ConstitutionalRLConfig(
                speed_ceiling_mps=cfg.safety.max_velocity_mps,
                battery_min_v=cfg.safety.battery_critical_v,
            ),
        ),
        policy=policy,
    )
    _log.info(
        "cognitive_core_initialized",
        weights_source=weights_source,
        belief_dim=cfg.model.belief_dim,
        desire_dim=cfg.model.desire_dim,
        intention_classes=cfg.model.intention_classes,
    )
    return core


def build_telemetry_publisher(cfg: Settings) -> TelemetryPublisherProtocol | None:
    """Build telemetry publisher if telemetry is enabled.

    Args:
        cfg: Root settings.

    Returns:
        ``TelemetryPublisher`` or ``None`` if telemetry disabled.
    """
    if not cfg.telemetry.enabled:
        return None
    from mousedroid.telemetry.publisher import TelemetryPublisher

    _log.info("telemetry_publisher_built", publish_hz=cfg.telemetry.publish_hz)
    return TelemetryPublisher(cfg.telemetry)


def build_telemetry_server(
    cfg: Settings,
    publisher: TelemetryPublisherProtocol | None,
    health_monitor: HealthMonitor,
    log_buffer: LogRingBuffer | None = None,
    metrics_registry: MetricsRegistry | None = None,
    camera: VisionProtocol | None = None,
    mission_dispatcher: MissionDispatcherProtocol | None = None,
) -> TelemetryServerProtocol | None:
    """Build telemetry server if telemetry is enabled.

    Args:
        cfg: Root settings.
        publisher: Telemetry publisher to consume frames from.
        health_monitor: Health monitor for health endpoint.
        log_buffer: Optional log ring buffer for log streaming.
        metrics_registry: Optional shared metrics registry reused by other runtime components.
        camera: Optional vision driver; used as a raw-frame source for
            the MJPEG ``/camera/stream`` endpoint when it also implements
            :class:`RawFrameSourceProtocol`.
        mission_dispatcher: Optional :class:`MissionDispatcherProtocol`.
            When supplied together with an enabled ``cfg.openclaw``, the
            ``POST /api/v1/mission`` endpoint is registered.

    Returns:
        ``TelemetryServer`` or ``None`` if telemetry disabled.
    """
    if not cfg.telemetry.enabled or publisher is None:
        return None

    # PR #4: when mock_hardware is on, prefer building the real
    # aiohttp server bound to localhost so the dashboard is exercisable
    # end-to-end without rover hardware. ``force_real_server=True``
    # (legacy override) still wins. ``mock_force_real_when_enabled=False``
    # restores the no-op MockTelemetryServer for tests that prefer it.
    if cfg.mock_hardware and not cfg.telemetry.force_real_server:
        if not cfg.telemetry.mock_force_real_when_enabled:
            from mousedroid.telemetry.mock_server import MockTelemetryServer

            _log.info("telemetry_mock_server_built")
            return MockTelemetryServer()
        _log.info("telemetry_real_server_in_mock_mode")

    # D4: validate bearer token is present in env when auth is enabled.
    auth_cfg = cfg.telemetry.auth
    if auth_cfg is not None and auth_cfg.auth_enabled:
        import os

        from mousedroid.telemetry.exceptions import TelemetryConfigError

        token = os.environ.get(auth_cfg.token_env_var, "")
        if not token:
            raise TelemetryConfigError(
                f"telemetry auth_enabled=True but ${auth_cfg.token_env_var} is unset or empty; "
                "export the token or set auth_enabled=False"
            )

    shared_metrics_registry = metrics_registry
    metrics_path = cfg.metrics.path
    telemetry_metrics_path_default = type(cfg.telemetry).model_fields["metrics_path"].default
    metrics_path_default = type(cfg.metrics).model_fields["path"].default
    if (
        metrics_path == metrics_path_default
        and cfg.telemetry.metrics_path != telemetry_metrics_path_default
    ):
        metrics_path = cfg.telemetry.metrics_path

    if shared_metrics_registry is None and cfg.metrics.enabled:
        shared_metrics_registry = build_metrics_registry(cfg)

    from mousedroid.hardware.protocols import RawFrameSourceProtocol
    from mousedroid.telemetry.server import TelemetryServer

    raw_frame_source: RawFrameSourceProtocol | None = None
    if camera is not None and isinstance(camera, RawFrameSourceProtocol):
        raw_frame_source = camera

    _log.info(
        "telemetry_server_built",
        host=cfg.telemetry.host,
        port=cfg.telemetry.port,
        raw_frame_source=raw_frame_source is not None,
    )
    failure_recorder = build_failure_recorder(cfg, shared_metrics_registry)
    # PR #4: wire the raw LiDAR queue when the publisher exposes one.
    # Older custom publishers without ``get_lidar_raw_queue`` keep
    # working — the server simply registers the raw route as 503.
    lidar_raw_queue = None
    get_raw_queue = getattr(publisher, "get_lidar_raw_queue", None)
    if callable(get_raw_queue):
        lidar_raw_queue = get_raw_queue()
    return TelemetryServer(
        cfg=cfg.telemetry,
        telemetry_queue=publisher.get_queue(),
        health_monitor=health_monitor,
        log_buffer=log_buffer,
        metrics_registry=shared_metrics_registry,
        metrics_path=metrics_path,
        publisher=publisher,
        lidar_max_range_m=cfg.lidar.max_range_m if cfg.lidar is not None else None,
        raw_frame_source=raw_frame_source,
        raw_frame_hz=cfg.telemetry.raw_frame_hz,
        cloud_enabled=cfg.gcp is not None,
        mission_dispatcher=mission_dispatcher,
        openclaw_cfg=cfg.openclaw,
        failure_recorder=failure_recorder,
        lidar_raw_queue=lidar_raw_queue,
    )


def build_mock_telemetry_source(
    cfg: Settings,
    publisher: TelemetryPublisherProtocol | None,
) -> Any:
    """Build a ``MockTelemetrySource`` when running in mock mode.

    Returns ``None`` when the source is disabled or when no publisher
    is available. The returned object exposes ``start()`` / ``stop()``
    coroutines so the orchestrator can manage its lifecycle alongside
    the telemetry server.

    Args:
        cfg: Root settings.
        publisher: Telemetry publisher to push synthetic payloads into.

    Returns:
        A ``MockTelemetrySource`` instance, or ``None`` if disabled.
    """
    if not cfg.mock_hardware:
        return None
    if not cfg.telemetry.enabled:
        return None
    if publisher is None:
        return None
    if not cfg.telemetry.mock_telemetry_source_enabled:
        return None

    from mousedroid.telemetry.mock_source import MockTelemetrySource

    source = MockTelemetrySource(cfg.telemetry, publisher)
    _log.info("mock_telemetry_source_built")
    return source


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


def build_health_monitor(cfg: Settings) -> HealthMonitor:
    """Build health monitor for GPU/thermal monitoring.

    Args:
        cfg: Root settings.

    Returns:
        Configured ``HealthMonitor``.
    """
    from mousedroid.health.monitor import HealthMonitor

    _log.info("health_monitor_built")
    return HealthMonitor(cfg.health, cfg.jetson)


def build_sensor_manager(
    cfg: Settings,
    vision: VisionProtocol | None,
    distance: DistanceSensorProtocol | None,
    esp32: ESP32CommProtocol,
    microphone: AudioProtocol | None = None,
    lidar: LidarProtocol | None = None,
) -> SensorManager:
    """Build sensor manager for aggregated sensor reads.

    Args:
        cfg: Root settings.
        vision: Camera/vision protocol.
        distance: Distance sensor protocol.
        esp32: ESP32 communication protocol.
        microphone: Optional audio protocol.
        lidar: Optional LiDAR protocol.

    Returns:
        Configured ``SensorManager``.
    """
    from mousedroid.hardware.audio.feature_extractor import AudioFeatureExtractor
    from mousedroid.hardware.lidar.feature_extractor import LidarFeatureExtractor
    from mousedroid.sensing.manager import SensorManager

    if vision is None:
        # SensorManager requires a concrete VisionProtocol — raising here
        # keeps the protocol contract explicit for callers that forgot to
        # wire vision (rather than deferring to a late AttributeError on
        # first capture_features call).
        msg = "build_sensor_manager requires a non-None VisionProtocol (got None)"
        raise ValueError(msg)

    audio_extractor = build_audio_feature_extractor(cfg)
    typed_extractor: AudioFeatureExtractor | None = (
        audio_extractor if isinstance(audio_extractor, AudioFeatureExtractor) else None
    )

    lidar_extractor = build_lidar_feature_extractor(cfg)
    typed_lidar_extractor: LidarFeatureExtractor | None = (
        lidar_extractor if isinstance(lidar_extractor, LidarFeatureExtractor) else None
    )

    _log.info(
        "sensor_manager_built",
        audio_features_enabled=typed_extractor is not None,
        lidar_enabled=lidar is not None,
    )
    return SensorManager(
        vision=vision,
        distance=distance,
        esp32=esp32,
        cfg=cfg,
        microphone=microphone,
        audio_feature_extractor=typed_extractor,
        lidar=lidar,
        lidar_feature_extractor=typed_lidar_extractor,
    )


def build_audio_feature_extractor(cfg: Settings) -> object | None:
    """Build audio feature extractor if microphone is configured.

    Args:
        cfg: Root settings.

    Returns:
        ``AudioFeatureExtractor`` or ``None`` if microphone is disabled.
    """
    if cfg.microphone is None or not cfg.microphone.enabled:
        return None

    from mousedroid.hardware.audio.feature_extractor import AudioFeatureExtractor

    extractor = AudioFeatureExtractor(cfg.microphone)
    _log.info("audio_feature_extractor_built", feature_dim=extractor.feature_dim)
    return extractor


def build_lidar(cfg: Settings) -> LidarProtocol | None:
    """Build LiDAR driver based on config.

    Returns ``MockLidar`` when ``mock_hardware`` is set, otherwise wraps
    a real ``LD19LidarDriver`` with circuit breaker + retry.

    Args:
        cfg: Root settings.

    Returns:
        LiDAR driver or ``None`` if LiDAR is disabled.
    """
    if cfg.lidar is None or not cfg.lidar.enabled:
        return None

    if cfg.mock_hardware:
        from mousedroid.hardware.lidar.mock_lidar import MockLidar

        _log.info("lidar_driver_mock_built")
        return MockLidar(cfg.lidar)

    from mousedroid.hardware.lidar.ld19_driver import LD19LidarDriver
    from mousedroid.resilience.resilient_lidar import ResilientLidarDriver

    inner = LD19LidarDriver(cfg.lidar)
    _log.info("lidar_driver_built", port=cfg.lidar.serial_port)
    return ResilientLidarDriver(inner, cfg.retry, cfg.circuit_breaker)


def build_lidar_feature_extractor(cfg: Settings) -> object | None:
    """Build LiDAR feature extractor if LiDAR is configured.

    Args:
        cfg: Root settings.

    Returns:
        ``LidarFeatureExtractor`` or ``None`` if LiDAR is disabled.
    """
    if cfg.lidar is None or not cfg.lidar.enabled:
        return None

    from mousedroid.hardware.lidar.feature_extractor import LidarFeatureExtractor

    extractor = LidarFeatureExtractor(cfg.lidar)
    _log.info("lidar_feature_extractor_built", feature_dim=extractor.feature_dim)
    return extractor


def build_tensorrt_compiler(cfg: Settings) -> TensorRTCompilerProtocol:
    """Build TensorRT compiler based on config and hardware availability.

    Returns a real ``JetsonTensorRTCompiler`` when ``cfg.jetson.tensorrt_enabled``
    is True. The real compiler itself falls back to ``torch.jit.trace`` at
    compile time if ``torch2trt`` is missing (operators get a runtime warning
    on the first compile call); the ``torch2trt_available`` field in the
    structured-log event below surfaces that decision at boot time too so
    operator dashboards can ingest it without waiting for the first inference.

    Falls back to ``MockTensorRTCompiler`` when ``tensorrt_enabled`` is False.

    F-009: consolidated the previous two log events
    (``tensorrt_compiler_built`` / ``tensorrt_compiler_mock_built``) into a
    single ``tensorrt_compiler_built`` event with a ``backend`` label so
    operator dashboards can ingest backend selection without a label split.

    Args:
        cfg: Root settings.

    Returns:
        Compiler conforming to ``TensorRTCompilerProtocol``.
    """
    # Import _TORCH2TRT_AVAILABLE once so both branches log the truthful
    # boolean. Previously the mock branch hardcoded ``torch2trt_available=False``
    # which misled dashboards on dev hosts where torch2trt IS installed but
    # tensorrt is just disabled in cfg.
    from mousedroid.efficiency.tensorrt import _TORCH2TRT_AVAILABLE

    if cfg.jetson.tensorrt_enabled:
        from mousedroid.efficiency.tensorrt import JetsonTensorRTCompiler

        _log.info(
            "tensorrt_compiler_built",
            backend="real",
            torch2trt_available=_TORCH2TRT_AVAILABLE,
            precision=cfg.jetson.precision,
            cache_dir=str(cfg.jetson.tensorrt_cache_dir),
            reason="tensorrt_enabled=true",
        )
        return JetsonTensorRTCompiler(cfg.jetson)

    from mousedroid.efficiency.tensorrt import MockTensorRTCompiler

    _log.info(
        "tensorrt_compiler_built",
        backend="mock",
        torch2trt_available=_TORCH2TRT_AVAILABLE,
        reason="tensorrt_enabled=false",
    )
    return MockTensorRTCompiler()


def build_hailo_runtime(cfg: Settings) -> HailoRuntimeProtocol | None:
    """Instantiate Hailo-8 accelerator runtime if configured.

    Creates the runtime instance but does **not** start it — device
    discovery and HEF loading happen in ``await runtime.start()``,
    which is called by the orchestrator during its startup phase.

    Returns ``None`` when Hailo is disabled or the ``hailo_platform``
    package cannot be imported.

    Args:
        cfg: Root settings.

    Returns:
        Hailo runtime instance (not yet started) or ``None``.
    """
    if cfg.hailo is None or not cfg.hailo.enabled:
        return None

    if cfg.mock_hardware:
        from mousedroid.hardware.accelerator.hailo_runtime import MockHailoRuntime

        _log.info("hailo_runtime_mock_built")
        return MockHailoRuntime(cfg.hailo)

    try:
        from mousedroid.hardware.accelerator.hailo_runtime import HailoRuntime

        runtime = HailoRuntime(cfg.hailo)
        _log.info("hailo_runtime_built", device_path=cfg.hailo.device_path)
        return runtime
    except Exception:
        _log.warning("hailo_runtime_build_failed_falling_back_to_gpu", exc_info=True)
        return None


def build_memory_tier(cfg: Settings) -> MemoryTier | None:
    """Build all four memory subsystems if memory is enabled.

    Args:
        cfg: Root settings.

    Returns:
        ``MemoryTier`` dataclass or ``None`` if ``cfg.memory.enabled`` is False.
    """
    if not cfg.memory.enabled:
        return None

    from mousedroid.memory.consolidation import MemoryConsolidation
    from mousedroid.memory.episodic import EpisodicReplay
    from mousedroid.memory.semantic import SemanticIndex
    from mousedroid.memory.tier import MemoryTier
    from mousedroid.memory.working import WorkingMemory

    episodic = EpisodicReplay(cfg.memory, seed=cfg.memory.replay_seed)
    semantic = SemanticIndex(cfg.memory)
    working = WorkingMemory(cfg.memory, embed_dim=cfg.memory.semantic_dim)
    consolidation = MemoryConsolidation(cfg.memory, episodic, semantic)

    _log.info(
        "memory_tier_built",
        episodic_capacity=cfg.memory.episodic_capacity,
        semantic_dim=cfg.memory.semantic_dim,
        working_context=cfg.memory.working_context_size,
    )
    return MemoryTier(
        episodic=episodic,
        semantic=semantic,
        working=working,
        consolidation=consolidation,
    )


def build_experience_logger(cfg: Settings) -> ExperienceLogger | None:
    """Build LMDB experience logger if memory is enabled and experience config is present.

    Args:
        cfg: Root settings.

    Returns:
        ``ExperienceLogger`` or ``None`` if memory is disabled or experience config is absent.
    """
    if not cfg.memory.enabled:
        return None

    experience_cfg = getattr(cfg, "experience", None)
    if experience_cfg is None:
        return None

    from mousedroid.experience.logger import ExperienceLogger

    logger = ExperienceLogger(experience_cfg)
    _log.info("experience_logger_built", path=experience_cfg.path)
    return logger


def build_curiosity_module(cfg: Settings) -> CuriosityProtocol | None:
    """Build intrinsic curiosity module if memory is enabled.

    Args:
        cfg: Root settings.

    Returns:
        ``IntrinsicCuriosityModule`` or ``None`` if memory is disabled.
    """
    if not cfg.memory.enabled:
        return None

    from mousedroid.curiosity.icm import IntrinsicCuriosityModule

    try:
        module = IntrinsicCuriosityModule(cfg.model, cfg.curiosity)
        _log.info("curiosity_module_built", scale=cfg.curiosity.intrinsic_reward_scale)
        return module
    except Exception:
        _log.warning("curiosity_module_build_failed", exc_info=True)
        return None


def build_watchdog(cfg: Settings) -> WatchdogProtocol:
    """Build watchdog notifier based on config.

    Returns :class:`SystemdNotifier` when the ``NOTIFY_SOCKET`` env var is
    present (set automatically by systemd for ``Type=notify`` services),
    :class:`FileHeartbeatNotifier` for Docker/custom monitoring, or
    :class:`NullNotifier` when watchdog is disabled.

    Args:
        cfg: Root settings.

    Returns:
        Watchdog notifier satisfying :class:`WatchdogProtocol`.
    """
    import os
    from pathlib import Path

    from mousedroid.health.watchdog import (
        FileHeartbeatNotifier,
        NullNotifier,
        SystemdNotifier,
    )

    if not cfg.loop.watchdog_enabled:
        return NullNotifier()

    mode = cfg.loop.watchdog_mode
    if mode == "none":
        return NullNotifier()
    if mode == "systemd":
        return SystemdNotifier()
    if mode == "file":
        return FileHeartbeatNotifier(Path(cfg.loop.watchdog_heartbeat_path))
    if mode == "auto":
        if os.environ.get("NOTIFY_SOCKET"):
            return SystemdNotifier()
        return FileHeartbeatNotifier(Path(cfg.loop.watchdog_heartbeat_path))

    _log.warning("unknown_watchdog_mode_falling_back_to_null", mode=mode)
    return NullNotifier()


def build_llm_replanner(cfg: Settings) -> Any:
    """Build the configured arm LLM replanner backend.

    Returns a concrete :class:`LLMReplannerProtocol` implementation. When
    ``cfg.arm_planning.llm_replanner`` is None or the config disables the
    backend, returns :class:`NullLLMReplanner` so the existing
    :class:`Replanner` fall-back to symbolic planning is preserved.

    Args:
        cfg: Root settings.

    Returns:
        A concrete LLM replanner conforming to ``LLMReplannerProtocol``.
    """
    from mousedroid.arm.planning.llm_replanners.null_backend import (
        NullLLMReplanner,
    )

    if cfg.arm_planning is None or cfg.arm_planning.llm_replanner is None:
        return NullLLMReplanner()
    rp_cfg = cfg.arm_planning.llm_replanner
    if not rp_cfg.enabled or rp_cfg.backend == "null":
        return NullLLMReplanner()
    if rp_cfg.backend == "anthropic":
        from mousedroid.arm.planning.llm_replanners.anthropic_backend import (
            AnthropicReplanner,
            AnthropicSDKMissingError,
        )

        try:
            return AnthropicReplanner(rp_cfg)
        except AnthropicSDKMissingError as exc:
            _log.warning(
                "anthropic_replanner_sdk_missing_falling_back",
                error=str(exc),
            )
            return NullLLMReplanner()
    # 'llama' is reserved for a future llm_gateway-backed implementation;
    # defaulting to null until that backend ships keeps factory honest.
    _log.debug("llm_replanner_backend_not_implemented", backend=rp_cfg.backend)
    return NullLLMReplanner()


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
    return target  # type: ignore[no-any-return]


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

    if not approval.require_approval_tool_patterns and not approval.require_approval_skill_patterns:
        return inner
    return PolicyApprovalGate(
        inner,
        tool_patterns=tuple(approval.require_approval_tool_patterns),
        skill_patterns=tuple(approval.require_approval_skill_patterns),
    )


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


def build_orchestrator(cfg: Settings) -> object:
    """Build fully-wired orchestrator.

    Args:
        cfg: Root settings.

    Returns:
        Fully configured ``MouseDroidOrchestrator``.
    """
    from mousedroid.orchestrator.orchestrator import MouseDroidOrchestrator

    # Build Hailo-8 runtime early — shared by camera and arm perception
    hailo_runtime = build_hailo_runtime(cfg)

    wm = build_world_model(cfg)
    agent = build_agent(cfg, wm)
    monitor = build_safety_monitor(cfg)
    esp32 = build_esp32_driver(cfg)

    camera: VisionProtocol | None = None
    try:
        camera = build_camera(cfg, hailo_runtime=hailo_runtime)
    except Exception as exc:  # pylint: disable=broad-except
        _log.warning("camera_init_failed_degrading", error=str(exc))

    distance: DistanceSensorProtocol | None = None
    try:
        distance = build_distance_sensor(cfg)
    except Exception as exc:  # pylint: disable=broad-except
        _log.warning("distance_sensor_init_failed_degrading", error=str(exc))

    microphone = build_microphone(cfg)

    lidar_driver: LidarProtocol | None = None
    try:
        lidar_driver = build_lidar(cfg)
    except Exception as exc:  # pylint: disable=broad-except
        _log.warning("lidar_init_failed_degrading", error=str(exc))

    sensor_manager = build_sensor_manager(
        cfg,
        vision=camera,
        distance=distance,
        esp32=esp32,
        microphone=microphone,
        lidar=lidar_driver,
    )

    cognitive_core: CognitiveCore | None = None
    if cfg.cognitive.enabled:
        try:
            cognitive_core = build_cognitive_core(cfg)
        except Exception as e:  # pylint: disable=broad-except
            if cfg.cognitive.fallback_to_mcts:
                _log.warning(
                    "cognitive_core_init_failed_falling_back_to_mcts",
                    error=str(e),
                )
            else:
                raise

    # Telemetry (optional — disabled by default)
    telemetry_publisher = build_telemetry_publisher(cfg)
    health_monitor = build_health_monitor(cfg)

    # PR #4: per-sensor liveness tracker — declared once at build time
    # so the orchestrator can attach a four-state liveness map
    # (disabled / awaiting / live / stale) to every telemetry frame.
    # Disabled when telemetry itself is off (no consumers).
    liveness_tracker = None
    if cfg.telemetry.enabled:
        from mousedroid.telemetry.sensor_liveness import SensorLivenessTracker

        liveness_tracker = SensorLivenessTracker(
            stale_s=cfg.telemetry.sensor_liveness_stale_s,
        )
        liveness_tracker.register("lidar", enabled=lidar_driver is not None)
        liveness_tracker.register("vision", enabled=camera is not None)
        liveness_tracker.register("audio", enabled=microphone is not None)
        liveness_tracker.register("motor", enabled=True)
        _log.info(
            "sensor_liveness_tracker_built",
            stale_s=cfg.telemetry.sensor_liveness_stale_s,
            lidar_enabled=lidar_driver is not None,
            vision_enabled=camera is not None,
            audio_enabled=microphone is not None,
        )

    # PR #4: synthetic telemetry source for mock_hardware mode (None in
    # production). Lifecycle is owned by the orchestrator.
    mock_telemetry_source = build_mock_telemetry_source(cfg, telemetry_publisher)

    # Optional log ring buffer for telemetry log streaming
    from mousedroid.telemetry.log_buffer import LogRingBuffer as _LogRingBuffer

    log_buffer: _LogRingBuffer | None = None
    buffer_size = cfg.telemetry.log_stream_buffer
    if buffer_size:
        log_buffer = _LogRingBuffer(buffer_size)

    metrics_registry = build_metrics_registry(cfg)
    failure_recorder = build_failure_recorder(cfg, metrics_registry)

    # Shared prompt-injection filter — reused by the LLM gateway and the
    # OpenClaw mission dispatcher so REST + MCP + LLM ingress share one
    # rejection envelope. Built before the telemetry server because the
    # dispatcher (which the server registers POST /api/v1/mission against)
    # consumes it.
    injection_filter = build_injection_filter(cfg)

    # OpenClaw mission dispatcher (None when openclaw is disabled). The
    # deferred orchestrator ref is bound just before returning the
    # orchestrator at the end of this function.
    from mousedroid.orchestrator.mission_dispatcher import build_mission_dispatcher

    mission_dispatcher, deferred_orchestrator_ref = build_mission_dispatcher(
        cfg.openclaw, injection_filter=injection_filter
    )

    telemetry_server = build_telemetry_server(
        cfg,
        telemetry_publisher,
        health_monitor,
        log_buffer=log_buffer,
        metrics_registry=metrics_registry,
        camera=camera,
        mission_dispatcher=mission_dispatcher,
    )

    # LLM gateway + mission parser (optional — gated by llm.enabled)
    llm_gateway: LLMGatewayProtocol | None = None
    if cfg.llm.enabled:
        llm_gateway = build_llm_gateway(
            cfg, injection_filter=injection_filter, metrics=metrics_registry
        )
    mission_parser: MissionParserProtocol | None = build_mission_parser(cfg)

    # VLA policy (optional — gated by vla.backend, default 'none')
    vla_policy: VLAPolicyProtocol | None = build_vla_policy(cfg)

    from mousedroid.common.tools.registry import create_default_registry

    _tool_registry = create_default_registry(
        llm_gateway=llm_gateway,
        metrics_registry=metrics_registry,
        gcp_cfg=cfg.gcp,
    )

    # Motor tools — only registered for the rover platform; the arm
    # platform has its own actuation surface and shouldn't expose
    # rover-specific velocity controls. Import is local so static
    # analysers don't drag the runtime symbol into the TYPE_CHECKING
    # block above.
    from mousedroid.config.schema import PlatformType as _Platform

    if cfg.platform == _Platform.MOUSE_DROID:
        from mousedroid.common.tools.motor_tools import MotorToolDeps, register_motor_tools

        register_motor_tools(
            _tool_registry,
            MotorToolDeps(esp32=esp32, cfg=cfg),
        )

    # MCP server (optional — disabled by default). Built after the tool
    # registry so it can bridge real tools, and after telemetry/log
    # buffers so it can expose them as resources. Memory tier is wired
    # below; when present and enabled, the server gets a non-None
    # ``memory_tier``.
    speaker = build_speaker(cfg)
    voice_engine = build_voice_engine(cfg, speaker=speaker)

    # Face display (optional — disabled by default)
    face_display = build_face_display(cfg)
    face_controller = build_face_controller(cfg, face_display)

    # Memory tier + experience logger + curiosity module (optional)
    memory_tier = build_memory_tier(cfg)
    experience_logger = build_experience_logger(cfg)
    curiosity_module = build_curiosity_module(cfg)

    # OpenClaw MEMORY.md exporter (None unless cfg.openclaw.shared_memory_path set)
    memory_exporter = build_memory_exporter(cfg)

    mcp_server = build_mcp_server(
        cfg,
        tool_registry=_tool_registry,
        safety_monitor=monitor,
        publisher=telemetry_publisher,
        log_buffer=log_buffer,
        metrics_registry=metrics_registry,
        memory_tier=memory_tier,
    )

    # Watchdog notifier (optional — disabled by default)
    watchdog = build_watchdog(cfg)

    # GCP Digital Twin (optional — disabled when gcp=None)
    cloud_sink = build_cloud_telemetry_sink(cfg, metrics_registry=metrics_registry)
    cloud_experience_exporter = build_cloud_experience_exporter(cfg)

    # Agent harness (all opt-in; passing None preserves byte-identical legacy behaviour)
    task_tracker = build_task_tracker(cfg)
    journal = build_journal(cfg)
    approval_gate = build_approval_gate(cfg)
    skill_loaders = build_skill_loaders(cfg)
    skill_registry = build_skill_registry(cfg, skill_loaders)
    skill_delegator = build_skill_delegator(
        cfg,
        skill_registry,
        approval_gate,
        journal,
        task_tracker,
        llm_gateway=llm_gateway,
    )
    # Wire the approval gate onto the tool registry so dispatched tools
    # flagged with ``requires_approval=True`` are gated through it.
    if hasattr(_tool_registry, "set_approval_gate"):
        _tool_registry.set_approval_gate(approval_gate)
    hook_registry = build_hook_registry(cfg, journal)

    # Tier C1 / C1.2 — wire the optional OTA weight-update pollers. Default
    # ``cfg.cloud.weight_update.poll_interval_s = 0.0`` keeps the mapping
    # empty so the orchestrator's swap helper short-circuits and existing
    # deployments remain byte-identical. When polling is enabled the
    # ``policy`` poller is always present; the ``world_model`` poller is
    # added iff ``cfg.cloud.weight_update.world_model_enabled is True``.
    weight_update_pollers = build_weight_update_pollers(cfg, metrics=metrics_registry)
    weight_update_loader = build_weight_update_loader(cfg)
    # Tier C2 / C2.1 — soft-constraint safety projector. Returns ``None``
    # when ``cfg.safety.projector.enabled`` is ``False`` (the default),
    # which makes the orchestrator's projection seam a no-op so pre-C2
    # deployments produce byte-identical actions.
    safety_projector = build_safety_projector(cfg, metrics=metrics_registry)
    # Tier C2.3 — build VLM head + LLM replanner so the lifecycle is no
    # longer the defensive None from PR #98. Both build_* helpers short-
    # circuit on their own flags so this remains byte-identical to
    # pre-Tier-C2.3 behaviour when the new ``cfg.mission.vlm_progress_enabled``
    # and ``cfg.mission.llm_replanner_enabled`` flags stay at False.
    vlm_progress = build_vlm_progress(cfg)
    mission_replanner = build_mission_replanner(
        cfg,
        llm_gateway=llm_gateway,
        metrics=metrics_registry,
    )
    # Tier C2 / C2.2 — mission lifecycle state machine. Returns ``None``
    # in four scenarios so the orchestrator's POST_TICK seam stays a no-op
    # and pre-Tier-C2.3 deployments are byte-identical:
    #   1. ``cfg.mission.replan_enabled`` is ``False`` (the default).
    #   2. ``cfg.mission.vlm_progress_enabled`` is ``False`` (vlm_progress=None).
    #   3. ``cfg.mission.llm_replanner_enabled`` is ``False`` OR
    #      ``cfg.llm.enabled`` is ``False`` (mission_replanner=None).
    #   4. Either of the above missing — defensive guard inside
    #      :func:`build_mission_lifecycle` (PR #98 Copilot HIGH fix).
    mission_lifecycle = build_mission_lifecycle(
        cfg,
        task_tracker=task_tracker,
        vlm_progress=vlm_progress,
        replanner=mission_replanner,
        metrics=metrics_registry,
    )

    orchestrator = MouseDroidOrchestrator(
        world_model=wm,
        agents=[agent],
        safety_monitor=monitor,
        esp32=esp32,
        sensor_manager=sensor_manager,
        cognitive_core=cognitive_core,
        cfg=cfg,
        telemetry_publisher=telemetry_publisher,
        telemetry_server=telemetry_server,
        voice_engine=voice_engine,
        hailo_runtime=hailo_runtime,
        memory_tier=memory_tier,
        experience_logger=experience_logger,
        curiosity_module=curiosity_module,
        llm_gateway=llm_gateway,
        mission_parser=mission_parser,
        watchdog=watchdog,
        cloud_sink=cloud_sink,
        cloud_experience_exporter=cloud_experience_exporter,
        tool_registry=_tool_registry,
        face_controller=face_controller,
        mcp_server=mcp_server,
        vla_policy=vla_policy,
        hook_registry=hook_registry,
        task_tracker=task_tracker,
        journal=journal,
        skill_delegator=skill_delegator,
        memory_exporter=memory_exporter,
        mission_dispatcher=mission_dispatcher,
        failure_recorder=failure_recorder,
        liveness_tracker=liveness_tracker,
        mock_telemetry_source=mock_telemetry_source,
        metrics=metrics_registry,
        weight_update_pollers=weight_update_pollers,
        weight_update_loader=weight_update_loader,
        safety_projector=safety_projector,
        mission_lifecycle=mission_lifecycle,
    )
    # Bind the deferred orchestrator reference so the OpenClaw mission
    # dispatcher (built before the orchestrator above) can route through
    # ``orchestrator.process_mission`` from this point onwards.
    if deferred_orchestrator_ref is not None:
        deferred_orchestrator_ref.bind(orchestrator)
    return orchestrator


# ---------------------------------------------------------------------------
# Robot Arm platform factory functions
# ---------------------------------------------------------------------------


def build_arm_driver(cfg: Settings) -> ArmDriverProtocol:
    """Build robot arm hardware driver based on config.

    Args:
        cfg: Root settings (must have arm config populated).

    Returns:
        Arm driver conforming to ``ArmDriverProtocol``.

    Raises:
        ValueError: If arm config is not populated.
    """
    if cfg.arm is None:
        msg = "arm config required for robot arm platform"
        raise ValueError(msg)

    if cfg.mock_hardware:
        from mousedroid.arm.hardware.mock_arm_driver import MockArmDriver

        _log.info("arm_driver_mock_built")
        return MockArmDriver(cfg.arm)

    from mousedroid.arm.hardware.so_arm100_driver import SoArm100Driver

    _log.info("arm_driver_real_built", port=cfg.arm.serial_port)
    return SoArm100Driver(cfg.arm)


def build_arm_planner(cfg: Settings) -> ArmPlannerProtocol:
    """Build symbolic planner for arm manipulation tasks.

    Args:
        cfg: Root settings (must have arm_planning and arm_task populated).

    Returns:
        Planner conforming to ``ArmPlannerProtocol``.

    Raises:
        ValueError: If arm planning or task config is not populated.
    """
    if cfg.arm_planning is None or cfg.arm_task is None:
        msg = "arm_planning and arm_task configs required for arm planner"
        raise ValueError(msg)

    from mousedroid.arm.planning.symbolic_planner import SymbolicPlanner

    _log.info("arm_planner_built", backend=cfg.arm_planning.planner_backend)
    return SymbolicPlanner(cfg.arm_planning, cfg.arm_task)


def build_arm_environment(cfg: Settings) -> ArmEnvironmentProtocol:
    """Build Gymnasium environment for arm training.

    Args:
        cfg: Root settings (must have arm_task and arm_training populated).

    Returns:
        Environment conforming to ``ArmEnvironmentProtocol``.

    Raises:
        ValueError: If arm task or training config is not populated.
    """
    if cfg.arm_task is None or cfg.arm_training is None:
        msg = "arm_task and arm_training configs required for arm environment"
        raise ValueError(msg)

    dof = cfg.arm.dof if cfg.arm is not None else 6

    if cfg.arm_task.task_type == "laundry_sorting":
        from mousedroid.arm.environments.laundry_sorting import LaundrySortingEnv

        _log.info("arm_env_laundry_built")
        return LaundrySortingEnv(cfg.arm_task, cfg.arm_training, dof=dof)

    from mousedroid.arm.environments.tower_of_hanoi import TowerOfHanoiEnv

    _log.info("arm_env_hanoi_built", num_disks=cfg.arm_task.num_disks)
    return TowerOfHanoiEnv(cfg.arm_task, cfg.arm_training, dof=dof)


def build_arm_controller(cfg: Settings) -> ArmControllerProtocol:
    """Build RL controller for arm manipulation.

    Composes a ``SACAgent`` with ``ActionPrimitives`` inside an
    ``ArmController`` that implements ``ArmControllerProtocol``.

    Args:
        cfg: Root settings (must have arm and arm_training populated).

    Returns:
        Controller conforming to ``ArmControllerProtocol``.

    Raises:
        ValueError: If arm or arm_training config is not populated.
    """
    if cfg.arm_training is None or cfg.arm is None:
        msg = "arm and arm_training configs required for arm controller"
        raise ValueError(msg)

    from mousedroid.arm.control.controller import ArmController
    from mousedroid.arm.control.primitives import ActionPrimitives
    from mousedroid.arm.control.sac_agent import SACAgent

    driver = build_arm_driver(cfg)
    agent = SACAgent(cfg.arm_training)
    primitives = ActionPrimitives(cfg.arm, driver)
    _log.info("arm_controller_built", algorithm=cfg.arm_training.algorithm)
    return ArmController(agent, primitives)


def build_arm_perception(
    cfg: Settings,
    hailo_runtime: HailoRuntimeProtocol | None = None,
) -> ArmPerceptionProtocol:
    """Build perception pipeline for arm manipulation.

    When a Hailo-8 runtime is provided and ``yolo_backend`` is
    ``"hailo"`` or ``"auto"``, YOLO detection is offloaded to the
    accelerator via :class:`HailoYOLODetector`.

    Args:
        cfg: Root settings (must have arm_perception and arm_task populated).
        hailo_runtime: Optional Hailo-8 runtime for accelerated YOLO detection.

    Returns:
        Perception facade conforming to ``ArmPerceptionProtocol``.

    Raises:
        ValueError: If arm_perception or arm_task config is not populated.
    """
    if cfg.arm_perception is None or cfg.arm_task is None:
        msg = "arm_perception and arm_task configs required for perception"
        raise ValueError(msg)

    import numpy as np

    from mousedroid.arm.perception.facade import ArmPerception

    # Default identity intrinsics — overridden by camera driver at runtime
    intrinsics = np.eye(3, dtype=np.float64)
    intrinsics[0, 0] = cfg.arm_perception.default_focal_length  # fx
    intrinsics[1, 1] = cfg.arm_perception.default_focal_length  # fy
    intrinsics[0, 2] = cfg.arm_perception.default_principal_x  # cx
    intrinsics[1, 2] = cfg.arm_perception.default_principal_y  # cy

    # Build Hailo YOLO detector if available
    detector = None
    if hailo_runtime is not None and cfg.arm_perception.yolo_backend in ("hailo", "auto"):
        from mousedroid.arm.perception.hailo_detector import HailoYOLODetector

        detector = HailoYOLODetector(cfg.arm_perception, hailo_runtime)
        _log.info("arm_perception_hailo_yolo_built")

    _log.info("arm_perception_built", camera=cfg.arm_perception.depth_camera_type)
    return ArmPerception(
        cfg.arm_perception,
        cfg.arm_task,
        intrinsics,
        object_detector=detector,
    )


# ---------------------------------------------------------------------------
# 4WD rover sim-to-real factory functions (Phase A scaffold)
# ---------------------------------------------------------------------------


def build_rover_env(cfg: Settings) -> RoverEnvProtocol:
    """Build the rover simulation environment selected by ``cfg.rover.sim.backend``.

    Backends:
        - ``"mock"`` (default): NumPy-only kinematic integrator. Has no
          physics or GPU dependency and is the only backend used in CI.
        - ``"isaac_lab"``: Isaac Lab env stub; requires
          ``pip install -e ".[isaac]"`` on a workstation with NVIDIA
          Isaac Lab prerequisites. Phase B fills in the actual
          articulation / sensor wiring.
        - ``"mujoco"``: reserved for a future PR — raises
          :class:`NotImplementedError`.

    Args:
        cfg: Root settings. ``cfg.rover`` must be populated.

    Returns:
        Environment conforming to :class:`RoverEnvProtocol`.

    Raises:
        ValueError: If ``cfg.rover`` is ``None``.
        NotImplementedError: For backends not yet wired in this phase.
    """
    if cfg.rover is None:
        msg = (
            "rover config required for build_rover_env; set the top-level "
            "'rover:' block in your YAML or pass RoverConfig() directly."
        )
        raise ValueError(msg)

    backend = cfg.rover.sim.backend
    if backend == "mock":
        from mousedroid.sim.mock_rover_env import MockRoverEnv

        _log.info(
            "rover_env_mock_built",
            mode=cfg.rover.action.mode,
            obs_keys=list(cfg.rover.observation.enabled_keys()),
        )
        return MockRoverEnv(
            cfg.rover,
            wheel_radius_m=cfg.robot.wheel_radius_m,
            track_width_m=cfg.robot.track_width_m,
        )

    if backend == "isaac_lab":
        from mousedroid.sim.isaaclab.rover_env import RoverIsaacLabEnv

        env = RoverIsaacLabEnv(
            cfg.rover,
            wheel_radius_m=cfg.robot.wheel_radius_m,
            track_width_m=cfg.robot.track_width_m,
            domain_randomization=cfg.domain_randomization,
        )
        _log.info(
            "rover_env_isaaclab_built",
            num_envs=cfg.rover.sim.num_envs,
            dr_enabled=cfg.domain_randomization.enabled,
        )
        return env

    if backend == "mujoco":
        msg = "MuJoCo rover backend is reserved; see Phase B of the sim-to-real plan."
        raise NotImplementedError(msg)

    msg = f"unknown rover sim backend: {backend!r}"
    raise ValueError(msg)


def build_cloud_telemetry_sink(
    cfg: Settings,
    *,
    metrics_registry: MetricsRegistry | None = None,
) -> CloudTelemetrySinkProtocol | None:
    """Build GCP Pub/Sub telemetry sink if GCP is configured.

    Returns ``None`` when ``cfg.gcp`` is ``None`` (offline mode) or when
    the ``google-cloud-pubsub`` package is not installed.

    Args:
        cfg: Root settings.
        metrics_registry: Optional metrics registry. When provided the
            sink forwards publish outcomes, publish latency, and circuit
            breaker state transitions to the registry.

    Returns:
        Cloud telemetry sink or None.
    """
    if cfg.gcp is None:
        return None
    try:
        from mousedroid.cloud.pubsub_sink import CloudTelemetrySink
    except ImportError:
        _log.warning(
            "cloud_pubsub_not_available",
            hint="Install via: pip install 'mousedroid[gcp]'",
        )
        return None

    sink = CloudTelemetrySink(cfg.gcp, metrics=metrics_registry)
    _log.info("cloud_telemetry_sink_built", metrics_wired=metrics_registry is not None)
    return sink


def build_cloud_experience_exporter(
    cfg: Settings,
) -> CloudExperienceExporterProtocol | None:
    """Build GCS experience exporter if GCP is configured.

    Returns ``None`` when ``cfg.gcp`` is ``None`` or when the
    ``google-cloud-storage`` package is not installed.

    Args:
        cfg: Root settings.

    Returns:
        Cloud experience exporter or None.
    """
    if cfg.gcp is None:
        return None
    try:
        from mousedroid.cloud.experience_exporter import CloudExperienceExporter
    except ImportError:
        _log.warning(
            "cloud_storage_not_available",
            hint="Install via: pip install 'mousedroid[gcp]'",
        )
        return None

    exporter = CloudExperienceExporter(cfg.gcp, cfg.experience)
    _log.info("cloud_experience_exporter_built")
    return exporter
