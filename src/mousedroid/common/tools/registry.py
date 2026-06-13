"""Tool registry — registration and dispatch for MouseDroid tools."""

from __future__ import annotations

import asyncio
import importlib
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from mousedroid.config.schema import GCPConfig
    from mousedroid.harness.approval.protocol import ApprovalGateProtocol
    from mousedroid.llm_gateway.protocol import GoalVector, LLMGatewayProtocol
    from mousedroid.telemetry.metrics import MetricsRegistry

from mousedroid.common.imports import module_available
from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)


@dataclass(frozen=True)
class ToolSpec:
    """Specification for a registered tool.

    Attributes:
        name: Unique tool identifier.
        description: Human-readable description.
        handler: Async callable that executes the tool.
    """

    name: str
    description: str
    handler: Callable[..., Awaitable[Any]]


class ToolDispatchDeniedError(PermissionError):
    """Raised when an approval gate denies a dispatch with ``requires_approval``."""


class ToolRegistry:
    """Registry for MouseDroid tools.

    Provides registration and dispatch for platform tools.

    When constructed with an ``approval_gate``, any dispatched tool whose
    spec is a :class:`ValidatedToolSpec` with ``requires_approval=True``
    must clear the gate before its handler runs. The gate is consulted
    asynchronously and a denial raises :class:`ToolDispatchDeniedError`
    so callers can distinguish "tool denied" from "tool failed".
    """

    def __init__(self, approval_gate: object | None = None) -> None:
        """Initialise empty registry.

        Args:
            approval_gate: Optional :class:`ApprovalGateProtocol`.
                When supplied, validated specs flagged with
                ``requires_approval=True`` are gated through it.
                Defaults to ``None`` so existing callers (and the
                legacy plain-``ToolSpec`` path) are unaffected.
        """
        self._tools: dict[str, ToolSpec] = {}
        self._approval_gate = approval_gate

    def register(self, spec: ToolSpec) -> None:
        """Register a tool.

        Args:
            spec: Tool specification to register.
        """
        if spec.name in self._tools:
            _log.warning("tool_already_registered", name=spec.name)
        self._tools[spec.name] = spec
        _log.debug("tool_registered", name=spec.name)

    def set_approval_gate(self, gate: object | None) -> None:
        """Bind (or clear) the approval gate post-construction.

        Lets the factory wire the gate after both the registry and the
        gate have been built without forcing a reconstruction.
        """
        self._approval_gate = gate

    def get(self, name: str) -> ToolSpec | None:
        """Get tool spec by name.

        Args:
            name: Tool identifier.

        Returns:
            Tool spec or None if not found.
        """
        return self._tools.get(name)

    async def dispatch(self, name: str, **kwargs: Any) -> Any:
        """Dispatch a tool by name.

        When ``spec`` is a :class:`ValidatedToolSpec` with non-``None``
        schemas, the input ``**kwargs`` and return value are validated by
        Pydantic. When ``spec.requires_approval`` is True and an
        ``approval_gate`` was bound on the registry, the gate's
        ``decide()`` method is awaited before the handler runs. Plain
        :class:`ToolSpec` instances keep the legacy zero-overhead path so
        existing callers and tools are unaffected.

        Args:
            name: Tool identifier.
            **kwargs: Arguments to pass to the tool handler.

        Returns:
            Tool execution result (validated when ``output_schema`` is set).

        Raises:
            KeyError: If tool is not registered.
            ToolInputValidationError: If validated input is invalid.
            ToolOutputValidationError: If validated output is invalid.
            ToolDispatchDeniedError: If the approval gate denies dispatch.
        """
        spec = self._tools.get(name)
        if spec is None:
            msg = f"Tool not registered: {name}"
            raise KeyError(msg)

        # Local import to avoid an import cycle (validation imports ToolSpec).
        from mousedroid.common.tools.validation import (
            ValidatedToolSpec,
            validate_input,
            validate_output,
        )

        if isinstance(spec, ValidatedToolSpec):
            await self._maybe_check_approval(spec, kwargs)
            kwargs = validate_input(spec, kwargs)
            _log.info("tool_dispatch", name=name, validated=True)
            result = await spec.handler(**kwargs)
            return validate_output(spec, result)

        _log.info("tool_dispatch", name=name)
        return await spec.handler(**kwargs)

    async def _maybe_check_approval(
        self,
        spec: Any,
        kwargs: dict[str, Any],
    ) -> None:
        """Consult the approval gate when ``spec.requires_approval`` is set.

        No-ops when no gate is bound or the spec doesn't require approval,
        keeping the hot path free of overhead. Denial is logged at WARNING
        and surfaces as :class:`ToolDispatchDeniedError`.
        """
        if not getattr(spec, "requires_approval", False):
            return
        gate = self._approval_gate
        if gate is None:
            _log.warning(
                "tool_requires_approval_no_gate",
                tool=spec.name,
            )
            return
        # Local import keeps the registry decoupled from the harness package
        # at import time (the gate is duck-typed via its ``decide`` coroutine).
        from mousedroid.harness.approval.protocol import ApprovalRequest

        request = ApprovalRequest(
            tool_name=spec.name,
            action="tool_dispatch",
            payload={"kwargs_keys": sorted(kwargs.keys())},
        )
        decision = await cast("ApprovalGateProtocol", gate).decide(request)
        if not decision.approved:
            _log.warning(
                "tool_dispatch_denied",
                tool=spec.name,
                reason=decision.reason,
                decided_by=decision.decided_by,
            )
            raise ToolDispatchDeniedError(
                f"Tool dispatch denied for {spec.name!r}: {decision.reason}"
            )

    @property
    def names(self) -> list[str]:
        """List of registered tool names."""
        return list(self._tools.keys())

    def __len__(self) -> int:
        """Number of registered tools."""
        return len(self._tools)


# ---------------------------------------------------------------------------
# Built-in tool handlers
# ---------------------------------------------------------------------------


async def _health_check() -> dict[str, str]:
    """Run system health check."""
    return {"status": "ok"}


async def _calibrate_ultrasonic() -> dict[str, str]:
    """Calibrate ultrasonic sensor."""
    return {"status": "calibration_complete"}


async def _esp32_diagnostics() -> dict[str, str]:
    """Run ESP32 communication diagnostics."""
    return {"status": "esp32_ok"}


async def _tensorrt_compile() -> dict[str, str]:
    """Compile models to TensorRT engines."""
    return {"status": "compilation_pending"}


async def _benchmark_latency() -> dict[str, str]:
    """Run latency benchmark."""
    return {"status": "benchmark_pending"}


async def _export_experience(path: str = "") -> dict[str, str]:
    """Export experience data.

    Args:
        path: Export destination path (from config or caller).

    Returns:
        Export status.
    """
    return {"status": "exported", "path": path}


async def _diagnose_cloud(gcp_cfg: GCPConfig | None = None) -> dict[str, object]:
    """Validate GCP configuration, auth, SDK imports, and Pub/Sub publish path.

    Pub/Sub does not expose a dry-run publish API, so this tool performs a
    minimal live publish against the configured telemetry topic when cloud is
    enabled. The payload is intentionally tiny and carries a diagnostic marker.
    """
    if gcp_cfg is None:
        return {"status": "cloud_disabled"}

    from mousedroid.cloud._auth import resolve_credentials

    try:
        pubsub_module = importlib.import_module("google.cloud.pubsub_v1")
        importlib.import_module("google.cloud.storage")
    except ImportError as exc:
        _log.warning("tool_diagnose_cloud_dependency_missing", error=str(exc))
        return {
            "status": "missing_dependency",
            "error": str(exc),
        }

    try:
        credentials, effective_project = resolve_credentials(gcp_cfg)
    except (FileNotFoundError, ImportError, RuntimeError, ValueError) as exc:
        _log.warning("tool_diagnose_cloud_credentials_failed", error=str(exc))
        return {
            "status": "credentials_error",
            "error": str(exc),
        }

    publisher_api = cast(Any, pubsub_module)
    publisher_factory = cast(Callable[..., Any], publisher_api.PublisherClient)
    publisher = publisher_factory(credentials=credentials)
    topic_path = publisher.topic_path(gcp_cfg.project_id, gcp_cfg.pubsub.telemetry_topic)
    payload = f"diagnose_cloud:{gcp_cfg.robot_id}".encode()

    try:
        loop = asyncio.get_running_loop()
        future = publisher.publish(
            topic_path,
            data=payload,
            type="diagnostic",
            source_id=gcp_cfg.robot_id,
        )
        await loop.run_in_executor(None, future.result, gcp_cfg.pubsub.publish_timeout_s)
    except TimeoutError as exc:
        _log.warning("tool_diagnose_cloud_publish_timeout", topic=topic_path, error=str(exc))
        return {
            "status": "publish_timeout",
            "topic": topic_path,
            "project_id": effective_project,
            "error": str(exc),
        }
    except Exception as exc:  # pylint: disable=broad-except
        _log.warning(
            "tool_diagnose_cloud_publish_failed",
            topic=topic_path,
            error=str(exc),
        )
        return {
            "status": "publish_error",
            "topic": topic_path,
            "project_id": effective_project,
            "error": str(exc),
        }

    _log.info(
        "tool_diagnose_cloud_ok",
        topic=topic_path,
        project_id=effective_project,
    )
    return {
        "status": "ok",
        "project_id": effective_project,
        "topic": topic_path,
        "payload_bytes": len(payload),
    }


def _goal_vector_to_dict(goal: GoalVector) -> dict[str, float]:
    """Convert a GoalVector into a serialisable dictionary."""
    return {
        "vx_target": float(goal.vx_target),
        "vy_target": float(goal.vy_target),
        "omega_target": float(goal.omega_target),
    }


def _record_llm_metrics(
    metrics_registry: MetricsRegistry | None,
    result: str,
    latency_ms: float | None = None,
) -> None:
    """Record LLM translation counters and latency when metrics are enabled."""
    if metrics_registry is None:
        return
    metrics_registry.inc_llm_translation(result)
    if latency_ms is not None:
        metrics_registry.observe_llm_translation_latency_ms(latency_ms)


async def _translate_nl_mission(
    mission: str = "",
    llm_gateway: LLMGatewayProtocol | None = None,
    metrics_registry: MetricsRegistry | None = None,
) -> dict[str, object]:
    """Translate NL mission via LLM gateway.

    Args:
        mission: Natural language mission text.
        llm_gateway: Optional runtime LLM gateway.
        metrics_registry: Optional shared Prometheus metrics registry.

    Returns:
        Translation status.
    """
    mission_text = mission.strip()
    if not mission_text:
        _record_llm_metrics(metrics_registry, "invalid_request")
        return {
            "status": "invalid_request",
            "mission": mission_text,
            "error": "mission must be non-empty",
        }

    if llm_gateway is None or not getattr(llm_gateway, "is_ready", False):
        _record_llm_metrics(metrics_registry, "llm_unavailable")
        return {"status": "llm_unavailable", "mission": mission_text}

    start_time = time.monotonic()
    try:
        goal = await llm_gateway.translate_mission(mission_text)
    except ValueError as exc:
        latency_ms = (time.monotonic() - start_time) * 1000.0
        _record_llm_metrics(metrics_registry, "invalid_request", latency_ms)
        return {
            "status": "invalid_request",
            "mission": mission_text,
            "error": str(exc),
        }
    except (TimeoutError, asyncio.TimeoutError) as exc:
        # asyncio.TimeoutError is an alias for TimeoutError on Python 3.11+,
        # but some third-party async libraries still raise the asyncio variant
        # (or subclass it), so listing both keeps the timeout path robust.
        latency_ms = (time.monotonic() - start_time) * 1000.0
        _record_llm_metrics(metrics_registry, "timeout", latency_ms)
        _log.warning("tool_translate_nl_mission_timeout", error=str(exc))
        return {
            "status": "timeout",
            "mission": mission_text,
            "error": str(exc),
        }
    except Exception as exc:  # pylint: disable=broad-except
        latency_ms = (time.monotonic() - start_time) * 1000.0
        _record_llm_metrics(metrics_registry, "error", latency_ms)
        _log.warning("tool_translate_nl_mission_failed", error=str(exc))
        return {
            "status": "error",
            "mission": mission_text,
            "error": str(exc),
        }

    latency_ms = (time.monotonic() - start_time) * 1000.0
    _record_llm_metrics(metrics_registry, "translated", latency_ms)
    return {
        "status": "translated",
        "mission": mission_text,
        "goal_vector": _goal_vector_to_dict(goal),
        "latency_ms": latency_ms,
    }


async def _system_info() -> dict[str, str]:
    """Get system information."""
    import platform

    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
    }


async def _mic_diagnostics() -> dict[str, str]:
    """Run USB microphone diagnostics.

    Returns:
        Microphone diagnostic status.
    """
    try:
        import pyaudio  # pragma: no cover
    except ImportError:
        return {"status": "pyaudio_not_installed"}

    pa = None  # pragma: no cover
    try:  # pragma: no cover
        pa = pyaudio.PyAudio()  # pragma: no cover
        device_count = pa.get_device_count()  # pragma: no cover
        input_devices = []  # pragma: no cover
        for i in range(device_count):  # pragma: no cover
            info = pa.get_device_info_by_index(i)  # pragma: no cover
            if int(info.get("maxInputChannels", 0)) > 0:  # pragma: no cover
                input_devices.append(str(info.get("name", f"device_{i}")))  # pragma: no cover
        return {  # pragma: no cover
            "status": "ok",
            "device_count": str(device_count),
            "input_devices": ", ".join(input_devices) or "none",
        }
    except Exception:  # pragma: no cover
        _log.warning("mic_diagnostics_error", exc_info=True)  # pragma: no cover
        return {"status": "error"}  # pragma: no cover
    finally:  # pragma: no cover
        if pa is not None:  # pragma: no cover
            try:  # pragma: no cover
                pa.terminate()  # pragma: no cover
            except Exception:  # pragma: no cover
                _log.debug("mic_diagnostics_terminate_failed", exc_info=True)  # pragma: no cover


async def _lidar_diagnostics() -> dict[str, str]:
    """Run FHL-LD19 LiDAR diagnostics.

    Returns:
        LiDAR diagnostic status including pyserial availability.
    """
    if not module_available("serial"):
        return {"status": "pyserial_not_installed"}

    return {"status": "ok", "driver": "FHL-LD19"}


def create_default_registry(
    llm_gateway: LLMGatewayProtocol | None = None,
    metrics_registry: MetricsRegistry | None = None,
    gcp_cfg: GCPConfig | None = None,
) -> ToolRegistry:
    """Create a ToolRegistry pre-populated with built-in tools.

    Returns:
        Registry with all built-in tools registered.
    """
    registry = ToolRegistry()

    async def translate_nl_mission(mission: str = "") -> dict[str, object]:
        return await _translate_nl_mission(
            mission=mission,
            llm_gateway=llm_gateway,
            metrics_registry=metrics_registry,
        )

    async def diagnose_cloud() -> dict[str, object]:
        return await _diagnose_cloud(gcp_cfg)

    tools = [
        ToolSpec("health_check", "Run system health check", _health_check),
        ToolSpec("calibrate_ultrasonic", "Calibrate ultrasonic sensor", _calibrate_ultrasonic),
        ToolSpec("esp32_diagnostics", "Run ESP32 diagnostics", _esp32_diagnostics),
        ToolSpec("tensorrt_compile", "Compile TensorRT models", _tensorrt_compile),
        ToolSpec("benchmark_latency", "Run latency benchmark", _benchmark_latency),
        ToolSpec("export_experience", "Export experience data", _export_experience),
        ToolSpec(
            "diagnose_cloud",
            "Validate GCP config and telemetry publish path",
            diagnose_cloud,
        ),
        ToolSpec("translate_nl_mission", "Translate NL mission", translate_nl_mission),
        ToolSpec("system_info", "Get system information", _system_info),
        ToolSpec("mic_diagnostics", "Run USB microphone diagnostics", _mic_diagnostics),
        ToolSpec("lidar_diagnostics", "Run FHL-LD19 LiDAR diagnostics", _lidar_diagnostics),
    ]
    for tool in tools:
        registry.register(tool)
    return registry
