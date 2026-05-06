"""Hailo-8 neural accelerator runtime wrapper.

Manages the Hailo-8 device lifecycle, loads compiled HEF models,
and dispatches inference requests. Provides a shared runtime
instance for both YOLO detection and feature extraction pipelines.

The runtime exposes both async (``run_inference``) and sync
(``infer_sync``) interfaces. Camera drivers and detectors that
implement synchronous protocols use ``infer_sync`` directly;
the orchestrator calls the async ``start`` / ``stop`` lifecycle.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, cast, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from mousedroid.constants import HAILO_MOCK_FEATURE_EXTRACTOR_DIM, HAILO_MOCK_YOLO_OUTPUT_SHAPE
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import HailoConfig

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency handle
# ---------------------------------------------------------------------------

_hailort: Any
try:
    import hailo_platform as _hailort  # type: ignore[no-redef]
except ImportError:  # pragma: no cover
    _hailort = None


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class HailoRuntimeProtocol(Protocol):
    """Interface for Hailo-8 accelerator runtime."""

    def infer_sync(
        self,
        model_name: str,
        input_data: NDArray[np.uint8],
    ) -> NDArray[np.float32]:
        """Run synchronous inference on a loaded HEF model.

        Thread-safe.  Used by synchronous callers (detectors, extractors).

        Args:
            model_name: Registered model identifier.
            input_data: Input tensor as uint8 numpy array.

        Returns:
            Output tensor as float32 numpy array.
        """
        ...  # pragma: no cover

    async def start(self) -> None:
        """Discover device and load HEF models."""
        ...  # pragma: no cover

    async def stop(self) -> None:
        """Release device and clean up resources."""
        ...  # pragma: no cover

    def is_available(self) -> bool:
        """Check whether the Hailo device is online and models are loaded."""
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Concrete implementation
# ---------------------------------------------------------------------------


class HailoRuntime:
    """Hailo-8 runtime managing device lifecycle and HEF inference.

    Wraps the ``hailo_platform`` API with async lifecycle (``start`` /
    ``stop``) and a thread-safe synchronous inference method
    (``infer_sync``).  A ``threading.Lock`` serializes concurrent
    hardware access to avoid PCIe bus contention with the NVMe SSD
    (both share the Orin Nano PCIe root complex).

    VStream pipelines are created once during model loading and reused
    across inference calls to avoid per-call buffer allocation overhead.

    Args:
        cfg: Hailo-8 configuration.
    """

    def __init__(self, cfg: HailoConfig) -> None:
        """Initialise Hailo runtime.

        Args:
            cfg: Hailo-8 configuration with device path and HEF model paths.
        """
        self._cfg = cfg
        self._device: Any = None
        self._models: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._available = False
        _log.info(
            "hailo_runtime_init",
            device_path=cfg.device_path,
            yolo_hef=str(cfg.yolo_hef_path),
            feature_hef=str(cfg.feature_extractor_hef_path),
        )

    async def start(self) -> None:
        """Discover the Hailo device and load HEF models.

        All blocking ``hailo_platform`` calls are dispatched via
        ``asyncio.to_thread`` to avoid blocking the event loop.
        """
        if _hailort is None:
            _log.warning("hailo_platform_not_installed")
            return

        try:
            self._device = await asyncio.to_thread(self._create_device)
            _log.info("hailo_device_discovered", device_path=self._cfg.device_path)
        except Exception:
            _log.warning(
                "hailo_device_not_found",
                device_path=self._cfg.device_path,
                exc_info=True,
            )
            return

        await self._load_hef("yolo", self._cfg.yolo_hef_path)
        await self._load_hef("feature_extractor", self._cfg.feature_extractor_hef_path)

        if self._models:
            self._available = True
            _log.info("hailo_runtime_started", loaded_models=list(self._models.keys()))
        else:
            _log.warning("hailo_no_models_loaded")

    async def stop(self) -> None:
        """Release the Hailo device and clean up resources."""
        self._available = False
        self._models.clear()
        if self._device is not None:
            try:
                await asyncio.to_thread(self._release_device)
            except Exception:
                _log.warning("hailo_device_release_error", exc_info=True)
            self._device = None
        _log.info("hailo_runtime_stopped")

    def is_available(self) -> bool:
        """Check whether the Hailo device is online and models are loaded."""
        return self._available

    def infer_sync(
        self,
        model_name: str,
        input_data: NDArray[np.uint8],
    ) -> NDArray[np.float32]:
        """Run synchronous, thread-safe inference on a loaded HEF model.

        Serializes concurrent requests via ``threading.Lock`` to avoid
        PCIe bandwidth contention.  This is the primary inference entry
        point for synchronous callers (detectors, feature extractors).

        Args:
            model_name: Registered model identifier.
            input_data: Input tensor as uint8 numpy array.

        Returns:
            Output tensor as float32 numpy array.

        Raises:
            RuntimeError: If model is not loaded or device is unavailable.
        """
        if not self._available:
            msg = "Hailo runtime is not available"
            raise RuntimeError(msg)

        if model_name not in self._models:
            msg = f"Model '{model_name}' not loaded on Hailo device"
            raise RuntimeError(msg)

        t0 = time.monotonic()
        with self._lock:
            result = self._run_pipeline(model_name, input_data)
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        _log.debug(
            "hailo_inference_complete",
            model=model_name,
            elapsed_ms=round(elapsed_ms, 2),
            output_shape=result.shape,
        )
        if elapsed_ms > self._cfg.timeout_ms:
            _log.warning(
                "hailo_inference_exceeded_timeout",
                model=model_name,
                elapsed_ms=round(elapsed_ms, 2),
                timeout_ms=self._cfg.timeout_ms,
            )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _create_device(self) -> Any:
        """Create a Hailo VDevice (blocking)."""
        params = _hailort.VDevice.create_params()
        return _hailort.VDevice(params)

    def _release_device(self) -> None:
        """Release the Hailo VDevice (blocking)."""
        if self._device is not None:
            self._device.release()

    async def _load_hef(self, name: str, hef_path: Path) -> None:
        """Load a single HEF model file and pre-create vstream pipeline.

        Args:
            name: Model identifier for later inference dispatch.
            hef_path: Path to the compiled ``.hef`` file.
        """
        if not hef_path.exists():
            _log.warning("hailo_hef_not_found", model=name, path=str(hef_path))
            return

        try:
            model_info = await asyncio.to_thread(self._load_hef_sync, name, hef_path)
            self._models[name] = model_info
            _log.info("hailo_hef_loaded", model=name, path=str(hef_path))
        except Exception:
            _log.warning(
                "hailo_hef_load_failed",
                model=name,
                path=str(hef_path),
                exc_info=True,
            )

    def _load_hef_sync(self, name: str, hef_path: Path) -> dict[str, Any]:
        """Load HEF and pre-create vstream params (blocking).

        VStream params are created once here and reused across inference
        calls to avoid per-call buffer allocation overhead.
        """
        hef = _hailort.HEF(str(hef_path))
        network_group = self._device.configure(hef)
        ng = network_group[0]

        input_vstream_infos = ng.get_input_vstream_infos()
        output_vstream_infos = ng.get_output_vstream_infos()

        input_params = _hailort.InputVStreamParams.make(
            ng,
            quantized=False,
            format_type=_hailort.FormatType.UINT8,
        )
        output_params = _hailort.OutputVStreamParams.make(
            ng,
            quantized=False,
            format_type=_hailort.FormatType.FLOAT32,
        )

        return {
            "hef": hef,
            "network_group": network_group,
            "input_vstream_infos": input_vstream_infos,
            "output_vstream_infos": output_vstream_infos,
            "input_params": input_params,
            "output_params": output_params,
        }

    def _run_pipeline(
        self,
        model_name: str,
        input_data: NDArray[np.uint8],
    ) -> NDArray[np.float32]:
        """Run inference using pre-created vstream params (blocking).

        Args:
            model_name: Model identifier to dispatch to.
            input_data: Input tensor as uint8 numpy array.

        Returns:
            Output tensor as float32 numpy array.
        """
        model_info = self._models[model_name]
        ng = model_info["network_group"][0]
        input_vstreams = model_info["input_vstream_infos"]
        output_vstreams = model_info["output_vstream_infos"]
        input_params = model_info["input_params"]
        output_params = model_info["output_params"]

        with _hailort.InferVStreams(ng, input_params, output_params) as pipeline:
            input_name = input_vstreams[0].name
            input_dict = {input_name: np.expand_dims(input_data, axis=0)}
            results = pipeline.infer(input_dict)

        output_name = output_vstreams[0].name
        output = cast(NDArray[np.float32], results[output_name].astype(np.float32))
        if output.ndim > 1 and output.shape[0] == 1:
            output = output.squeeze(axis=0)

        return output


# ---------------------------------------------------------------------------
# Mock implementation
# ---------------------------------------------------------------------------


class MockHailoRuntime:
    """Mock Hailo-8 runtime for testing and mock_hardware mode.

    Returns zero-filled arrays of configurable output shapes.

    Args:
        cfg: Hailo-8 configuration.
        output_shapes: Optional override for per-model output shapes.
    """

    # Default output shapes per model — overridable via constructor
    DEFAULT_OUTPUT_SHAPES: ClassVar[dict[str, tuple[int, ...]]] = {
        "yolo": HAILO_MOCK_YOLO_OUTPUT_SHAPE,
        "feature_extractor": (HAILO_MOCK_FEATURE_EXTRACTOR_DIM,),
    }

    def __init__(
        self,
        cfg: HailoConfig,
        output_shapes: dict[str, tuple[int, ...]] | None = None,
    ) -> None:
        """Initialise mock runtime.

        Args:
            cfg: Hailo-8 configuration (used for consistency).
            output_shapes: Optional mapping of model name to output shape.
                Defaults to :attr:`DEFAULT_OUTPUT_SHAPES`.
        """
        self._cfg = cfg
        self._available = False
        self._output_shapes: dict[str, tuple[int, ...]] = (
            output_shapes if output_shapes is not None else dict(self.DEFAULT_OUTPUT_SHAPES)
        )
        _log.info(
            "mock_hailo_runtime_init",
            output_shapes=dict(self._output_shapes),
        )

    async def start(self) -> None:
        """Simulate device discovery and model loading."""
        self._available = True
        _log.info("mock_hailo_runtime_started")

    async def stop(self) -> None:
        """Simulate device release."""
        self._available = False
        _log.info("mock_hailo_runtime_stopped")

    def is_available(self) -> bool:
        """Return mock availability status."""
        return self._available

    def infer_sync(
        self,
        model_name: str,
        input_data: NDArray[np.uint8],
    ) -> NDArray[np.float32]:
        """Return zero-filled output of the expected shape.

        Args:
            model_name: Model identifier.
            input_data: Input tensor (ignored).

        Returns:
            Zero-filled float32 array.

        Raises:
            RuntimeError: If runtime not started or model name unknown.
        """
        if not self._available:
            msg = "Mock Hailo runtime is not available"
            raise RuntimeError(msg)

        if model_name not in self._output_shapes:
            msg = f"Unknown model '{model_name}' in mock runtime"
            raise RuntimeError(msg)

        shape = self._output_shapes[model_name]
        return np.zeros(shape, dtype=np.float32)
