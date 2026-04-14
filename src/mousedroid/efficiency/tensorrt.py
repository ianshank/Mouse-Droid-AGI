"""TensorRT model compilation for Jetson deployment (Pillar 10).

Provides async TensorRT compilation with disk caching, graceful fallback
to PyTorch when TensorRT is unavailable, and a mock implementation for tests.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import torch
import torch.nn as nn
from torch import Tensor

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import JetsonConfig

_log = get_logger(__name__)

# Try importing torch2trt at module level; concrete classes handle absence.
_TORCH2TRT_AVAILABLE: bool
try:
    import torch2trt as _torch2trt  # noqa: F401

    _TORCH2TRT_AVAILABLE = True
except ImportError:
    _TORCH2TRT_AVAILABLE = False


@runtime_checkable
class TensorRTCompilerProtocol(Protocol):
    """Interface for TensorRT model compilation."""

    async def compile_model(
        self,
        model: nn.Module,
        input_shapes: dict[str, list[int]],
        precision: str,
    ) -> Any:
        """Compile a PyTorch model to TensorRT engine.

        Args:
            model: PyTorch model to compile.
            input_shapes: Mapping of input names to their shapes.
            precision: Target precision (``"fp16"`` or ``"fp32"``).

        Returns:
            Compiled TensorRT engine or equivalent optimized model.
        """
        ...

    async def load_compiled(self, path: Path) -> Any:
        """Load a previously compiled TensorRT engine from disk.

        Args:
            path: Path to the serialized engine file.

        Returns:
            Deserialized TensorRT engine.
        """
        ...

    def is_available(self) -> bool:
        """Check whether TensorRT compilation is available on this system.

        Returns:
            True if TensorRT backend can be used.
        """
        ...


def _model_fingerprint(
    model: nn.Module,
    input_shapes: dict[str, list[int]],
    precision: str,
) -> str:
    """Compute a deterministic fingerprint for cache key generation.

    Args:
        model: PyTorch model.
        input_shapes: Input shape specification.
        precision: Target precision.

    Returns:
        Hex digest string identifying this compilation configuration.
    """
    h = hashlib.sha256()
    h.update(model.__class__.__name__.encode())
    # Include full architecture string for structural uniqueness.
    h.update(str(model).encode())
    for name, shape in sorted(input_shapes.items()):
        h.update(f"{name}:{shape}".encode())
    h.update(precision.encode())
    # Include parameter count as an additional structural fingerprint.
    param_count = sum(p.numel() for p in model.parameters())
    h.update(str(param_count).encode())
    return h.hexdigest()[:16]


def _trace_model(model: nn.Module, sample_input: Tensor) -> Any:
    """Fallback tracing helper isolated from torch's incomplete stubs.

    Args:
        model: PyTorch model to trace.
        sample_input: Sample input tensor.

    Returns:
        Traced model via ``torch.jit.trace``.
    """
    return torch.jit.trace(model, sample_input)  # type: ignore[no-untyped-call]


class JetsonTensorRTCompiler:
    """Compile PyTorch models to TensorRT for efficient Jetson inference.

    Uses ``torch2trt`` when available, with disk caching of compiled engines.
    Falls back gracefully to PyTorch JIT tracing when TensorRT is unavailable.

    Args:
        cfg: Jetson hardware configuration.
    """

    def __init__(self, cfg: JetsonConfig) -> None:
        self._enabled = cfg.tensorrt_enabled
        self._precision = cfg.precision
        self._workspace_gb = cfg.workspace_gb
        self._cache_dir = Path(cfg.tensorrt_cache_dir)
        self._dla_enabled = cfg.dla_enabled

    def is_available(self) -> bool:
        """Check whether TensorRT compilation is available.

        Returns:
            True if enabled and torch2trt is installed.
        """
        return self._enabled and _TORCH2TRT_AVAILABLE

    def _cache_path(self, fingerprint: str) -> Path:
        """Build cache file path for a given fingerprint.

        Args:
            fingerprint: Model compilation fingerprint.

        Returns:
            Path to the cached engine file.
        """
        return self._cache_dir / f"engine_{fingerprint}.pth"

    async def compile_model(
        self,
        model: nn.Module,
        input_shapes: dict[str, list[int]],
        precision: str,
    ) -> Any:
        """Compile model to TensorRT with disk caching.

        Args:
            model: PyTorch model to compile.
            input_shapes: Mapping of input names to their shapes.
            precision: Target precision (``"fp16"`` or ``"fp32"``).

        Returns:
            Compiled TensorRT engine, JIT-traced model, or original model.
        """
        if not self._enabled:
            _log.info("tensorrt_compilation_disabled")
            return model

        fingerprint = _model_fingerprint(model, input_shapes, precision)
        cache_path = self._cache_path(fingerprint)

        # Check disk cache first.
        if cache_path.exists():
            _log.info(
                "tensorrt_cache_hit",
                fingerprint=fingerprint,
                path=str(cache_path),
            )
            return await self.load_compiled(cache_path)

        _log.info(
            "tensorrt_compiling",
            precision=precision,
            workspace_gb=self._workspace_gb,
            fingerprint=fingerprint,
        )

        start = time.monotonic()
        compiled = await self._do_compile(model, input_shapes, precision)
        elapsed_ms = (time.monotonic() - start) * 1000.0

        _log.info(
            "tensorrt_compilation_complete",
            elapsed_ms=round(elapsed_ms, 1),
            fingerprint=fingerprint,
            backend="torch2trt" if _TORCH2TRT_AVAILABLE else "jit_trace",
        )

        # Persist to disk cache.
        await self._save_compiled(compiled, cache_path)
        return compiled

    async def _do_compile(
        self,
        model: nn.Module,
        input_shapes: dict[str, list[int]],
        precision: str,
    ) -> Any:
        """Run the actual compilation in a thread pool.

        Args:
            model: PyTorch model.
            input_shapes: Input shape mapping.
            precision: Target precision.

        Returns:
            Compiled model.
        """

        # Build sample inputs for all declared input shapes.
        def _build_samples(device: str) -> list[Tensor]:
            return [torch.randn(*shape, device=device) for shape in input_shapes.values()]

        if not _TORCH2TRT_AVAILABLE:
            _log.warning("torch2trt_not_available_falling_back_to_jit_trace")
            samples = _build_samples("cpu")
            model.eval()
            # JIT trace accepts a tuple of inputs for multi-input models.
            trace_input: Any = samples[0] if len(samples) == 1 else tuple(samples)
            return await asyncio.to_thread(_trace_model, model, trace_input)

        # torch2trt compilation is CPU-bound; offload to thread.
        def _compile_sync() -> Any:
            import torch2trt

            device = "cuda" if torch.cuda.is_available() else "cpu"
            samples = _build_samples(device)
            model.eval()

            fp16_mode = precision == "fp16"
            return torch2trt.torch2trt(
                model,
                samples,
                fp16_mode=fp16_mode,
                max_workspace_size=int(self._workspace_gb * (1 << 30)),
                use_onnx=False,
            )

        return await asyncio.to_thread(_compile_sync)

    async def _save_compiled(self, compiled: Any, path: Path) -> None:
        """Save compiled engine to disk.

        Args:
            compiled: Compiled TensorRT engine or traced model.
            path: Destination path.
        """

        def _save_sync() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(compiled, torch.jit.ScriptModule):
                torch.jit.save(compiled, str(path))  # type: ignore[no-untyped-call]
            else:
                torch.save(compiled, str(path))

        try:
            await asyncio.to_thread(_save_sync)
            _log.info("tensorrt_engine_cached", path=str(path))
        except (OSError, RuntimeError):
            _log.warning("tensorrt_cache_save_failed", path=str(path))

    async def load_compiled(self, path: Path) -> Any:
        """Load a compiled engine from disk.

        Args:
            path: Path to the serialized engine.

        Returns:
            Deserialized engine.

        Raises:
            FileNotFoundError: If the cache file does not exist.
        """
        if not path.exists():
            msg = f"Compiled engine not found: {path}"
            raise FileNotFoundError(msg)

        def _load_sync() -> Any:
            try:
                return torch.jit.load(str(path))  # type: ignore[no-untyped-call]
            except Exception:
                # weights_only=False is required to load torch2trt modules.
                # SECURITY: only load from the local tensorrt_cache_dir which
                # should have restricted permissions (0700).
                # SECURITY: weights_only=False is required to load
                # torch2trt modules.  Restrict tensorrt_cache_dir to 0700.
                return torch.load(str(path), weights_only=False)

        result = await asyncio.to_thread(_load_sync)
        _log.info("tensorrt_engine_loaded", path=str(path))
        return result


class MockTensorRTCompiler:
    """Mock TensorRT compiler for testing — no real compilation.

    Returns the original model unchanged and simulates cache behaviour.
    """

    def __init__(self) -> None:
        self._compiled_models: dict[str, nn.Module] = {}

    def is_available(self) -> bool:
        """Always returns True for testing.

        Returns:
            True.
        """
        return True

    async def compile_model(
        self,
        model: nn.Module,
        input_shapes: dict[str, list[int]],
        precision: str,
    ) -> Any:
        """Return the model unchanged, simulating compilation.

        Args:
            model: PyTorch model.
            input_shapes: Input shape spec.
            precision: Target precision.

        Returns:
            The original model.
        """
        fingerprint = _model_fingerprint(model, input_shapes, precision)
        self._compiled_models[fingerprint] = model
        _log.info(
            "mock_tensorrt_compile",
            fingerprint=fingerprint,
            precision=precision,
        )
        return model

    async def load_compiled(self, path: Path) -> Any:
        """Simulate loading from disk.

        Args:
            path: Path (ignored).

        Returns:
            A simple linear model as placeholder.
        """
        _log.info("mock_tensorrt_load", path=str(path))
        return nn.Linear(1, 1)

    @property
    def compiled_models(self) -> dict[str, nn.Module]:
        """Access compiled model cache for test assertions.

        Returns:
            Dictionary of fingerprint to compiled model.
        """
        return self._compiled_models


# Keep the legacy class for backwards compatibility.
class TensorRTOptimizer:
    """Convert PyTorch models to TensorRT for efficient Jetson inference.

    Precision and workspace size read from ``JetsonConfig``.
    This is the original synchronous API retained for backwards compatibility.
    """

    def __init__(self, cfg: JetsonConfig) -> None:
        """Initialise TensorRT optimizer.

        Args:
            cfg: Jetson hardware configuration.
        """
        self.enabled = cfg.tensorrt_enabled
        self.precision = cfg.precision
        self.workspace_gb = cfg.workspace_gb
        self.dla_enabled = cfg.dla_enabled

    def optimize(self, model: nn.Module, sample_input: Tensor) -> Any:
        """Convert model to TensorRT engine.

        Args:
            model: PyTorch model to optimize.
            sample_input: Sample input tensor for tracing.

        Returns:
            Optimized model or engine. Returns original model if disabled.
        """
        if not self.enabled:
            _log.info("tensorrt_disabled")
            return model

        _log.info(
            "tensorrt_optimize",
            precision=self.precision,
            workspace_gb=self.workspace_gb,
            dla=self.dla_enabled,
        )
        return self._compile(model, sample_input)

    def _compile(self, model: nn.Module, sample_input: Tensor) -> Any:  # pragma: no cover
        """Compile model to TensorRT (requires tensorrt package).

        Args:
            model: PyTorch model.
            sample_input: Sample input for tracing.

        Returns:
            TensorRT engine or traced model.
        """
        try:
            import torch_tensorrt

            return torch_tensorrt.compile(
                model,
                inputs=[sample_input],
                enabled_precisions={self.precision},
                workspace_size=int(self.workspace_gb * (1 << 30)),
            )
        except ImportError:
            _log.warning("torch_tensorrt_not_available_falling_back_to_traced")
            return _trace_model(model, sample_input)
