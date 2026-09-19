"""TensorRT model compilation for Jetson deployment (Pillar 10).

Provides async TensorRT compilation with disk caching, graceful fallback
to PyTorch when TensorRT is unavailable, and a mock implementation for tests.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import stat
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import torch
import torch.nn as nn
from torch import Tensor

from mousedroid.common.imports import module_importable
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from mousedroid.config.schema import JetsonConfig

_log = get_logger(__name__)

# Probe torch2trt at module level via a *real* guarded import: the positive
# branch of ``_compile_sync`` does ``import torch2trt`` and uses it, so a
# spec-present-but-import-fails install (e.g. a torch2trt built against a
# different TensorRT ABI) must resolve to the JIT-trace fallback, not crash at
# compile time. Spec-only probing (``module_available``) would skip the fallback.
_TORCH2TRT_AVAILABLE: bool = module_importable("torch2trt")


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


#: Placeholder for a runtime component this host cannot report. It is
#: *recorded* rather than omitted on purpose -- see :func:`_runtime_identity`.
_IDENTITY_UNAVAILABLE = "unavailable"


class UntrustedEngineCacheError(RuntimeError):
    """Raised when a cached engine cannot be deserialized safely.

    Loading a ``torch2trt`` engine needs ``torch.load(weights_only=False)``,
    which is arbitrary pickle deserialization -- code execution on load. The
    only thing standing between that and an attacker is the cache directory
    being private to this user, so when it is not, the engine is refused
    rather than executed (peer review D-25).

    A named exception rather than ``assert``: CLAUDE.md makes ruff ``S101``
    blocking in ``src/`` because ``PYTHONOPTIMIZE=1`` in ``Dockerfile.jetson``
    strips asserts, which would turn this guard into no guard on the rover.
    """


def _safe_version(getter: Callable[[], object]) -> str:
    """Resolve one runtime version component, never raising.

    A component that cannot be determined resolves to
    :data:`_IDENTITY_UNAVAILABLE` rather than being dropped. That distinction
    matters: an *omitted* component makes the fingerprint of a host that
    cannot report it collide with one that can, which is precisely the
    stale-engine collision :func:`_runtime_identity` exists to prevent.

    Args:
        getter: Zero-argument callable producing the component.

    Returns:
        The component as a string, or :data:`_IDENTITY_UNAVAILABLE`.
    """
    try:
        value = getter()
    except Exception:
        # Broad on purpose: ANY failure to determine a component -- absent
        # module, missing attribute, a driver query that throws -- means the
        # same thing, "this host cannot report it", and must not propagate
        # into a cache-key computation.
        return _IDENTITY_UNAVAILABLE
    return _IDENTITY_UNAVAILABLE if value is None else str(value)


def _tensorrt_version() -> object:
    """Return the installed TensorRT version, or ``None`` when absent."""
    if not module_importable("tensorrt"):
        return None
    import tensorrt

    return getattr(tensorrt, "__version__", None)


def _torch2trt_version() -> object:
    """Return the installed torch2trt version, or ``None`` when absent."""
    if not _TORCH2TRT_AVAILABLE:
        return None
    import torch2trt

    return getattr(torch2trt, "__version__", None)


def _runtime_identity() -> tuple[str, ...]:
    """Identify the runtime that would *produce* a compiled engine.

    Peer review D-7. ``_model_fingerprint`` described only the *model* --
    class name, architecture string, input shapes, precision, parameter
    count -- so two engines built from the same model under different
    TensorRT, CUDA or driver versions shared a cache key.
    ``compile_model`` treats a matching file as a cache hit, and
    ``docker-compose.jetson.yml`` mounts the cache directory from outside the
    image (the named volume ``mousedroid_tensorrt_cache`` since F-051), so an
    engine survives an image rebuild, a JetPack upgrade or a GPU swap and is
    loaded as current. A stale engine produces *wrong numbers
    rather than a crash*, which is the worst failure mode to debug after the
    fact.

    Every component is resolved through :func:`_safe_version`, so this is
    safe on a host with no GPU and no TensorRT -- which is every CI runner
    and every dev machine, and therefore the common path, not the edge case.

    Returns:
        Ordered, stable components. Order is fixed because it is hashed.
    """
    return (
        f"torch={_safe_version(lambda: torch.__version__)}",
        f"cuda={_safe_version(lambda: torch.version.cuda)}",
        f"tensorrt={_safe_version(_tensorrt_version)}",
        f"torch2trt={_safe_version(_torch2trt_version)}",
        f"compute={_safe_version(_device_capability)}",
    )


def _device_capability() -> object:
    """Return the CUDA compute capability of device 0, or ``None``."""
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_capability()


def cache_dir_is_private(cache_dir: Path) -> bool:
    """Whether ``cache_dir`` is inaccessible to group and other.

    ``load_compiled``'s ``weights_only=False`` fallback has always carried a
    comment claiming the cache directory "should have restricted permissions
    (0700)". Nothing enforced it: ``_save_compiled``'s ``mkdir`` passed no
    ``mode``, so the directory landed at the umask default (``0o755`` on a
    stock host), there was no ``chmod`` anywhere in ``src/``, and the default
    location is mounted from outside the image -- a host bind-mount when this
    was written, the named volume ``mousedroid_tensorrt_cache`` since F-051.
    Either way the directory is one this process did not create and cannot
    vouch for, which is why the check exists rather than a trusted mkdir.
    This function is what makes that comment true (peer review D-25).

    On Windows ``st_mode`` carries no POSIX permission semantics, so the
    question is unanswerable there and the answer is ``True`` -- refusing
    every cache on a platform whose mode bits are meaningless would be a
    portability bug, not a security control.

    Args:
        cache_dir: Directory holding serialized engines.

    Returns:
        ``True`` when the directory is private (or the platform has no POSIX
        mode bits), ``False`` when group or other can reach it.
    """
    if os.name != "posix":
        return True
    try:
        mode = stat.S_IMODE(cache_dir.stat().st_mode)
    except OSError:
        # A directory we cannot stat is one we cannot vouch for.
        return False
    return not (mode & (stat.S_IRWXG | stat.S_IRWXO))


def _model_fingerprint(
    model: nn.Module,
    input_shapes: dict[str, list[int]],
    precision: str,
) -> str:
    """Compute a deterministic fingerprint for cache key generation.

    Covers both *what* was compiled (the model, shapes and precision) and
    *what compiled it* (:func:`_runtime_identity`).

    Adding the runtime identity **invalidates every previously cached
    engine** by design: old keys simply never match again, so a stale engine
    is ignored rather than mis-loaded. The cost is one recompile per model on
    the first run after an upgrade. Note for whoever wires
    ``OptimizedInference`` into the tick: that compile is lazy, driven from
    ``OptimizedInference.infer``, so a cold recompile lands *inside* an
    inference call. That is harmless today, because nothing constructs
    ``OptimizedInference`` at all, and a 30 Hz budget hazard the day it does.

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
    for component in _runtime_identity():
        h.update(component.encode())
    return h.hexdigest()[:16]


def _trace_model(model: nn.Module, sample_input: Tensor) -> Any:
    """Fallback tracing helper isolated from torch's incomplete stubs.

    Args:
        model: PyTorch model to trace.
        sample_input: Sample input tensor.

    Returns:
        Traced model via ``torch.jit.trace``.
    """
    return torch.jit.trace(model, sample_input)  # type: ignore[no-untyped-call]  # torch.jit is untyped


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

        # Check disk cache first. A cache we cannot vouch for is a MISS, not
        # an error: a cache is an optimization, so declining to trust an entry
        # must mean *rebuild it*, never *fail*. Raising here would drop every
        # existing rig to eager PyTorch through
        # ``OptimizedInference._ensure_compiled``'s ``except Exception``,
        # logged once at warning level -- a silent performance cliff nobody
        # would notice. Recompiling is always correct, and it self-heals:
        # ``_save_compiled`` re-creates the directory with private
        # permissions on the way out (peer review D-25).
        if cache_path.exists() and not cache_dir_is_private(self._cache_dir):
            _log.warning(
                "tensorrt_cache_not_private_recompiling",
                path=str(self._cache_dir),
                hint=(
                    "cache directory is group- or world-accessible; the "
                    "cached engine is being rebuilt rather than deserialized"
                ),
            )
        elif cache_path.exists():
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
            # ``mode`` is masked by the umask, so mkdir alone does NOT achieve
            # 0700 -- it lands at 0o755 on a stock host. The explicit chmod is
            # what makes it true, and it also hardens a directory an earlier
            # release created world-readable. Both are POSIX-only concepts.
            path.parent.mkdir(parents=True, exist_ok=True, mode=stat.S_IRWXU)
            if os.name == "posix":
                try:
                    os.chmod(path.parent, stat.S_IRWXU)
                except OSError:
                    # A cache directory owned by another user -- the likely
                    # case for the externally-mounted default, where Docker
                    # created the named volume root-owned and the container
                    # runs unprivileged. Do NOT abort the save: it may still
                    # succeed, and the directory may already be private
                    # because someone else made it so.
                    #
                    # Logged distinctly rather than swallowed by the caller's
                    # generic handler, because the failure is otherwise
                    # invisible and self-perpetuating: an unprivate directory
                    # makes every compile_model a cache miss, so the rover
                    # silently recompiles on every run with no indication why.
                    _log.warning(
                        "tensorrt_cache_chmod_failed",
                        path=str(path.parent),
                        hint=(
                            "cannot restrict the engine cache to this user; "
                            "engines will be recompiled every run rather than "
                            "deserialized. Fix the directory's ownership or "
                            "point jetson.tensorrt_cache_dir somewhere writable"
                        ),
                    )
            if isinstance(compiled, torch.jit.ScriptModule):
                torch.jit.save(compiled, str(path))
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
                return torch.jit.load(str(path))  # type: ignore[no-untyped-call]  # torch.jit is untyped
            except RuntimeError:
                # A torch2trt engine is not TorchScript, so this is the NORMAL
                # path for one, not an error path -- ``torch.jit.load`` raises
                # ``RuntimeError`` for it exactly as it does for a corrupt or
                # version-mismatched file.
                #
                # SECURITY: ``weights_only=False`` is arbitrary pickle
                # deserialization -- code execution on load. It is required to
                # load torch2trt modules, and the only thing bounding it is
                # the cache directory being private to this user. That is now
                # created and verified rather than merely asserted in a
                # comment (peer review D-25); refuse rather than execute when
                # it does not hold.
                #
                # Narrowing the catch does NOT by itself tell a
                # version-mismatched engine apart from a hostile pickle --
                # both arrive here as ``RuntimeError``. The runtime identity
                # in ``_model_fingerprint`` prevents the first; this guard
                # prevents the second.
                if not cache_dir_is_private(path.parent):
                    msg = (
                        f"refusing to deserialize {path}: its directory is "
                        "group- or world-accessible, so an untrusted pickle "
                        "could be loaded as a compiled engine"
                    )
                    raise UntrustedEngineCacheError(msg) from None
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
