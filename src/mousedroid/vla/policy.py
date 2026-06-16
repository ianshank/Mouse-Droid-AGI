"""VLA policy types, protocol, and ``MockVLA`` reference implementation.

This module is **import-graph-isolated**: it must not import any heavyweight
dependencies (``torch`` is the existing project-wide dependency; no
``onnxruntime``, no ``transformers`` here). The Phase 3b
``DistilledVLAOnnx`` implementation lives alongside ``MockVLA`` and
lazy-imports its runtime in :meth:`DistilledVLAOnnx.warmup`.

All thresholds, dimensions, and timeouts come from
:class:`mousedroid.config.schema.VLAConfig` /
:class:`mousedroid.config.schema.LoopConfig`. No values are hardcoded.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import structlog
import torch

# ``DEFAULT_ORT_PROVIDERS`` now lives in the neutral ``common.onnx_session``
# module; the redundant ``as`` alias re-exports it explicitly (for ``mypy
# --strict``'s no-implicit-reexport) so ``from mousedroid.vla.policy import
# DEFAULT_ORT_PROVIDERS`` and ``from mousedroid.vla import DEFAULT_ORT_PROVIDERS``
# keep working for backward compatibility.
from mousedroid.common.onnx_session import (
    DEFAULT_ORT_PROVIDERS as DEFAULT_ORT_PROVIDERS,
)
from mousedroid.common.onnx_session import (
    resolve_providers,
    warmup_session,
)

if TYPE_CHECKING:
    from mousedroid.telemetry.metrics import MetricsRegistry

_log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class VLAObservation:
    """Inputs to a VLA policy.

    Wraps the current latent state plus an optional natural-language
    instruction. The orchestrator constructs this each tick.
    """

    h: torch.Tensor
    z: torch.Tensor
    instruction: str = ""


@dataclass(frozen=True)
class VLAAction:
    """Output of a VLA policy.

    Carries the action tensor (shape ``[action_dim]``) and an optional
    confidence in ``[0.0, 1.0]`` for downstream gating.
    """

    action: torch.Tensor
    confidence: float = 1.0


@runtime_checkable
class VLAPolicyProtocol(Protocol):
    """Protocol every VLA policy implementation must satisfy."""

    @property
    def name(self) -> str:
        """Human-readable policy name (used in telemetry)."""
        ...

    def predict(self, observation: VLAObservation) -> VLAAction:
        """Compute the next action for ``observation``.

        Implementations must run inference under ``torch.no_grad()`` and
        return a tensor with the configured ``model.action_dim``.
        """
        ...


class MockVLA:
    """Deterministic, zero-dependency VLA policy used for tests and dry-runs.

    The action returned is configurable; default is a zero action of the
    requested ``action_dim``. Useful as an upper-bound timing reference
    (effectively free) and as a structural test target for the
    orchestrator's VLA branch.
    """

    def __init__(
        self,
        *,
        action_dim: int,
        canned_action: torch.Tensor | None = None,
        confidence: float = 1.0,
        name: str = "mock_vla",
        metrics: MetricsRegistry | None = None,
    ) -> None:
        """Initialize a ``MockVLA``.

        Args:
            action_dim: Dimensionality of the action vector. Must match
                ``model.action_dim`` from the active settings.
            canned_action: Optional fixed action tensor. When ``None`` a
                zero action of length ``action_dim`` is returned.
            confidence: Confidence value emitted with each ``VLAAction``.
                Must be in ``[0.0, 1.0]``.
            name: Telemetry name; defaults to ``"mock_vla"``.
            metrics: Optional :class:`MetricsRegistry`. When provided, each
                ``predict()`` call records an observation on
                ``mousedroid_vla_inference_seconds`` for end-to-end metric-
                pipeline visibility on mock-hardware deployments. ``None``
                (default) preserves byte-identical pre-PR-A2.1 behavior.

        Raises:
            ValueError: If ``action_dim`` is not positive, ``confidence``
                is outside ``[0.0, 1.0]``, or ``canned_action`` shape
                does not match ``action_dim``.
        """
        if action_dim <= 0:
            msg = f"action_dim must be > 0 (got {action_dim})"
            raise ValueError(msg)
        if not 0.0 <= confidence <= 1.0:
            msg = f"confidence must be in [0.0, 1.0] (got {confidence})"
            raise ValueError(msg)
        if canned_action is not None and canned_action.shape != (action_dim,):
            msg = (
                f"canned_action shape {tuple(canned_action.shape)} does not match "
                f"(action_dim,) = ({action_dim},)"
            )
            raise ValueError(msg)

        self._action_dim = action_dim
        self._canned_action = (
            canned_action.detach().clone()
            if canned_action is not None
            else torch.zeros(action_dim, dtype=torch.float32)
        )
        self._confidence = confidence
        self._name = name
        self._metrics = metrics
        _log.debug(
            "mock_vla_initialized",
            action_dim=action_dim,
            confidence=confidence,
            name=name,
        )

    @property
    def name(self) -> str:
        """Telemetry name."""
        return self._name

    def predict(self, observation: VLAObservation) -> VLAAction:
        """Return the canned action regardless of ``observation``.

        Args:
            observation: Ignored. Accepted to satisfy the protocol.

        Returns:
            A :class:`VLAAction` with a fresh clone of the canned action.
        """
        del observation  # unused — MockVLA is stateless and deterministic
        start = time.perf_counter()
        with torch.no_grad():
            action = self._canned_action.detach().clone()
        elapsed = time.perf_counter() - start
        if self._metrics is not None:
            self._metrics.observe_vla_inference_seconds(elapsed)
        return VLAAction(action=action, confidence=self._confidence)


class DistilledVLAOnnx:
    """Distilled VLA student running on ONNX Runtime (Phase 3b).

    The class is constructed cheaply: it captures configuration only.
    All ``onnxruntime`` work happens in :meth:`warmup` (lazy import) so
    that ``import mousedroid.vla.policy`` stays free of heavyweight
    runtime dependencies (verified by an import-graph isolation test).

    Provider fallback: the configured provider chain is intersected with
    ``onnxruntime.get_available_providers()`` so that a CUDA-only host
    silently degrades to ``CPUExecutionProvider`` without raising.
    """

    def __init__(
        self,
        *,
        model_path: Path | str,
        action_dim: int,
        providers: list[str] | tuple[str, ...] | None = None,
        h_input_name: str = "h",
        z_input_name: str = "z",
        action_output_name: str = "action",
        warmup_iterations: int = 1,
        confidence: float = 1.0,
        name: str = "distilled_vla_onnx",
        metrics: MetricsRegistry | None = None,
    ) -> None:
        """Capture ONNX configuration; defer session creation to warmup.

        Args:
            model_path: Filesystem path to the ``.onnx`` graph.
            action_dim: Expected dimensionality of the action vector;
                used to validate session output shape on every call.
            providers: Requested ORT execution-provider chain. ``None``
                uses :data:`DEFAULT_ORT_PROVIDERS`.
            h_input_name: ONNX input name bound to ``observation.h``.
            z_input_name: ONNX input name bound to ``observation.z``.
            action_output_name: ONNX output name producing the action.
            warmup_iterations: Number of dummy passes during warmup.
            confidence: Static confidence emitted with every action.
            name: Telemetry name.
            metrics: Optional :class:`MetricsRegistry`. When provided, each
                ``predict()`` call brackets the ONNX inference with
                ``time.perf_counter()`` and records an observation on
                ``mousedroid_vla_inference_seconds`` outside the
                ``torch.no_grad()`` block. ``None`` (default) preserves
                byte-identical pre-PR-A2.1 behavior.

        Raises:
            ValueError: If ``action_dim`` is not positive, ``confidence``
                is outside ``[0.0, 1.0]``, or ``warmup_iterations`` is
                negative.
        """
        if action_dim <= 0:
            msg = f"action_dim must be > 0 (got {action_dim})"
            raise ValueError(msg)
        if not 0.0 <= confidence <= 1.0:
            msg = f"confidence must be in [0.0, 1.0] (got {confidence})"
            raise ValueError(msg)
        if warmup_iterations < 0:
            msg = f"warmup_iterations must be >= 0 (got {warmup_iterations})"
            raise ValueError(msg)

        self._model_path = Path(model_path)
        self._action_dim = action_dim
        self._requested_providers: tuple[str, ...] = (
            tuple(providers) if providers is not None else DEFAULT_ORT_PROVIDERS
        )
        self._h_input_name = h_input_name
        self._z_input_name = z_input_name
        self._action_output_name = action_output_name
        self._warmup_iterations = warmup_iterations
        self._confidence = confidence
        self._name = name
        self._metrics = metrics

        self._session: Any | None = None
        self._active_providers: tuple[str, ...] = ()

        _log.debug(
            "distilled_vla_onnx_initialized",
            model_path=str(self._model_path),
            action_dim=action_dim,
            requested_providers=list(self._requested_providers),
            warmup_iterations=warmup_iterations,
            name=name,
        )

    @property
    def name(self) -> str:
        """Telemetry name."""
        return self._name

    @property
    def active_providers(self) -> tuple[str, ...]:
        """Concrete ORT providers chosen after :meth:`warmup`."""
        return self._active_providers

    @staticmethod
    def _resolve_providers(
        requested: tuple[str, ...],
        available: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Intersect ``requested`` with ``available`` preserving order.

        Thin delegation to :func:`mousedroid.common.onnx_session.resolve_providers`
        (the neutral, VLA-independent helper); see there for the fallback
        contract.
        """
        return resolve_providers(requested, available)

    def warmup(self) -> None:
        """Create the ORT session and run dummy inferences.

        Delegates the session lifecycle to the neutral
        :func:`mousedroid.common.onnx_session.warmup_session` helper —
        ``onnxruntime`` is imported lazily there. Calling :meth:`predict`
        before :meth:`warmup` will trigger this lazily.

        Raises:
            FileNotFoundError: If ``model_path`` does not exist.
            ImportError: If ``onnxruntime`` is not installed.
        """
        if self._session is not None:
            return  # already warmed
        self._session, self._active_providers = warmup_session(
            self._model_path,
            self._requested_providers,
            self._warmup_iterations,
            [self._action_output_name],
            log_prefix="distilled_vla_onnx",
        )

    def predict(self, observation: VLAObservation) -> VLAAction:
        """Run a single forward pass of the distilled VLA student.

        Lazily warms up the session on first call. Inference is wrapped
        in :func:`torch.no_grad` per the protocol contract.

        Args:
            observation: Latent state to evaluate.

        Returns:
            A :class:`VLAAction` whose ``action`` tensor has shape
            ``(action_dim,)`` and dtype ``float32``.

        Raises:
            ValueError: If the ONNX output shape disagrees with the
                configured ``action_dim``.
            FileNotFoundError: From :meth:`warmup` when the model file
                is missing.
            ImportError: From :meth:`warmup` when ``onnxruntime`` is
                not installed.
        """
        if self._session is None:
            self.warmup()
        assert self._session is not None

        start = time.perf_counter()
        with torch.no_grad():
            h_np = observation.h.detach().cpu().numpy().astype("float32", copy=False)
            z_np = observation.z.detach().cpu().numpy().astype("float32", copy=False)
            outputs = self._session.run(
                [self._action_output_name],
                {self._h_input_name: h_np, self._z_input_name: z_np},
            )
            action_np = outputs[0]
            # ``torch.from_numpy`` shares memory with the underlying numpy
            # buffer, which is owned by the ORT session and may be mutated by
            # the next ``run`` call. Clone so the returned action is stable.
            action_tensor = torch.from_numpy(action_np).to(torch.float32).reshape(-1).clone()
            if action_tensor.shape != (self._action_dim,):
                msg = (
                    f"DistilledVLAOnnx output shape {tuple(action_tensor.shape)} "
                    f"does not match (action_dim,) = ({self._action_dim},)"
                )
                raise ValueError(msg)
            result = VLAAction(action=action_tensor, confidence=self._confidence)
        elapsed = time.perf_counter() - start
        if self._metrics is not None:
            # Observation lives outside the no_grad block (no tensor ops);
            # MetricsRegistry guards NaN / negative samples internally.
            self._metrics.observe_vla_inference_seconds(elapsed)
        return result
