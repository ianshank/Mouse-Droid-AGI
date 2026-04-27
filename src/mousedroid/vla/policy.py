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

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import structlog
import torch

_log = structlog.get_logger(__name__)

# Default ORT execution-provider chain. Overridable per-policy via config.
DEFAULT_ORT_PROVIDERS: tuple[str, ...] = (
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
)


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
        with torch.no_grad():
            return VLAAction(
                action=self._canned_action.detach().clone(),
                confidence=self._confidence,
            )


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

        Always falls back to ``CPUExecutionProvider`` if it is available
        and the intersection is empty, so warmup never raises on a host
        that has at least the CPU provider.
        """
        chosen = tuple(p for p in requested if p in available)
        if chosen:
            return chosen
        if "CPUExecutionProvider" in available:
            return ("CPUExecutionProvider",)
        # Pathological case: no providers available — let ORT raise.
        return ()

    def warmup(self) -> None:
        """Create the ORT session and run dummy inferences.

        This is the only place that imports ``onnxruntime``. Calling
        :meth:`predict` before :meth:`warmup` will trigger this lazily.

        Raises:
            FileNotFoundError: If ``model_path`` does not exist.
            ImportError: If ``onnxruntime`` is not installed.
        """
        if self._session is not None:
            return  # already warmed
        if not self._model_path.is_file():
            msg = f"ONNX model not found at {self._model_path}"
            raise FileNotFoundError(msg)

        # Lazy import keeps mousedroid.vla.policy free of onnxruntime.
        import onnxruntime as ort

        available = tuple(ort.get_available_providers())
        active = self._resolve_providers(self._requested_providers, available)
        _log.info(
            "distilled_vla_onnx_warmup_start",
            requested_providers=list(self._requested_providers),
            available_providers=list(available),
            active_providers=list(active),
            model_path=str(self._model_path),
        )

        self._session = ort.InferenceSession(
            str(self._model_path),
            providers=list(active),
        )
        self._active_providers = active

        for i in range(self._warmup_iterations):
            self._run_session_with_zeros()
            _log.debug("distilled_vla_onnx_warmup_pass", iteration=i + 1)

        _log.info(
            "distilled_vla_onnx_warmup_complete",
            active_providers=list(active),
            warmup_iterations=self._warmup_iterations,
        )

    def _run_session_with_zeros(self) -> None:
        """Run a single dummy inference using zero-filled inputs.

        Inspects the live session's input metadata so warmup does not
        require knowing latent shapes a priori — those come from the
        ONNX graph itself.
        """
        assert self._session is not None
        feeds: dict[str, Any] = {}
        # numpy is a project dependency (torch -> numpy); import locally
        # to keep the module-level import graph minimal.
        import numpy as _np

        for inp in self._session.get_inputs():
            shape = tuple(d if isinstance(d, int) and d > 0 else 1 for d in (inp.shape or []))
            feeds[inp.name] = _np.zeros(shape, dtype=_np.float32)
        self._session.run([self._action_output_name], feeds)

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
            return VLAAction(action=action_tensor, confidence=self._confidence)
