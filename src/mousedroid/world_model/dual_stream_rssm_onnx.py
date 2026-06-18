"""ONNX runtime drop-in for ``DualStreamRSSM`` (B2 Story 2).

The :class:`DualStreamRSSMOnnx` class is a :class:`WorldModelProtocol`
implementation that loads an exported ``.onnx`` and runs ``observe_step``
via ``onnxruntime.InferenceSession`` with the TensorRT → CUDA → CPU
execution-provider fallback chain. The factory engine selector (B2
Story 3) lets operators flip from the PyTorch path to the ONNX path at
runtime via ``cfg.world_model.engine = "onnx_trt"``.

This module mirrors :class:`DistilledVLAOnnx` from
:mod:`mousedroid.vla.policy` — same lazy-import contract (``onnxruntime``
is only loaded inside :meth:`warmup`), same provider-fallback logic
(:meth:`_resolve_providers`), same ``torch.no_grad()`` wrapping at the
call boundary. The shared session-lifecycle logic (provider resolution,
the lazy-import warmup, and the zero-filled warmup pass) *and* the
:data:`~mousedroid.common.onnx_session.DEFAULT_ORT_PROVIDERS` default all
live in the neutral :mod:`mousedroid.common.onnx_session` helper module,
which both wrappers delegate to and which imports neither ``vla`` nor
``world_model``. This module therefore imports nothing from the ``vla``
package — the runtime is fully VLA-independent (pinned by
``tests/unit/world_model/test_onnx_vla_decoupling.py``), so a future edit
that re-introduces a ``world_model -> vla`` import fails loudly.

What this class does NOT implement: :meth:`imagine_step`. The CfC
maintains internal state across imagined rollouts which the export
graph cannot reproduce one-step-at-a-time. ``cfg.world_model.engine =
"onnx_trt"`` ships ``observe_step`` acceleration only; MCTS planning
continues to use the PyTorch model. The factory selector ensures the
right model serves the right call.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor

from mousedroid.common.onnx_session import (
    DEFAULT_ORT_PROVIDERS,
    resolve_providers,
    warmup_session,
)
from mousedroid.config.schema import ModelConfig
from mousedroid.logging.setup import get_logger
from mousedroid.sensing.protocol import ObservationProtocol
from mousedroid.world_model.observation_packer import pack_observation
from mousedroid.world_model.onnx_io import (
    OBSERVE_STEP_INPUT_AUDIO,
    OBSERVE_STEP_INPUT_H,
    OBSERVE_STEP_INPUT_LIDAR,
    OBSERVE_STEP_INPUT_MOTOR,
    OBSERVE_STEP_INPUT_PREV_ACTION,
    OBSERVE_STEP_INPUT_ULTRASONIC,
    OBSERVE_STEP_INPUT_VALID_MASK,
    OBSERVE_STEP_INPUT_VISION,
    OBSERVE_STEP_INPUT_Z,
    OBSERVE_STEP_OUTPUT_NAMES,
)

if TYPE_CHECKING:
    from mousedroid.telemetry.metrics import MetricsRegistry

_log = get_logger(__name__)


class DualStreamRSSMOnnx:
    """``WorldModelProtocol`` impl running on ``onnxruntime``.

    Constructed cheaply (captures config only). ``onnxruntime`` is
    imported lazily inside :meth:`warmup` so ``import
    mousedroid.world_model.*`` stays free of heavyweight runtime
    dependencies.

    Provider fallback: the configured provider chain is intersected with
    ``onnxruntime.get_available_providers()`` so a CUDA-only host
    silently degrades to ``CPUExecutionProvider`` without raising.

    Args:
        model_path: Filesystem path to the exported ``.onnx``.
        cfg: ``ModelConfig`` describing dimensions of the model.
        providers: Requested ORT execution-provider chain. ``None``
            uses :data:`~mousedroid.common.onnx_session.DEFAULT_ORT_PROVIDERS`.
        warmup_iterations: Dummy inferences to run during warmup.
        name: Telemetry name for logging.
        metrics: Optional :class:`MetricsRegistry`. When provided, each
            :meth:`observe_step` brackets ``session.run(...)`` with
            ``time.perf_counter()`` and observes a sample on the
            world-model latency histogram (helper wired by Tier C3.1).
            ``None`` (default) preserves byte-identical pre-PR behavior
            — operators on a deployment without telemetry can omit the
            kwarg and pay zero histogram-observation cost.
    """

    def __init__(
        self,
        *,
        model_path: Path | str,
        cfg: ModelConfig,
        providers: list[str] | tuple[str, ...] | None = None,
        warmup_iterations: int = 1,
        name: str = "dual_stream_rssm_onnx",
        metrics: MetricsRegistry | None = None,
    ) -> None:
        if cfg.cfc_hidden_dim <= 0:
            msg = (
                "DualStreamRSSMOnnx requires cfc_hidden_dim > 0; "
                f"got {cfg.cfc_hidden_dim}. Set ModelConfig.cfc_hidden_dim "
                "in your config YAML."
            )
            raise ValueError(msg)
        if warmup_iterations < 0:
            msg = f"warmup_iterations must be >= 0 (got {warmup_iterations})"
            raise ValueError(msg)

        self._model_path = Path(model_path)
        self._cfg = cfg
        self._requested_providers: tuple[str, ...] = (
            tuple(providers) if providers is not None else DEFAULT_ORT_PROVIDERS
        )
        self._warmup_iterations = warmup_iterations
        self._name = name
        self._metrics = metrics

        self._session: Any | None = None
        self._active_providers: tuple[str, ...] = ()

        # ONNX output names — shared with the exporter via the
        # ``onnx_io`` module so a future rename in either place trips
        # type-check, not silent runtime divergence. ``list(...)`` is
        # required because ``onnxruntime.InferenceSession.run`` mutates
        # the list argument internally on some ORT versions.
        self._output_names = list(OBSERVE_STEP_OUTPUT_NAMES)

        _log.debug(
            "dual_stream_rssm_onnx_initialized",
            model_path=str(self._model_path),
            requested_providers=list(self._requested_providers),
            warmup_iterations=warmup_iterations,
            name=name,
            cfc_hidden_dim=cfg.cfc_hidden_dim,
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

        Thin delegation to
        :func:`mousedroid.common.onnx_session.resolve_providers`. The
        session-lifecycle logic shared with :class:`DistilledVLAOnnx`
        now lives in the neutral ``common/onnx_session`` module, which
        imports neither the ``vla`` nor ``world_model`` package — so this
        runtime stays independent of the VLA module while no longer
        carrying its own copy of the logic.
        """
        return resolve_providers(requested, available)

    def warmup(self) -> None:
        """Create the ORT session and run dummy inferences.

        Idempotent — calling :meth:`warmup` after the session exists is
        a no-op. :meth:`observe_step` triggers this lazily on first call
        so operators don't need to remember the explicit warmup call. The
        session lifecycle is delegated to the neutral
        :func:`mousedroid.common.onnx_session.warmup_session` helper,
        which performs the lazy ``onnxruntime`` import.

        Raises:
            FileNotFoundError: If ``model_path`` doesn't exist.
            ImportError: If ``onnxruntime`` is not installed.
        """
        if self._session is not None:
            return
        self._session, self._active_providers = warmup_session(
            self._model_path,
            self._requested_providers,
            self._warmup_iterations,
            self._output_names,
            log_prefix="dual_stream_rssm_onnx",
        )

    def observe_step(
        self,
        observation: ObservationProtocol,
        prev_action: Tensor,
        h: Tensor,
        z: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, float]:
        """Process one observation step via the ORT session.

        Public surface mirrors :meth:`DualStreamRSSM.observe_step` — same
        inputs, same ``(new_h, new_z, obs_embed, surprise:float)``
        return shape. The ``surprise`` is cast to a Python ``float`` at
        the boundary to match the existing :class:`WorldModelProtocol`
        contract.

        Args:
            observation: Sensor bundle implementing :class:`ObservationProtocol`.
            prev_action: Previous action, shape ``(1, action_dim)``.
            h: Previous combined hidden state, shape ``(1, combined_dim)``.
            z: Previous latent sample, shape ``(1, latent_dim)``.

        Returns:
            ``(new_h, new_z, obs_embed, surprise)``.
        """
        if self._session is None:
            self.warmup()
        assert self._session is not None

        # ONNX consumes numpy arrays on CPU. Run the packer with
        # ``device=cpu`` regardless of where the torch tensors live so
        # the conversion uses the production path even in CUDA hosts.
        packed = pack_observation(observation, self._cfg, device=torch.device("cpu"))

        feeds: dict[str, np.ndarray[Any, Any]] = {
            OBSERVE_STEP_INPUT_VISION: packed.vision.detach().cpu().numpy(),
            OBSERVE_STEP_INPUT_MOTOR: packed.motor.detach().cpu().numpy(),
            OBSERVE_STEP_INPUT_VALID_MASK: packed.valid_mask.detach().cpu().numpy(),
            OBSERVE_STEP_INPUT_PREV_ACTION: prev_action.detach().cpu().numpy(),
            OBSERVE_STEP_INPUT_H: h.detach().cpu().numpy(),
            OBSERVE_STEP_INPUT_Z: z.detach().cpu().numpy(),
        }
        if packed.ultrasonic is not None:
            feeds[OBSERVE_STEP_INPUT_ULTRASONIC] = packed.ultrasonic.detach().cpu().numpy()
        if packed.audio is not None:
            feeds[OBSERVE_STEP_INPUT_AUDIO] = packed.audio.detach().cpu().numpy()
        if packed.lidar is not None:
            feeds[OBSERVE_STEP_INPUT_LIDAR] = packed.lidar.detach().cpu().numpy()

        start = time.perf_counter()
        with torch.no_grad():
            outputs = self._session.run(self._output_names, feeds)
        elapsed = time.perf_counter() - start

        # ORT returns numpy arrays; convert each back to torch on the
        # caller's device (matching h.device for downstream consumers).
        target_device = h.device
        new_h = torch.from_numpy(outputs[0]).to(device=target_device, dtype=torch.float32)
        new_z = torch.from_numpy(outputs[1]).to(device=target_device, dtype=torch.float32)
        obs_embed = torch.from_numpy(outputs[2]).to(device=target_device, dtype=torch.float32)
        surprise_arr = outputs[3]
        # surprise is a scalar tensor exported from observe_step_traceable;
        # ORT may yield shape (1,) or scalar — flatten and grab the value.
        surprise = float(np.asarray(surprise_arr).reshape(-1)[0])

        if self._metrics is not None:
            # Tier C3.1 wired ``observe_world_model_observe_step_seconds``
            # unconditionally on :class:`MetricsRegistry`. The defensive
            # ``getattr(..., None)`` lookup used during the B2 Story 2
            # interim has been removed; the helper now exists on every
            # registry instance. If you encounter ``AttributeError`` here,
            # a downstream module is passing a non-``MetricsRegistry`` stub
            # — make it expose the helper or pass ``metrics=None``.
            self._metrics.observe_world_model_observe_step_seconds(elapsed)

        return new_h, new_z, obs_embed, surprise

    def imagine_step(
        self,
        action: Tensor,
        h: Tensor,
        z: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Not implemented — the ONNX export ships ``observe_step`` only.

        The factory engine selector (B2 Story 3) routes ``imagine_step``
        calls (MCTS rollouts) to the PyTorch model regardless of the
        configured engine. Operators consuming this class directly must
        keep the PyTorch model around for planning.
        """
        del action, h, z
        msg = (
            "DualStreamRSSMOnnx does not implement imagine_step. The MCTS "
            "planner runs on the PyTorch DualStreamRSSM regardless of "
            "cfg.world_model.engine (see B2 Story 3 factory dispatch)."
        )
        raise NotImplementedError(msg)
