"""RSSM-refinement on-device learner (Phase 6 WS-E2).

The mouse-droid has NO learned policy net — action selection is MCTS planning
over the RSSM world model. So on-device learning refines the **RSSM** itself
(the dynamics MCTS plans through). :class:`RSSMRefiner` runs a bounded
``train_sequence``-driven refinement on fresh rover experience WITHOUT corrupting
the live/base RSSM:

* The base RSSM is deep-copied into a *candidate* before any gradient flows; every
  optimizer step touches only the candidate. The base parameters are
  bitwise-unchanged on return (pinned by
  ``tests/property/test_rssm_refiner_base_untouched.py``).
* The candidate is refined via :meth:`RSSM.train_sequence` over a ``(B, T, ...)``
  sequence batch (assembled from replay by :func:`build_sequence_batch`) for
  ``cfg.update_steps`` steps at ``cfg.learning_rate``.
* The step is a ``torch.autograd.grad`` manual-SGD update over
  ``candidate.parameters() + decoders.parameters()`` — NO ``loss.backward()`` (so
  no new mypy untyped-call suppression is added) and no optimizer object.
  ``allow_unused=True`` is MANDATORY: the RSSM ``reward_head`` (+ ``prior`` /
  ``observation_decoder``) are not on the recon/KL graph, so their grads come back
  ``None`` and the loop skips them.
* The reconstruction heads (:class:`RawModalityDecoders`) are THROWAWAY: built
  from the candidate's read-only ``cfg``, refined jointly, but NEVER persisted
  into the slot. Only ``candidate.state_dict()`` (the RSSM) is returned, so the
  deployment checkpoint stays byte-identical (no ``decode_*`` keys).
* ``λ=0`` v1: no EWC penalty term. ``train_sequence``'s loss is already
  ``recon + kl_beta*kl``; the RSSM-native diagonal-Fisher EWC follow-up is a clean
  seam for later. (``EWCOnlineLearner`` is NOT a drop-in: it assumes
  ``forward(tensor)`` and omits ``allow_unused`` — it raises on the RSSM.)
* Determinism: the global torch RNG is seeded with ``cfg.scoring_seed`` before the
  refinement so the reparameterisation noise in ``train_sequence`` is reproducible;
  the prior RNG + train-mode state are captured + restored.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from mousedroid.learning.on_device.protocol import OnDeviceUpdateResult
from mousedroid.learning.on_device.seed_states import build_valid_mask
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import ModelConfig, OnDeviceLearningConfig
    from mousedroid.experience.record import MouseDroidExperienceRecord
    from mousedroid.world_model.encoder import MultimodalEncoder
    from mousedroid.world_model.rssm import RSSM

_log = get_logger(__name__)


def build_sequence_batch(
    records: Sequence[MouseDroidExperienceRecord],
    cfg_model: ModelConfig,
    encoder: MultimodalEncoder,
    *,
    sequence_length: int,
    n_episodes: int,
    device: torch.device,
) -> dict[str, Tensor]:
    """Assemble a ``(B, T, ...)`` sequence batch from flat replay records.

    Mirrors the EXACT key set / shape contract of
    :meth:`mousedroid.training.rssm_pretrainer.RSSMPretrainer._to_device` so the
    assembled batch is directly consumable by :meth:`RSSM.train_sequence`:

    * ``motor`` ``(B, T, motor_state_dim)`` — always;
    * ``action`` ``(B, T, action_dim)`` — always;
    * ``valid_mask`` ``(B, T, n_modalities)`` — length 4 (no lidar) or 5 (with
      lidar), synthesized per-record from the LIVE encoder flags (reuses the WS-E1
      :func:`~mousedroid.learning.on_device.seed_states.build_valid_mask`);
    * ``ultrasonic`` ``(B, T, ultrasonic_dim)`` — when ``encoder.ultrasonic_enabled``;
    * ``lidar`` ``(B, T, lidar_dim)`` — when ``encoder.lidar_enabled`` (zero-filled:
      replay records carry no lidar features);
    * ``vision`` ``(B, T, vision_dim)`` — when ``encoder.vision_enabled``; ``(B, T, 0)``
      otherwise (matching the pretrainer's vision-off shape).

    The records are partitioned into ``n_episodes`` contiguous windows of
    ``sequence_length`` consecutive steps (chronological order). At least
    ``n_episodes * sequence_length`` records are required.

    Args:
        records: Flat replay records (chronological order).
        cfg_model: The model config supplying the per-modality dims.
        encoder: The LIVE multimodal encoder whose ``*_enabled`` flags drive the
            mask length + which modality tensors are assembled.
        sequence_length: Temporal length ``T`` of each window.
        n_episodes: Batch dimension ``B`` (number of windows).
        device: Device on which to place every assembled tensor.

    Returns:
        A ``(B, T, ...)`` batch dict consumable by ``RSSM.train_sequence``.

    Raises:
        ValueError: If fewer than ``n_episodes * sequence_length`` records are
            supplied (cannot fill the requested batch).
    """
    needed = n_episodes * sequence_length
    if len(records) < needed:
        msg = (
            f"build_sequence_batch needs >= {needed} records "
            f"(n_episodes={n_episodes} * sequence_length={sequence_length}); "
            f"got {len(records)}"
        )
        raise ValueError(msg)

    motor_rows: list[list[NDArray[np.float32]]] = []
    action_rows: list[list[NDArray[np.float32]]] = []
    mask_rows: list[list[NDArray[np.float32]]] = []
    ultra_rows: list[list[NDArray[np.float32]]] = []
    lidar_rows: list[list[NDArray[np.float32]]] = []
    vision_rows: list[list[NDArray[np.float32]]] = []

    for episode in range(n_episodes):
        start = episode * sequence_length
        window = records[start : start + sequence_length]
        motor_rows.append([np.asarray(r.motor_state, dtype=np.float32) for r in window])
        action_rows.append([np.asarray(r.action, dtype=np.float32) for r in window])
        mask_rows.append([build_valid_mask(r, encoder) for r in window])
        if encoder.ultrasonic_enabled:
            ultra_rows.append([np.asarray([r.distance_m], dtype=np.float32) for r in window])
        if encoder.lidar_enabled:
            lidar_rows.append([np.zeros(cfg_model.lidar_dim, dtype=np.float32) for _ in window])
        if encoder.vision_enabled:
            vision_rows.append(
                [_vision_vector(r.vision_features, cfg_model.vision_dim) for r in window]
            )

    def _stack(rows: list[list[NDArray[np.float32]]]) -> Tensor:
        return torch.as_tensor(np.asarray(rows, dtype=np.float32), device=device)

    batch: dict[str, Tensor] = {
        "motor": _stack(motor_rows),
        "action": _stack(action_rows),
        "valid_mask": _stack(mask_rows),
    }
    if encoder.ultrasonic_enabled:
        batch["ultrasonic"] = _stack(ultra_rows)
    if encoder.lidar_enabled:
        batch["lidar"] = _stack(lidar_rows)
    # ``vision`` is always assembled (shape (B, T, 0) when disabled) to mirror the
    # pretrainer's ``_to_device``; ``train_sequence`` only reads it when enabled.
    if encoder.vision_enabled:
        batch["vision"] = _stack(vision_rows)
    else:
        batch["vision"] = torch.zeros(
            (n_episodes, sequence_length, 0), dtype=torch.float32, device=device
        )
    return batch


def _vision_vector(features: NDArray[np.float32], vision_dim: int) -> NDArray[np.float32]:
    """Return a length-``vision_dim`` vision vector (zero-fill an empty record)."""
    arr = np.asarray(features, dtype=np.float32).reshape(-1)
    if arr.size == vision_dim:
        return arr
    return np.zeros(vision_dim, dtype=np.float32)


class RSSMRefiner:
    """Bounded RSSM-refinement learner implementing :class:`RSSMSequenceLearner`.

    Holds the live/base RSSM and, on :meth:`update`, deep-copies it into a
    candidate, refines the candidate via :meth:`RSSM.train_sequence` over a
    ``(B, T, ...)`` batch using the SPIKE-LOCKED ``autograd.grad`` manual-SGD loop,
    and returns the refined RSSM ``state_dict`` as an
    :class:`~mousedroid.learning.on_device.protocol.OnDeviceUpdateResult`. The base
    RSSM is NEVER mutated.

    Args:
        base_rssm: The live :class:`RSSM` to refine (must expose ``train_sequence``).
            Held by reference; deep-copied per :meth:`update`.
        cfg: On-device learning config (``update_steps`` / ``learning_rate`` /
            ``scoring_seed``; ``ewc_lambda`` is ignored in this λ=0 v1).
        task: Optional task label echoed into the structured log + result metadata
            (e.g. ``"navigation"``), for operator triage across multiple refiners.

    Raises:
        TypeError: If ``base_rssm`` does not expose a callable ``train_sequence``
            (the capability guard — ``DualStreamRSSM`` / ``DualStreamRSSMOnnx``
            have none, so the factory's WS-E0 gate never wires them here).
    """

    def __init__(
        self,
        base_rssm: RSSM,
        cfg: OnDeviceLearningConfig,
        *,
        task: str | None = None,
    ) -> None:
        if not callable(getattr(base_rssm, "train_sequence", None)):
            msg = (
                "RSSMRefiner requires a model exposing a callable train_sequence "
                "(plain RSSM); the live engine lacks it — refinement is unsupported"
            )
            raise TypeError(msg)
        self._base_rssm = base_rssm
        self._cfg = cfg
        self._task = task

    def update(self, batch: Mapping[str, Tensor]) -> OnDeviceUpdateResult:
        """Run ``cfg.update_steps`` bounded refinement steps; return the candidate.

        Args:
            batch: A ``(B, T, ...)`` sequence dict from :func:`build_sequence_batch`.

        Returns:
            An :class:`OnDeviceUpdateResult` whose ``candidate_state_dict`` is the
            refined RSSM ``state_dict`` (NO decoder keys), the final train loss, and
            ``cfg.update_steps``.
        """
        from mousedroid.world_model.rssm import RawModalityDecoders

        steps = self._cfg.update_steps
        lr = self._cfg.learning_rate
        seed = self._cfg.scoring_seed

        _log.info(
            "on_device_refine_start",
            steps=steps,
            learning_rate=lr,
            task=self._task,
            seed=seed,
        )

        # Candidate = deep copy of the base; all gradient flow is confined here so
        # the base RSSM stays bitwise-identical. It lives on the base's device.
        candidate = copy.deepcopy(self._base_rssm)
        device = next(candidate.parameters()).device

        # Seed BEFORE building the decoders AND the rollout: the throwaway decoder
        # heads draw their init from the global RNG, and train_sequence's reparam
        # noise also draws from it, so a fixed seed makes the WHOLE refinement
        # reproducible. The prior RNG + train-mode state are captured + restored so
        # the call is side-effect free for a caller sharing the process RNG.
        #
        # ``torch.manual_seed`` reseeds CPU AND every CUDA generator, and
        # ``train_sequence``'s ``randn_like`` reparam noise draws from the CUDA
        # generator when the candidate is on CUDA — so on a GPU rover the CUDA RNG
        # must be captured + restored too, else a caller sharing the process CUDA
        # RNG is silently perturbed. Guarded by ``cuda.is_available()`` AND the
        # candidate's device so a CPU-only host never touches the CUDA API.
        rng_state = torch.get_rng_state()
        cuda_rng_state = (
            torch.cuda.get_rng_state_all()
            if device.type == "cuda" and torch.cuda.is_available()
            else None
        )
        was_training = candidate.training

        final_loss = 0.0
        try:
            torch.manual_seed(seed)
            # Throwaway reconstruction heads built from the candidate's read-only
            # cfg, placed on the candidate's device so the joint train_sequence
            # never hits a cross-device matmul. Refined jointly, NEVER persisted.
            decoders = RawModalityDecoders(candidate.cfg).to(device)
            params = list(candidate.parameters()) + list(decoders.parameters())
            candidate.train()
            decoders.train()
            for _ in range(steps):
                out = candidate.train_sequence(dict(batch), decoders)
                loss = out["loss"]
                # allow_unused=True is MANDATORY: reward_head / prior /
                # observation_decoder are not on the recon/KL graph, so their grads
                # come back None — omitting the flag would RAISE.
                grads = torch.autograd.grad(loss, params, allow_unused=True)
                with torch.no_grad():
                    for param, grad in zip(params, grads, strict=True):
                        if grad is not None:  # None-grad guard (MANDATORY).
                            param.add_(grad, alpha=-lr)
                final_loss = float(loss.detach())
        finally:
            torch.set_rng_state(rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)
            if not was_training:
                candidate.eval()

        with torch.no_grad():
            candidate_state_dict: dict[str, Tensor] = {
                name: param.detach().clone() for name, param in candidate.state_dict().items()
            }

        _log.info(
            "on_device_refine_complete",
            steps=steps,
            final_loss=final_loss,
            task=self._task,
        )

        return OnDeviceUpdateResult(
            candidate_state_dict=candidate_state_dict,
            train_loss=final_loss,
            n_steps=steps,
            metadata={"learning_rate": lr, "ewc_lambda": 0.0},
        )


__all__ = ["RSSMRefiner", "build_sequence_batch"]
