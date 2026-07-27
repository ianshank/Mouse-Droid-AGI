"""Adam pretraining loop for the RSSM dynamics core over sim episode batches."""

from __future__ import annotations

from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.amp.grad_scaler import GradScaler

from mousedroid.logging.setup import get_logger
from mousedroid.training.sim_episode_generator import EpisodeBatch

if TYPE_CHECKING:
    from mousedroid.config.schema import DriftTrainingConfig
    from mousedroid.world_model.rssm import RSSM, DriftCorrectionHead

_log = get_logger(__name__)


class RSSMPretrainer:
    """Owns the optimizer + epoch loop for ``RSSM.train_sequence``.

    AMP keeps the forward in mixed precision while the KL stays fp32 (handled
    inside ``train_sequence``). The loop is synchronous; the orchestrator runs
    it inside ``asyncio.to_thread`` so the event loop / thermal pause is not
    starved.
    """

    def __init__(
        self,
        model: RSSM,
        *,
        lr: float,
        grad_clip: float,
        amp: bool,
        device: torch.device,
        drift: DriftTrainingConfig | None = None,
    ) -> None:
        """Initialise the pretrainer.

        Args:
            model: The trainable RSSM (exposes ``train_sequence``).
            lr: Adam learning rate.
            grad_clip: Global grad-norm clip.
            amp: Enable mixed precision (only honoured on CUDA).
            device: Target device for the model + batches.
            drift: Optional F-023 corrupted-history training block
                (``training.drift``). ``None`` (default) or ``enabled=False``
                keeps the loop byte-identical to pre-feature. When enabled,
                each batch flips a seeded coin (``corruption_prob``) between
                the standard ``train_sequence`` and
                ``train_sequence_corrupted``; the optional evaluation-only
                ``DriftCorrectionHead`` trains via its SEPARATE loss key.
        """
        from mousedroid.world_model.rssm import DriftCorrectionHead, RawModalityDecoders

        self._model = model.to(device)
        # Pretraining reconstruction heads live here (not on the RSSM) so the
        # deployment model stays byte-identical. They train jointly with the RSSM.
        self._decoders = RawModalityDecoders(model.cfg).to(device)
        self._opt = torch.optim.Adam(
            list(model.parameters()) + list(self._decoders.parameters()), lr=lr
        )
        self._grad_clip = grad_clip
        self._amp = amp and device.type == "cuda"
        # torch.amp.GradScaler is the non-deprecated API but is not re-exported
        # in torch.amp's __all__ (mypy attr-defined). Import the implementation
        # submodule directly — runtime-valid AND mypy-clean, so no suppression.
        self._scaler = GradScaler(enabled=self._amp)
        self._device = device
        # F-023 corrupted-history seam — everything below stays None on the
        # legacy path so pre-feature construction is byte-identical.
        self._drift = drift if drift is not None and drift.enabled else None
        self._drift_head: DriftCorrectionHead | None = None
        self._drift_gen: torch.Generator | None = None
        if self._drift is not None:
            if self._drift.residual_head:
                self._drift_head = DriftCorrectionHead(model.cfg).to(device)
                self._opt.add_param_group({"params": list(self._drift_head.parameters())})
            self._drift_gen = torch.Generator(device="cpu")
            self._drift_gen.manual_seed(self._drift.seed)
            _log.info(
                "rssm_pretrain_drift_enabled",
                corruption_prob=self._drift.corruption_prob,
                max_prefix_frac=self._drift.max_prefix_frac,
                residual_head=self._drift.residual_head,
            )

    def _forward_batch(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Standard or (seeded-coin) corrupted forward for one batch."""
        if self._drift is None or self._drift_gen is None:
            return self._model.train_sequence(tensors, self._decoders)
        corrupt = bool(
            torch.rand(1, generator=self._drift_gen).item() < self._drift.corruption_prob
        )
        if not corrupt:
            return self._model.train_sequence(tensors, self._decoders)
        return self._model.train_sequence_corrupted(
            tensors,
            self._decoders,
            max_prefix_frac=self._drift.max_prefix_frac,
            recovery_weight=self._drift.recovery_weight,
            residual_head=self._drift_head,
            generator=self._drift_gen,
        )

    def _to_device(self, batch: EpisodeBatch) -> dict[str, torch.Tensor]:
        return {
            "motor": batch.motor.to(self._device),
            "ultrasonic": batch.ultrasonic.to(self._device),
            "lidar": batch.lidar.to(self._device),
            "valid_mask": batch.valid_mask.to(self._device),
            "action": batch.action.to(self._device),
            "vision": batch.vision.to(self._device),
        }

    def train(
        self, batches: list[EpisodeBatch], *, epochs: int, checkpoint_path: Path
    ) -> list[float]:
        """Run the Adam loop and write a state-dict checkpoint.

        Args:
            batches: Episode batches to iterate each epoch.
            epochs: Number of passes over ``batches``.
            checkpoint_path: Destination for the model state-dict.

        Returns:
            Per-epoch mean loss history.
        """
        if not batches:
            _log.warning("rssm_pretrain_no_batches")
            return []
        history: list[float] = []
        self._model.train()
        self._decoders.train()
        out: dict[str, torch.Tensor] = {}
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in batches:
                tensors = self._to_device(batch)
                self._opt.zero_grad()
                with torch.autocast(device_type=self._device.type, enabled=self._amp):
                    out = self._forward_batch(tensors)
                loss = out["loss"]
                # The residual head's loss lives under a SEPARATE key (never
                # folded into ``loss`` — the k=0 equality contract); it joins
                # the backward pass only, so logging/history stay comparable.
                residual = out.get("residual_loss")
                backward_target = loss if residual is None else loss + residual
                scaled = self._scaler.scale(backward_target)
                scaled.backward()  # type: ignore[no-untyped-call]
                self._scaler.unscale_(self._opt)
                # Clip ALL optimizer-managed params (model + decoders + the
                # optional drift head), not just the model — otherwise their
                # grads can explode unchecked.
                clip_params = chain(
                    self._model.parameters(),
                    self._decoders.parameters(),
                    self._drift_head.parameters() if self._drift_head is not None else [],
                )
                torch.nn.utils.clip_grad_norm_(clip_params, self._grad_clip)
                self._scaler.step(self._opt)
                self._scaler.update()
                epoch_loss += float(loss.detach())
            mean = epoch_loss / max(1, len(batches))
            history.append(mean)
            _log.info(
                "rssm_pretrain_epoch",
                epoch=epoch,
                loss=mean,
                recon=float(out["recon"]),
                kl=float(out["kl"]),
                posterior_std=float(out["posterior_std"]),
            )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), checkpoint_path)
        _log.info("rssm_pretrain_checkpoint_written", path=str(checkpoint_path))
        return history
