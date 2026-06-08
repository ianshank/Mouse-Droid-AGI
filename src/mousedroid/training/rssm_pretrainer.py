"""Adam pretraining loop for the RSSM dynamics core over sim episode batches."""

from __future__ import annotations

from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from mousedroid.logging.setup import get_logger
from mousedroid.training.sim_episode_generator import EpisodeBatch

if TYPE_CHECKING:
    from mousedroid.world_model.rssm import RSSM

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
    ) -> None:
        """Initialise the pretrainer.

        Args:
            model: The trainable RSSM (exposes ``train_sequence``).
            lr: Adam learning rate.
            grad_clip: Global grad-norm clip.
            amp: Enable mixed precision (only honoured on CUDA).
            device: Target device for the model + batches.
        """
        from mousedroid.world_model.rssm import RawModalityDecoders

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
        # in torch's __all__, so mypy flags attr-defined; the runtime symbol exists.
        self._scaler = torch.amp.GradScaler(enabled=self._amp)  # type: ignore[attr-defined]
        self._device = device

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
                    out = self._model.train_sequence(tensors, self._decoders)
                loss = out["loss"]
                scaled = self._scaler.scale(loss)
                scaled.backward()  # type: ignore[no-untyped-call]  # torch stub gap
                self._scaler.unscale_(self._opt)
                # Clip ALL optimizer-managed params (model + decoders), not just
                # the model — otherwise decoder grads can explode unchecked.
                torch.nn.utils.clip_grad_norm_(
                    chain(self._model.parameters(), self._decoders.parameters()),
                    self._grad_clip,
                )
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
