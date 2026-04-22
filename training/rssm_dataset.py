"""PyTorch Dataset wrapping saved synthetic sequences for RSSM training."""

from __future__ import annotations

from pathlib import Path

import structlog
import torch
from torch import Tensor
from torch.utils.data import Dataset

_log = structlog.get_logger(__name__)

SequenceBatch = dict[str, Tensor]


class RSSMSequenceDataset(Dataset[SequenceBatch]):
    """Dataset of fixed-length observation/action sequences for RSSM training.

    Each item returns a dict containing:
        - vision: ``(seq_len, vision_dim)``
        - ultrasonic: ``(seq_len, ultrasonic_dim)`` (may be zero-width)
        - motor_state: ``(seq_len, motor_state_dim)``
        - valid_mask: ``(seq_len, n_modalities)``
        - lidar: ``(seq_len, lidar_dim)`` (may be zero-width)
        - actions: ``(seq_len, action_dim)``

    Sequences shorter than ``seq_len`` are zero-padded; longer ones are truncated.

    Args:
        data_path: Path to the ``.pt`` file saved by ``SyntheticSequenceGenerator``.
        seq_len: Fixed sequence length for batching.
    """

    def __init__(self, data_path: Path | str, seq_len: int = 50) -> None:
        self._episodes: list[list[dict[str, Tensor]]] = torch.load(
            data_path,
            weights_only=False,
        )
        self._seq_len = seq_len
        _log.info(
            "dataset_loaded",
            path=str(data_path),
            n_episodes=len(self._episodes),
            seq_len=seq_len,
        )

    def __len__(self) -> int:  # noqa: D105
        return len(self._episodes)

    def __getitem__(self, idx: int) -> SequenceBatch:  # noqa: D105
        episode = self._episodes[idx]
        seq_t = min(len(episode), self._seq_len)

        # Infer dims from first transition
        first = episode[0]
        vision_dim = first["vision"].shape[0]
        ultrasonic_dim = first.get("ultrasonic", torch.zeros(0)).shape[0]
        motor_dim = first["motor_state"].shape[0]
        lidar_dim = first.get("lidar", torch.zeros(0)).shape[0]
        valid_mask_dim = first["valid_mask"].shape[0]
        action_dim = first["action"].shape[0]

        batch = {
            "vision": torch.zeros(self._seq_len, vision_dim),
            "ultrasonic": torch.zeros(self._seq_len, ultrasonic_dim),
            "motor_state": torch.zeros(self._seq_len, motor_dim),
            "valid_mask": torch.zeros(self._seq_len, valid_mask_dim),
            "lidar": torch.zeros(self._seq_len, lidar_dim),
            "actions": torch.zeros(self._seq_len, action_dim),
        }

        for t in range(seq_t):
            step = episode[t]
            batch["vision"][t] = step["vision"]
            batch["motor_state"][t] = step["motor_state"]
            batch["valid_mask"][t] = step["valid_mask"]
            batch["actions"][t] = step["action"]
            if batch["ultrasonic"].shape[-1] > 0:
                batch["ultrasonic"][t] = step.get("ultrasonic", torch.zeros(ultrasonic_dim))
            if batch["lidar"].shape[-1] > 0:
                batch["lidar"][t] = step.get("lidar", torch.zeros(lidar_dim))

        return batch
