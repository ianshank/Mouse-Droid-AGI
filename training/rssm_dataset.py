"""PyTorch Dataset wrapping saved synthetic sequences for RSSM training."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset


class RSSMSequenceDataset(Dataset[tuple[Tensor, ...]]):
    """Dataset of fixed-length observation/action sequences for RSSM training.

    Each item returns a tuple of:
        - vision: ``(seq_len, vision_dim)``
        - ultrasonic: ``(seq_len, 1)``
        - motor_state: ``(seq_len, motor_state_dim)``
        - valid_mask: ``(seq_len, 3)``
        - actions: ``(seq_len, action_dim)``

    Sequences shorter than ``seq_len`` are zero-padded; longer ones are truncated.

    Args:
        data_path: Path to the ``.pt`` file saved by ``SyntheticSequenceGenerator``.
        seq_len: Fixed sequence length for batching.
    """

    def __init__(self, data_path: Path | str, seq_len: int = 50) -> None:
        self._episodes: list[list[dict[str, Tensor]]] = torch.load(
            data_path, weights_only=False,
        )
        self._seq_len = seq_len

    def __len__(self) -> int:
        return len(self._episodes)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        episode = self._episodes[idx]
        T = min(len(episode), self._seq_len)

        # Infer dims from first transition
        first = episode[0]
        vision_dim = first["vision"].shape[0]
        motor_dim = first["motor_state"].shape[0]
        action_dim = first["action"].shape[0]

        vision = torch.zeros(self._seq_len, vision_dim)
        ultrasonic = torch.zeros(self._seq_len, 1)
        motor_state = torch.zeros(self._seq_len, motor_dim)
        valid_mask = torch.zeros(self._seq_len, 3)
        actions = torch.zeros(self._seq_len, action_dim)

        for t in range(T):
            step = episode[t]
            vision[t] = step["vision"]
            ultrasonic[t] = step["ultrasonic"]
            motor_state[t] = step["motor_state"]
            valid_mask[t] = step["valid_mask"]
            actions[t] = step["action"]

        return vision, ultrasonic, motor_state, valid_mask, actions
