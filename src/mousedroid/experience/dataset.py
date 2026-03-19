"""Offline RL dataset loader — reads LMDB experience records into PyTorch tensors.

Bridges the LMDB experience store to offline RL training by providing
batch iteration over stored transitions ``(s, a, r, s', done)``.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import lmdb
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from mousedroid.constants import GB_TO_BYTES
from mousedroid.experience.record import MouseDroidExperienceRecord
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import ExperienceConfig, ModelConfig

_log = get_logger(__name__)


class OfflineRLDataset:
    """Dataset loader for offline RL training from LMDB experience records.

    Reads stored experience records and yields batches of
    ``(state, action, reward, next_state, done)`` tensors suitable
    for offline RL algorithms (CQL, IQL, etc.).

    Args:
        experience_cfg: LMDB storage configuration.
        model_cfg: Model dimensions for state vector construction.
        device: Target torch device for tensors.
    """

    def __init__(
        self,
        experience_cfg: ExperienceConfig,
        model_cfg: ModelConfig,
        device: torch.device | None = None,
    ) -> None:
        self._path = Path(experience_cfg.path)
        self._map_size = experience_cfg.map_size_gb * GB_TO_BYTES
        self._vision_dim = model_cfg.vision_dim
        self._motor_dim = model_cfg.motor_state_dim
        self._action_dim = model_cfg.action_dim
        self._device = device or torch.device("cpu")
        self._env: lmdb.Environment | None = None
        self._keys: list[bytes] = []
        self._cached_transitions: tuple[
            NDArray[Any],
            NDArray[Any],
            NDArray[Any],
            NDArray[Any],
            NDArray[Any],
        ] | None = None
        self._cached_terminal_gap_s: float | None = None

    @property
    def state_dim(self) -> int:
        """Dimension of concatenated state vector (vision + distance + motor)."""
        return self._vision_dim + 1 + self._motor_dim

    def open(self) -> None:
        """Open LMDB environment and cache all keys."""
        if not self._path.exists():
            msg = f"Experience database not found: {self._path}"
            raise FileNotFoundError(msg)

        self._env = lmdb.open(
            str(self._path),
            map_size=self._map_size,
            readonly=True,
            lock=False,
        )

        with self._env.begin() as txn:
            cursor = txn.cursor()
            self._keys = [key for key, _ in cursor]
        self._cached_transitions = None
        self._cached_terminal_gap_s = None

        _log.info(
            "offline_dataset_opened",
            path=str(self._path),
            n_records=len(self._keys),
        )

    def close(self) -> None:
        """Close LMDB environment."""
        if self._env is not None:
            self._env.close()
            self._env = None
            self._keys = []
            self._cached_transitions = None
            self._cached_terminal_gap_s = None

    def __len__(self) -> int:
        """Return the number of transition pairs in the dataset."""
        return max(len(self._keys) - 1, 0)

    def _record_to_state(self, record: MouseDroidExperienceRecord) -> NDArray[np.floating[Any]]:
        """Concatenate record fields into a flat state vector.

        Args:
            record: Experience record.

        Returns:
            1-D float32 array of shape ``(state_dim,)``.
        """
        result: NDArray[np.floating[Any]] = np.concatenate([
            record.vision_features.flatten()[:self._vision_dim],
            np.array([record.distance_m], dtype=np.float32),
            record.motor_state.flatten()[:self._motor_dim],
        ]).astype(np.float32)
        return result

    def _load_record(self, key: bytes) -> MouseDroidExperienceRecord | None:
        """Load a single record from LMDB by key."""
        if self._env is None:
            return None
        with self._env.begin() as txn:
            data = txn.get(key)
            if data is None:
                return None
            return MouseDroidExperienceRecord.deserialize(data)

    def get_transitions(
        self,
        terminal_gap_s: float = 5.0,
    ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]:
        """Load all transitions as numpy arrays.

        Consecutive records form ``(s, a, r, s')`` pairs. The ``done`` flag
        is inferred from timestamp gaps exceeding ``terminal_gap_s``.

        Args:
            terminal_gap_s: Timestamp gap (seconds) to mark episode boundary.

        Returns:
            Tuple of ``(states, actions, rewards, next_states, dones)``
            with shapes ``(N, state_dim)``, ``(N, action_dim)``,
            ``(N,)``, ``(N, state_dim)``, ``(N,)``.
        """
        if not self._keys:
            empty_s = np.zeros((0, self.state_dim), dtype=np.float32)
            empty_a = np.zeros((0, self._action_dim), dtype=np.float32)
            empty_r = np.zeros(0, dtype=np.float32)
            empty_d = np.zeros(0, dtype=np.float32)
            return empty_s, empty_a, empty_r, empty_s.copy(), empty_d

        records: list[MouseDroidExperienceRecord] = []
        for key in self._keys:
            rec = self._load_record(key)
            if rec is not None:
                records.append(rec)

        if len(records) < 2:
            empty_s = np.zeros((0, self.state_dim), dtype=np.float32)
            empty_a = np.zeros((0, self._action_dim), dtype=np.float32)
            empty_r = np.zeros(0, dtype=np.float32)
            empty_d = np.zeros(0, dtype=np.float32)
            return empty_s, empty_a, empty_r, empty_s.copy(), empty_d

        n_transitions = len(records) - 1
        states = np.zeros((n_transitions, self.state_dim), dtype=np.float32)
        actions = np.zeros((n_transitions, self._action_dim), dtype=np.float32)
        rewards = np.zeros(n_transitions, dtype=np.float32)
        next_states = np.zeros((n_transitions, self.state_dim), dtype=np.float32)
        dones = np.zeros(n_transitions, dtype=np.float32)

        for i in range(n_transitions):
            states[i] = self._record_to_state(records[i])
            actions[i] = records[i].action.flatten()[:self._action_dim]
            rewards[i] = records[i].reward
            next_states[i] = self._record_to_state(records[i + 1])

            time_gap = abs(records[i + 1].timestamp - records[i].timestamp)
            if time_gap > terminal_gap_s:
                dones[i] = 1.0

        _log.info(
            "transitions_loaded",
            n_transitions=n_transitions,
            n_episodes=int(dones.sum()) + 1,
        )

        return states, actions, rewards, next_states, dones

    def iterate_batches(
        self,
        batch_size: int,
        terminal_gap_s: float = 5.0,
        shuffle: bool = True,
        seed: int | None = None,
    ) -> Iterator[dict[str, Tensor]]:
        """Yield batches of transitions as PyTorch tensors.

        Args:
            batch_size: Number of transitions per batch.
            terminal_gap_s: Timestamp gap to mark episode boundaries.
            shuffle: Shuffle transitions before batching.
            seed: Random seed for reproducibility.

        Yields:
            Dict with keys ``states``, ``actions``, ``rewards``,
            ``next_states``, ``dones`` — each a tensor on ``self._device``.
        """
        cache_miss = (
            self._cached_transitions is None
            or self._cached_terminal_gap_s != terminal_gap_s
        )
        if cache_miss:
            self._cached_transitions = self.get_transitions(
                terminal_gap_s=terminal_gap_s,
            )
            self._cached_terminal_gap_s = terminal_gap_s

        if self._cached_transitions is None:
            return

        states, actions, rewards, next_states, dones = self._cached_transitions

        n = len(states)
        if n == 0:
            return

        rng = np.random.default_rng(seed)
        indices = np.arange(n)
        if shuffle:
            rng.shuffle(indices)

        for start in range(0, n, batch_size):
            batch_idx = indices[start : start + batch_size]
            yield {
                "states": torch.as_tensor(states[batch_idx], device=self._device),
                "actions": torch.as_tensor(actions[batch_idx], device=self._device),
                "rewards": torch.as_tensor(rewards[batch_idx], device=self._device),
                "next_states": torch.as_tensor(
                    next_states[batch_idx], device=self._device,
                ),
                "dones": torch.as_tensor(dones[batch_idx], device=self._device),
            }
