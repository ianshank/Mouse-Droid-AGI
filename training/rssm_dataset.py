"""PyTorch Dataset wrapping saved synthetic sequences for RSSM training."""

from __future__ import annotations

import math
from pathlib import Path
from random import Random

import structlog
import torch
from torch import Tensor
from torch.utils.data import Dataset

from mousedroid.config.schema import ExperienceConfig, ModelConfig, TrainingReplayConfig
from mousedroid.constants import SENSOR_SLOT_MAP
from mousedroid.experience.dataset import OfflineRLDataset
from mousedroid.experience.record import MouseDroidExperienceRecord

_log = structlog.get_logger(__name__)

SequenceBatch = dict[str, Tensor]


def _coerce_step_tensor(value: object | None, dim: int) -> Tensor:
    """Coerce one replay/synthetic field to a fixed-length float tensor."""
    if dim <= 0:
        return torch.zeros(0, dtype=torch.float32)

    if value is None:
        source = torch.zeros(0, dtype=torch.float32)
    else:
        source = torch.as_tensor(value, dtype=torch.float32).flatten()

    result = torch.zeros(dim, dtype=torch.float32)
    n_values = min(dim, int(source.numel()))
    if n_values > 0:
        result[:n_values] = source[:n_values]
    return result


def _coerce_valid_mask(value: object | None) -> Tensor:
    """Coerce a valid-mask to the stable encoder slot layout."""
    return _coerce_step_tensor(value, len(SENSOR_SLOT_MAP))


def _normalize_episode(
    episode: list[dict[str, Tensor]],
    model_cfg: ModelConfig,
) -> list[dict[str, Tensor]]:
    """Normalize episode tensors to the configured model dimensions."""
    normalized: list[dict[str, Tensor]] = []
    for step in episode:
        normalized.append(
            {
                "vision": _coerce_step_tensor(step.get("vision"), model_cfg.vision_dim),
                "ultrasonic": _coerce_step_tensor(
                    step.get("ultrasonic"), model_cfg.ultrasonic_dim
                ),
                "motor_state": _coerce_step_tensor(
                    step.get("motor_state"), model_cfg.motor_state_dim
                ),
                "valid_mask": _coerce_valid_mask(step.get("valid_mask")),
                "lidar": _coerce_step_tensor(step.get("lidar"), model_cfg.lidar_dim),
                "action": _coerce_step_tensor(step.get("action"), model_cfg.action_dim),
            }
        )
    return normalized


def _record_to_episode_step(record: MouseDroidExperienceRecord, model_cfg: ModelConfig) -> dict[str, Tensor]:
    """Convert one LMDB experience record to the RSSM episode-step contract."""
    valid_mask = torch.zeros(len(SENSOR_SLOT_MAP), dtype=torch.float32)
    valid_mask[SENSOR_SLOT_MAP["vision"]] = 1.0
    valid_mask[SENSOR_SLOT_MAP["motor"]] = 1.0
    if model_cfg.ultrasonic_dim > 0:
        valid_mask[SENSOR_SLOT_MAP["ultrasonic"]] = 1.0

    return {
        "vision": _coerce_step_tensor(record.vision_features, model_cfg.vision_dim),
        "ultrasonic": _coerce_step_tensor([record.distance_m], model_cfg.ultrasonic_dim),
        "motor_state": _coerce_step_tensor(record.motor_state, model_cfg.motor_state_dim),
        "valid_mask": valid_mask,
        "lidar": torch.zeros(model_cfg.lidar_dim, dtype=torch.float32),
        "action": _coerce_step_tensor(record.action, model_cfg.action_dim),
    }


def _load_replay_episodes(
    replay_cfg: TrainingReplayConfig,
    experience_cfg: ExperienceConfig,
    model_cfg: ModelConfig,
) -> list[list[dict[str, Tensor]]]:
    """Load replay episodes from LMDB and convert them to RSSM sequence steps."""
    dataset = OfflineRLDataset(experience_cfg, model_cfg)
    dataset.open()
    try:
        episodes = dataset.get_episodes(terminal_gap_s=replay_cfg.terminal_gap_s)
    finally:
        dataset.close()

    return [
        [_record_to_episode_step(record, model_cfg) for record in episode]
        for episode in episodes
        if episode
    ]


def _select_replay_subset(
    replay_episodes: list[list[dict[str, Tensor]]],
    synthetic_count: int,
    replay_cfg: TrainingReplayConfig,
) -> list[list[dict[str, Tensor]]]:
    """Select the replay subset implied by the configured sim:real ratio."""
    if replay_cfg.max_real_episodes is not None:
        replay_episodes = replay_episodes[: replay_cfg.max_real_episodes]

    if not replay_episodes:
        return []

    if synthetic_count == 0:
        return replay_episodes

    if replay_cfg.real_episode_ratio <= 0.0:
        return []

    target_count = min(
        len(replay_episodes),
        max(1, math.ceil(synthetic_count * replay_cfg.real_episode_ratio)),
    )
    selected = list(replay_episodes)
    if replay_cfg.seed is not None:
        Random(replay_cfg.seed).shuffle(selected)
    return selected[:target_count]


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

    def __init__(
        self,
        data_path: Path | str | None,
        seq_len: int = 50,
        *,
        replay_cfg: TrainingReplayConfig | None = None,
        experience_cfg: ExperienceConfig | None = None,
        model_cfg: ModelConfig | None = None,
    ) -> None:
        data_file = Path(data_path) if data_path is not None else None
        synthetic_episodes: list[list[dict[str, Tensor]]] = []
        if data_file is not None and data_file.exists():
            synthetic_episodes = torch.load(
                data_file,
                weights_only=False,
            )
            if model_cfg is not None:
                synthetic_episodes = [
                    _normalize_episode(episode, model_cfg) for episode in synthetic_episodes
                ]

        replay_episodes: list[list[dict[str, Tensor]]] = []
        if replay_cfg is not None and replay_cfg.enabled:
            if experience_cfg is None or model_cfg is None:
                msg = "Replay ingestion requires both experience_cfg and model_cfg"
                raise ValueError(msg)

            replay_source = experience_cfg.model_copy(
                update={"path": replay_cfg.source_path or experience_cfg.path}
            )
            replay_episodes = _load_replay_episodes(replay_cfg, replay_source, model_cfg)

        self._episodes = synthetic_episodes + _select_replay_subset(
            replay_episodes,
            synthetic_count=len(synthetic_episodes),
            replay_cfg=replay_cfg or TrainingReplayConfig(),
        )
        self._seq_len = seq_len
        if not self._episodes:
            if replay_cfg is not None and replay_cfg.enabled:
                msg = (
                    "No RSSM replay or synthetic episodes available. "
                    f"data_path={data_file}, replay_path="
                    f"{(replay_cfg.source_path or experience_cfg.path) if experience_cfg else None}"
                )
            else:
                msg = f"RSSM training data file not found: {data_file}"
            raise FileNotFoundError(msg)

        _log.info(
            "dataset_loaded",
            path=str(data_file) if data_file is not None else None,
            n_episodes=len(self._episodes),
            synthetic_episodes=len(synthetic_episodes),
            replay_episodes=len(replay_episodes),
            replay_enabled=replay_cfg.enabled if replay_cfg is not None else False,
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
