"""Shared tiny-RSSM fixtures for the F-023 drift/corrupted-training tests.

Hoisted out of ``tests/unit/world_model/test_train_sequence_corrupted.py``,
``tests/unit/training/test_drift_metrics.py`` and
``tests/unit/training/test_drift_reduction.py``, which previously each carried a
byte-identical ``_tiny_cfg`` / ``_seq_batch`` / ``_model_pair`` copy that would
drift. Importable via ``from tests.unit._rssm_drift_helpers import ...`` (the
``tests.regression._rssm_golden_helper`` pattern).
"""

from __future__ import annotations

import torch

from mousedroid.config.schema import ModelConfig
from mousedroid.constants import SENSOR_SLOT_MAP
from mousedroid.world_model.rssm import RSSM, RawModalityDecoders


def tiny_rssm_cfg() -> ModelConfig:
    """Tiny vision-OFF, ultrasonic-ON RSSM config used across the drift tests."""
    return ModelConfig.model_validate(
        {
            "vision_dim": 0,
            "vision_proj_dim": 0,
            "ultrasonic_dim": 1,
            "motor_state_dim": 4,
            "hidden_dim": 16,
            "latent_dim": 8,
            "action_dim": 3,
            "obs_dim": 16,
            "ultrasonic_proj_dim": 4,
            "motor_proj_dim": 8,
        }
    )


def seq_batch(
    mcfg: ModelConfig, *, episodes: int, seq_len: int, seed: int = 0
) -> dict[str, torch.Tensor]:
    """Seeded ``(B, T, ...)`` batch with motor + ultrasonic valid-mask slots set."""
    gen = torch.Generator().manual_seed(seed)
    valid = torch.zeros(episodes, seq_len, len(SENSOR_SLOT_MAP))
    valid[..., SENSOR_SLOT_MAP["motor"]] = 1.0
    valid[..., SENSOR_SLOT_MAP["ultrasonic"]] = 1.0
    return {
        "motor": torch.randn(episodes, seq_len, mcfg.motor_state_dim, generator=gen),
        "ultrasonic": torch.randn(episodes, seq_len, mcfg.ultrasonic_dim, generator=gen),
        "valid_mask": valid,
        "action": torch.randn(episodes, seq_len, mcfg.action_dim, generator=gen),
    }


def seeded_model_pair(seed: int = 7) -> tuple[RSSM, RawModalityDecoders]:
    """An (RSSM, decoders) pair with byte-identical seeded inits (CPU)."""
    mcfg = tiny_rssm_cfg()
    torch.manual_seed(seed)
    model = RSSM(mcfg)
    torch.manual_seed(seed)
    decoders = RawModalityDecoders(mcfg)
    return model, decoders
